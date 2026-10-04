"""Pi0 Multi-Agent Attention model (PyTorch version) with per-agent selective attention.

Each agent's action tokens only attend to:
- Global view image tokens
- Their own wrist view image tokens  
- Their own prompt portion (e.g., "agent0: Wait for the other robots")

They do NOT attend to other agents' wrist view tokens.
"""

import dataclasses
import logging
import math
import re

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


logger = logging.getLogger("openpi")


@dataclasses.dataclass
class TokenBoundaries:
    """Tracks token positions for selective attention (PyTorch version).
    
    Suffix structure (for 2 agents, horizon=50):
        [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
        Total: num_agents + num_agents * action_horizon tokens
    """
    image_boundaries: dict[str, tuple[int, int]]
    prompt_boundaries: dict[int, tuple[int, int]]
    global_image_range: tuple[int, int]
    prefix_length: int
    language_range: tuple[int, int]
    # Agent ID -> state token index in suffix
    state_token_indices: dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of state tokens (one per agent)
    num_state_tokens: int = 1
    # Agent ID -> (start_idx, end_idx) for that agent's action tokens in suffix
    action_token_boundaries: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)
    # Action horizon per agent
    action_horizon_per_agent: int = 50


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Standard attention mask creation."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_per_agent_attn_mask_pytorch(
    input_mask: torch.Tensor,  # [B, total_seq_len]
    ar_mask: torch.Tensor,     # [B, total_seq_len]
    boundaries: TokenBoundaries,
    num_agents: int,
    action_horizon: int,
    agent_to_wrist_image: dict[int, str],
    actions_per_agent: int | None = None,
) -> torch.Tensor:
    """Create per-agent attention mask for decoupled action tokens (PyTorch version).
    
    Suffix structure (for 2 agents, horizon=50):
        [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
    
    Each agent's action tokens only attend to:
    - Global image tokens
    - Their own wrist image tokens
    - Their own prompt portion
    - Their own state token (NOT other agents' state tokens)
    - Their own action tokens (causally)
    
    Returns:
        Attention mask [B, total_seq_len, total_seq_len]
    """
    batch_size = input_mask.shape[0]
    total_len = input_mask.shape[1]
    device = input_mask.device
    
    # Start with standard attention mask
    base_mask = make_att_2d_masks(input_mask, ar_mask)
    
    # Calculate positions
    prefix_len = boundaries.prefix_length
    num_state_tokens = boundaries.num_state_tokens
    
    if actions_per_agent is None:
        actions_per_agent = action_horizon // num_agents
    
    # For each agent, modify attention for their action tokens
    for agent_id in range(num_agents):
        # Get this agent's action token range from boundaries
        if agent_id in boundaries.action_token_boundaries:
            agent_action_start_in_suffix, agent_action_end_in_suffix = boundaries.action_token_boundaries[agent_id]
            agent_action_start = prefix_len + agent_action_start_in_suffix
            agent_action_end = prefix_len + agent_action_end_in_suffix
        else:
            # Fallback for old structure
            action_start = prefix_len + num_state_tokens
            agent_action_start = action_start + agent_id * actions_per_agent
            agent_action_end = agent_action_start + actions_per_agent
        
        if agent_action_end > total_len:
            agent_action_end = total_len
            if agent_action_start >= total_len:
                continue
        
        # Create allowed key mask for this agent
        allowed_key_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        
        # 1. Allow global image
        global_start, global_end = boundaries.global_image_range
        allowed_key_mask[global_start:global_end] = True
        
        # 2. Allow own wrist image
        if agent_id in agent_to_wrist_image:
            wrist_image_name = agent_to_wrist_image[agent_id]
            if wrist_image_name in boundaries.image_boundaries:
                wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
                allowed_key_mask[wrist_start:wrist_end] = True
        
        # 3. Allow own prompt or all language
        if agent_id in boundaries.prompt_boundaries:
            prompt_start, prompt_end = boundaries.prompt_boundaries[agent_id]
            allowed_key_mask[prompt_start:prompt_end] = True
        else:
            lang_start, lang_end = boundaries.language_range
            allowed_key_mask[lang_start:lang_end] = True
        
        # 4. Allow ONLY this agent's own state token
        if agent_id in boundaries.state_token_indices:
            agent_state_idx = prefix_len + boundaries.state_token_indices[agent_id]
            allowed_key_mask[agent_state_idx] = True
        else:
            # Fallback: allow all state tokens
            for state_idx in range(num_state_tokens):
                allowed_key_mask[prefix_len + state_idx] = True
        
        # 5. Allow own action tokens
        allowed_key_mask[agent_action_start:agent_action_end] = True
        
        # Apply mask to this agent's action queries
        for query_idx in range(agent_action_start, agent_action_end):
            if query_idx >= total_len:
                break
            
            query_mask = allowed_key_mask.clone()
            
            # Apply causality for action tokens
            for key_idx in range(agent_action_start, agent_action_end):
                if key_idx > query_idx:
                    query_mask[key_idx] = False
            
            # Respect padding
            query_mask = query_mask & input_mask[0]
            
            # Update mask
            base_mask[:, query_idx, :] = query_mask.unsqueeze(0).expand(batch_size, -1)
    
    # Also update attention for state tokens: each sees only its agent's stuff
    for agent_id in range(num_agents):
        if agent_id not in boundaries.state_token_indices:
            continue
        
        state_query_idx = prefix_len + boundaries.state_token_indices[agent_id]
        if state_query_idx >= total_len:
            continue
        
        allowed_key_mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        
        # Global image
        global_start, global_end = boundaries.global_image_range
        allowed_key_mask[global_start:global_end] = True
        
        # Own wrist image
        if agent_id in agent_to_wrist_image:
            wrist_image_name = agent_to_wrist_image[agent_id]
            if wrist_image_name in boundaries.image_boundaries:
                wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
                allowed_key_mask[wrist_start:wrist_end] = True
        
        # Own prompt or all language
        if agent_id in boundaries.prompt_boundaries:
            prompt_start, prompt_end = boundaries.prompt_boundaries[agent_id]
            allowed_key_mask[prompt_start:prompt_end] = True
        else:
            lang_start, lang_end = boundaries.language_range
            allowed_key_mask[lang_start:lang_end] = True
        
        # Own state token (self)
        allowed_key_mask[state_query_idx] = True
        
        # Respect padding
        allowed_key_mask = allowed_key_mask & input_mask[0]
        
        base_mask[:, state_query_idx, :] = allowed_key_mask.unsqueeze(0).expand(batch_size, -1)
    
    return base_mask


class PI0MultiAgentAttnPytorch(nn.Module):
    """Pi0 model with per-agent selective attention (PyTorch version)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = getattr(config, 'pi05', False)
        
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # Action processing layers - SHARED across all agents (same as original Pi0)
        # This allows loading pretrained Pi0 weights directly
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # Per-agent state projections (these are new, randomly initialized)
            self.state_projs = nn.ModuleDict({
                f"agent{i}": nn.Linear(config.action_dim_per_agent, action_expert_config.width)
                for i in range(config.num_agents)
            })
            # Shared action time MLP (same as original Pi0, can load pretrained weights)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Multi-agent config
        self.num_agents = config.num_agents
        self.action_dim_per_agent = config.action_dim_per_agent
        self.action_horizon = config.action_horizon
        self._agent_to_wrist_image = dict(config.agent_to_wrist_image)
        self.global_image_key = config.global_image_key
        
        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly."
        try:
            from transformers.models.siglip import check
            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for PI0MultiAgentAttnPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for PI0MultiAgentAttnPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            list(observation.images.keys()),  # Return image keys for boundary tracking
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix_with_boundaries(
        self, images, img_masks, lang_tokens, lang_masks, image_keys
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TokenBoundaries]:
        """Embed images and language with boundary tracking."""
        embs = []
        pad_masks = []
        att_masks = []
        
        image_boundaries = {}
        global_image_range = (0, 0)
        current_pos = 0

        # Process images
        for img, img_mask, img_key in zip(images, img_masks, image_keys, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            
            start_pos = current_pos
            end_pos = current_pos + num_img_embs
            
            # Track image boundaries
            if img_key == self.global_image_key or "base" in img_key or "global" in img_key:
                global_image_range = (start_pos, end_pos)
            else:
                image_boundaries[img_key] = (start_pos, end_pos)
            
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
            current_pos = end_pos

        # Process language tokens
        language_start = current_pos
        
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        current_pos += num_lang_embs
        language_end = current_pos

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        boundaries = TokenBoundaries(
            image_boundaries=image_boundaries,
            prompt_boundaries={},
            global_image_range=global_image_range,
            prefix_length=current_pos,
            language_range=(language_start, language_end),
        )

        return embs, pad_masks, att_masks, boundaries

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed per-agent action tokens (and state tokens for pi0).
        
        Pi0 suffix (for 2 agents, horizon=50):
            [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
            Total: num_agents + num_agents * action_horizon tokens
        
        Pi0.5 suffix (for 2 agents, horizon=50):
            [action_0_t0...action_0_t49, action_1_t0...action_1_t49]
            Total: num_agents * action_horizon tokens (no state tokens)
        
        Returns:
            embs: Embedded tokens
            pad_masks: Padding masks
            att_masks: Attention masks
            adarms_cond: AdaRMS conditioning (for pi05, None for pi0)
            state_token_indices: Mapping from agent_id to state token index in suffix
            action_token_boundaries: Mapping from agent_id to (start, end) in suffix
        """
        embs = []
        pad_masks = []
        att_masks = []
        state_token_indices = {}
        action_token_boundaries = {}
        bsize = state.shape[0]
        device = state.device
        current_suffix_idx = 0

        if self.pi05:
            # Pi0.5: no state tokens, uses AdaRMS for time conditioning
            # Still create per-agent action token blocks for multi-agent attention masking
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
            )
            time_emb = time_emb.type(dtype=timestep.dtype)
            
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)
            
            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            adarms_cond = time_emb
            
            for agent_id in range(self.num_agents):
                if self.action_in_proj.weight.dtype == torch.float32:
                    noisy_actions_cast = noisy_actions.to(torch.float32)
                else:
                    noisy_actions_cast = noisy_actions
                
                def action_proj_func(actions):
                    return self.action_in_proj(actions)
                
                agent_action_emb = self._apply_checkpoint(action_proj_func, noisy_actions_cast)
                
                embs.append(agent_action_emb)
                pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
                att_masks += [1] + ([0] * (self.action_horizon - 1))
                
                action_start = current_suffix_idx
                action_end = current_suffix_idx + self.action_horizon
                action_token_boundaries[agent_id] = (action_start, action_end)
                current_suffix_idx = action_end
        else:
            # Pi0: per-agent state tokens + per-agent action tokens
            
            # 1. Create per-agent state tokens
            for agent_id in range(self.num_agents):
                start_dim = agent_id * self.action_dim_per_agent
                end_dim = start_dim + self.action_dim_per_agent
                agent_state = state[:, start_dim:end_dim]
                
                state_proj = self.state_projs[f"agent{agent_id}"]
                if state_proj.weight.dtype == torch.float32:
                    agent_state = agent_state.to(torch.float32)
                
                def state_proj_func(s, proj=state_proj):
                    return proj(s)
                
                agent_state_emb = self._apply_checkpoint(state_proj_func, agent_state)
                
                embs.append(agent_state_emb[:, None, :])
                pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
                att_masks += [1]
                
                state_token_indices[agent_id] = current_suffix_idx
                current_suffix_idx += 1
            
            # 2. Create per-agent action tokens using SHARED projection
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
            )
            time_emb = time_emb.type(dtype=timestep.dtype)
            
            for agent_id in range(self.num_agents):
                if self.action_in_proj.weight.dtype == torch.float32:
                    noisy_actions_cast = noisy_actions.to(torch.float32)
                else:
                    noisy_actions_cast = noisy_actions
                
                def action_proj_func(actions):
                    return self.action_in_proj(actions)
                
                agent_action_emb = self._apply_checkpoint(action_proj_func, noisy_actions_cast)
                
                time_tokens = time_emb[:, None, :].expand(-1, self.action_horizon, -1)
                action_time_tokens = torch.cat([agent_action_emb, time_tokens], dim=-1)
                
                def mlp_func(x):
                    x = self.action_time_mlp_in(x)
                    x = F.silu(x)
                    return self.action_time_mlp_out(x)
                
                agent_action_tokens = self._apply_checkpoint(mlp_func, action_time_tokens)
                
                embs.append(agent_action_tokens)
                pad_masks.append(torch.ones(bsize, self.action_horizon, dtype=torch.bool, device=device))
                att_masks += [1] + ([0] * (self.action_horizon - 1))
                
                action_start = current_suffix_idx
                action_end = current_suffix_idx + self.action_horizon
                action_token_boundaries[agent_id] = (action_start, action_end)
                current_suffix_idx = action_end
            
            adarms_cond = None

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond, state_token_indices, action_token_boundaries

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Training forward pass with per-agent attention masking."""
        images, img_masks, lang_tokens, lang_masks, state, image_keys = self._preprocess_observation(
            observation, train=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed with boundary tracking
        prefix_embs, prefix_pad_masks, prefix_att_masks, boundaries = self.embed_prefix_with_boundaries(
            images, img_masks, lang_tokens, lang_masks, image_keys
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, state_token_indices, action_token_boundaries = self.embed_suffix(state, x_t, time)
        
        # Update boundaries with state token and action token info
        boundaries.state_token_indices = state_token_indices
        boundaries.num_state_tokens = self.num_agents if not self.pi05 else 0
        boundaries.action_token_boundaries = action_token_boundaries
        boundaries.action_horizon_per_agent = self.action_horizon
        
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # Create per-agent attention mask
        att_2d_masks = make_per_agent_attn_mask_pytorch(
            pad_masks,
            att_masks,
            boundaries,
            num_agents=self.num_agents,
            action_horizon=self.action_horizon * self.num_agents,  # Total action tokens
            agent_to_wrist_image=self._agent_to_wrist_image,
            actions_per_agent=self.action_horizon,  # Each agent has action_horizon tokens
        )
        
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # Decode per-agent action outputs
        v_t = self._decode_action_outputs(suffix_out, boundaries)

        return F.mse_loss(u_t, v_t, reduction="none")
    
    def _decode_action_outputs(
        self, 
        suffix_out: torch.Tensor, 
        boundaries: TokenBoundaries
    ) -> torch.Tensor:
        """Decode per-agent action token outputs back to [B, horizon, action_dim].
        
        Uses SHARED action_out_proj for each agent. Works for both pi0 and pi0.5.
        Each agent takes action_dim_per_agent dims from their projected output,
        and the last agent fills the remaining dims to reach full action_dim.
        
        Args:
            suffix_out: Transformer output for suffix [B, suffix_len, width]
            boundaries: Token boundaries with action_token_boundaries
        
        Returns:
            v_t: Velocity field [B, action_horizon, action_dim]
        """
        agent_velocities = []
        num_agents = self.num_agents
        dim_per_agent = self.action_dim_per_agent
        
        for agent_id in range(num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                raise ValueError(f"Missing action_token_boundaries for agent {agent_id}")
            
            action_start, action_end = boundaries.action_token_boundaries[agent_id]
            agent_action_out = suffix_out[:, action_start:action_end, :]
            agent_action_out = agent_action_out.to(dtype=torch.float32)
            
            def proj_func(x):
                return self.action_out_proj(x)
            
            agent_v_full = self._apply_checkpoint(proj_func, agent_action_out)
            
            if agent_id < num_agents - 1:
                agent_v = agent_v_full[:, :, :dim_per_agent]
            else:
                remaining_dims = self.config.action_dim - agent_id * dim_per_agent
                agent_v = agent_v_full[:, :, :remaining_dims]
            
            agent_velocities.append(agent_v)
        
        v_t = torch.cat(agent_velocities, dim=-1)
        
        return v_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Inference with per-agent attention masking."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, image_keys = self._preprocess_observation(
            observation, train=False
        )

        # Embed prefix with boundaries
        prefix_embs, prefix_pad_masks, prefix_att_masks, boundaries = self.embed_prefix_with_boundaries(
            images, img_masks, lang_tokens, lang_masks, image_keys
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute prefix KV cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, state_token_indices, action_token_boundaries = self.embed_suffix(
                state, x_t, expanded_time
            )
            
            # Update boundaries with state token and action token info
            boundaries.state_token_indices = state_token_indices
            boundaries.num_state_tokens = self.num_agents if not self.pi05 else 0
            boundaries.action_token_boundaries = action_token_boundaries
            boundaries.action_horizon_per_agent = self.action_horizon
            
            v_t = self.denoise_step_with_boundaries(
                suffix_embs,
                suffix_pad_masks,
                suffix_att_masks,
                prefix_pad_masks,
                past_key_values,
                boundaries,
                adarms_cond,
            )
            x_t = x_t + dt * v_t
            time += dt
            
        return x_t

    def denoise_step_with_boundaries(
        self,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        prefix_pad_masks,
        past_key_values,
        boundaries: TokenBoundaries,
        adarms_cond,
    ):
        """Apply one denoising step with per-agent attention masking."""
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Create cross-attention mask with per-agent masking
        cross_attn_mask = self._make_cross_attn_mask_pytorch(
            prefix_pad_masks, suffix_pad_masks, boundaries, suffix_len
        )
        
        # Create suffix self-attention mask with per-agent masking
        suffix_att_2d_masks = self._make_suffix_self_attn_mask_pytorch(
            suffix_pad_masks, suffix_att_masks, boundaries, suffix_len
        )
        full_att_2d_masks = torch.cat([cross_attn_mask, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        
        # Decode per-agent action outputs
        v_t = self._decode_action_outputs(suffix_out, boundaries)
        return v_t
    
    def _make_suffix_self_attn_mask_pytorch(
        self,
        suffix_mask: torch.Tensor,
        suffix_ar_mask: torch.Tensor,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> torch.Tensor:
        """Create suffix self-attention mask with per-agent decoupled masking.
        
        Pi0 suffix:  [state_0, state_1, action_0_t0...t49, action_1_t0...t49]
        Pi0.5 suffix: [action_0_t0...t49, action_1_t0...t49] (no state tokens)
        """
        batch_size = suffix_mask.shape[0]
        device = suffix_mask.device
        
        base_mask = make_att_2d_masks(suffix_mask, suffix_ar_mask)
        
        for agent_id in range(self.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
            
            agent_action_start, agent_action_end = boundaries.action_token_boundaries[agent_id]
            
            if agent_action_end > suffix_len:
                agent_action_end = suffix_len
            if agent_action_start >= suffix_len:
                continue
            
            for query_idx in range(agent_action_start, agent_action_end):
                if query_idx >= suffix_len:
                    break
                
                allowed_mask = torch.zeros(suffix_len, dtype=torch.bool, device=device)
                
                # Allow own state token if present (pi0 only)
                if agent_id in boundaries.state_token_indices:
                    agent_state_idx = boundaries.state_token_indices[agent_id]
                    allowed_mask[agent_state_idx] = True
                
                # Allow own action tokens (causally)
                for key_idx in range(agent_action_start, query_idx + 1):
                    allowed_mask[key_idx] = True
                
                allowed_mask = allowed_mask & suffix_mask[0]
                base_mask[:, query_idx, :] = allowed_mask.unsqueeze(0).expand(batch_size, -1)
        
        # State token self-attention: each state only sees itself (pi0 only)
        for agent_id in range(self.num_agents):
            if agent_id not in boundaries.state_token_indices:
                continue
            
            state_idx = boundaries.state_token_indices[agent_id]
            if state_idx >= suffix_len:
                continue
            
            allowed_mask = torch.zeros(suffix_len, dtype=torch.bool, device=device)
            allowed_mask[state_idx] = True
            allowed_mask = allowed_mask & suffix_mask[0]
            base_mask[:, state_idx, :] = allowed_mask.unsqueeze(0).expand(batch_size, -1)
        
        return base_mask

    def _make_cross_attn_mask_pytorch(
        self,
        prefix_mask: torch.Tensor,
        suffix_mask: torch.Tensor,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> torch.Tensor:
        """Create cross-attention mask for suffix attending to prefix with per-agent masking.
        
        Pi0 suffix:  [state_0, state_1, action_0_t0...t49, action_1_t0...t49]
        Pi0.5 suffix: [action_0_t0...t49, action_1_t0...t49] (no state tokens)
        """
        batch_size = prefix_mask.shape[0]
        prefix_len = prefix_mask.shape[1]
        device = prefix_mask.device
        
        # Start with full cross attention
        base_cross_mask = prefix_mask[:, None, :].expand(batch_size, suffix_len, prefix_len).clone()
        
        def create_agent_prefix_mask(agent_id):
            """Create allowed prefix mask for a given agent."""
            allowed_prefix_mask = torch.zeros(prefix_len, dtype=torch.bool, device=device)
            
            # Allow global image
            global_start, global_end = boundaries.global_image_range
            allowed_prefix_mask[global_start:global_end] = True
            
            # Allow own wrist image
            if agent_id in self._agent_to_wrist_image:
                wrist_name = self._agent_to_wrist_image[agent_id]
                if wrist_name in boundaries.image_boundaries:
                    wrist_start, wrist_end = boundaries.image_boundaries[wrist_name]
                    allowed_prefix_mask[wrist_start:wrist_end] = True
            
            # Allow language tokens
            if agent_id in boundaries.prompt_boundaries:
                prompt_start, prompt_end = boundaries.prompt_boundaries[agent_id]
                allowed_prefix_mask[prompt_start:prompt_end] = True
            else:
                lang_start, lang_end = boundaries.language_range
                allowed_prefix_mask[lang_start:lang_end] = True
            
            return allowed_prefix_mask
        
        # Apply per-agent masking for state tokens (cross-attention to prefix)
        for agent_id in range(self.num_agents):
            if agent_id not in boundaries.state_token_indices:
                continue
            
            state_idx = boundaries.state_token_indices[agent_id]
            if state_idx >= suffix_len:
                continue
            
            allowed_prefix_mask = create_agent_prefix_mask(agent_id)
            combined_mask = allowed_prefix_mask & prefix_mask[0]
            base_cross_mask[:, state_idx, :] = combined_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Apply per-agent masking for action tokens using action_token_boundaries
        for agent_id in range(self.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
                
            agent_action_start, agent_action_end = boundaries.action_token_boundaries[agent_id]
            
            if agent_action_end > suffix_len:
                agent_action_end = suffix_len
            if agent_action_start >= suffix_len:
                continue
            
            allowed_prefix_mask = create_agent_prefix_mask(agent_id)
            
            # Apply to this agent's action positions
            for suffix_idx in range(agent_action_start, agent_action_end):
                combined_mask = allowed_prefix_mask & prefix_mask[0]
                base_cross_mask[:, suffix_idx, :] = combined_mask.unsqueeze(0).expand(batch_size, -1)
        
        return base_cross_mask




