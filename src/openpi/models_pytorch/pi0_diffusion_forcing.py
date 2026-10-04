"""
PI0 with Diffusion Forcing for Multi-Arm Robustness

Core Insight:
=============
Diffusion Forcing's key idea: different parts of the sequence can have different noise levels.
This teaches the model to not over-rely on any single part.

Challenge:
==========
Original Diffusion Forcing uses causal generation (token by token).
PI0's Flow Matching generates all actions in parallel (non-causal).

Solution:
=========
Even without causal attention, we can apply per-arm noise levels:
1. Each arm gets its own time step t_i (noise level)
2. Noisy actions: x_t[arm_i] = t_i * noise[arm_i] + (1-t_i) * actions[arm_i]
3. Target velocity: u_t = noise - actions (same for all, independent of t)
4. Model sees mixed-noise input, learns to be robust

Key Mathematical Insight:
========================
In Flow Matching, the target velocity u_t = noise - actions is INDEPENDENT of t.
So even with different t_i per arm, we use the same loss target.
The model learns to predict velocity from partially-clean, partially-noisy observations.

This is EXACTLY what we want for multi-arm robustness:
- Model sees some arms clearly (low t_i)
- Model sees other arms noisy (high t_i)  
- Model must predict ALL arms' velocities correctly
- Result: model doesn't over-rely on any single arm
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


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


def sample_beta(alpha, beta, shape, device):
    """Sample from Beta distribution."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample(shape)


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from 1D masks using cumsum trick."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0DiffusionForcing(nn.Module):
    """
    PI0 with Diffusion Forcing: minimal modification to original PI0.
    
    Key changes from original PI0:
    1. Action noise: each arm can have a different noise level (time step)
    2. State noise (optional): add observation noise to other arms' states
    
    This teaches the model to be robust to noisy observations from other arms.
    Everything else (network structure, attention, inference) stays the same.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = getattr(config, 'pi05', False)

        # Multi-arm configuration (new parameters)
        self.num_arms = getattr(config, 'num_arms', 4)
        self.arm_action_dim = getattr(config, 'arm_action_dim', 8)
        
        # Diffusion Forcing parameters for ACTIONS
        # noise_offset: how much extra noise for non-focal arms (0 = original behavior)
        self.noise_offset = getattr(config, 'noise_offset', 0.3)
        # prob of using per-arm noise vs uniform noise
        self.per_arm_noise_prob = getattr(config, 'per_arm_noise_prob', 0.7)
        
        # State noise parameters (for observation robustness)
        # Whether to add noise to other arms' state observations
        self.add_state_noise = getattr(config, 'add_state_noise', True)
        # Noise std for other arms' states (0 = no noise)
        self.state_noise_std = getattr(config, 'state_noise_std', 0.05)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # === SAME AS ORIGINAL PI0 ===
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Disable torch.compile for inference stability
        # The while loop in sample_actions is not compatible with CUDA graph capture
        # self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly."
        try:
            from transformers.models.siglip import check
            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # ========== HELPER METHODS (SAME AS ORIGINAL) ==========

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        """Sample single time (original behavior)."""
        time_beta = sample_beta(1.5, 1.0, (bsize,), device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # ========== NEW: DIFFUSION FORCING METHODS ==========

    def sample_time_per_arm(self, bsize, device):
        """
        Sample per-arm time steps for Diffusion Forcing.
        
        Strategy:
        1. Sample a base time t from Beta distribution
        2. Randomly select a "focal" arm (the one we care most about)
        3. Focal arm uses base time t
        4. Other arms use t + offset (more noise, less reliable)
        
        Returns:
            time_per_arm: [B, num_arms] - time step for each arm
            focal_arm: [B] - index of focal arm for each sample
        """
        # Sample base time
        base_time = sample_beta(1.5, 1.0, (bsize,), device)
        base_time = base_time * 0.999 + 0.001  # [B]
        
        # Randomly select focal arm for each sample
        focal_arm = torch.randint(0, self.num_arms, (bsize,), device=device)  # [B]
        
        # Create per-arm times
        # Start with base_time + offset for all arms
        offset = torch.rand(bsize, self.num_arms, device=device) * self.noise_offset
        time_per_arm = base_time.unsqueeze(1) + offset  # [B, num_arms]
        
        # Focal arm uses base_time (no offset)
        focal_mask = F.one_hot(focal_arm, self.num_arms).bool()  # [B, num_arms]
        time_per_arm = torch.where(focal_mask, base_time.unsqueeze(1), time_per_arm)
        
        # Clamp to valid range
        time_per_arm = torch.clamp(time_per_arm, 0.001, 1.0)
        
        return time_per_arm, focal_arm

    def add_noise_to_state(self, state, focal_arm):
        """
        Add noise to non-focal arms' state observations.
        
        This simulates real-world scenarios where:
        - Own arm's state is accurate (from local sensors)
        - Other arms' states may be noisy (network delay, sensor noise, etc.)
        
        Args:
            state: [B, D] where D = num_arms * arm_action_dim
            focal_arm: [B] index of focal arm (no noise added)
            
        Returns:
            noisy_state: [B, D] with noise added to non-focal arms
        """
        if not self.add_state_noise or self.state_noise_std <= 0:
            return state
            
        B, D = state.shape
        device = state.device
        
        # Reshape to [B, num_arms, arm_dim]
        state_per_arm = state.view(B, self.num_arms, self.arm_action_dim)
        
        # Generate noise for all arms
        noise = torch.randn_like(state_per_arm) * self.state_noise_std
        
        # Create mask: focal arm gets no noise
        focal_mask = F.one_hot(focal_arm, self.num_arms).bool()  # [B, num_arms]
        focal_mask = focal_mask.unsqueeze(-1)  # [B, num_arms, 1]
        
        # Apply noise only to non-focal arms
        noisy_state_per_arm = state_per_arm + torch.where(focal_mask, torch.zeros_like(noise), noise)
        
        # Reshape back
        return noisy_state_per_arm.view(B, D)

    def create_per_arm_noisy_actions(self, actions, noise, time_per_arm):
        """
        Create noisy actions with different noise levels per arm.
        
        Args:
            actions: [B, H, D] where D = num_arms * arm_action_dim
            noise: [B, H, D]
            time_per_arm: [B, num_arms]
            
        Returns:
            x_t: [B, H, D] noisy actions
        """
        B, H, D = actions.shape
        
        # Reshape to separate arms: [B, H, num_arms, arm_dim]
        actions_per_arm = actions.view(B, H, self.num_arms, self.arm_action_dim)
        noise_per_arm = noise.view(B, H, self.num_arms, self.arm_action_dim)
        
        # Expand time: [B, num_arms] -> [B, 1, num_arms, 1]
        t = time_per_arm[:, None, :, None]
        
        # x_t = t * noise + (1-t) * actions, per arm
        x_t_per_arm = t * noise_per_arm + (1 - t) * actions_per_arm
        
        # Reshape back: [B, H, D]
        x_t = x_t_per_arm.view(B, H, D)
        
        return x_t

    # ========== EMBED METHODS (SAME AS ORIGINAL) ==========

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """
        Embed suffix (same as original PI0).
        
        Note: timestep here is a scalar time (we use mean of per-arm times).
        The per-arm information is already encoded in noisy_actions.
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, 
            min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.to(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # ========== FORWARD (MODIFIED FOR DIFFUSION FORCING) ==========

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """
        Training forward with Diffusion Forcing.
        
        The KEY difference from original PI0:
        - With probability `per_arm_noise_prob`, we use per-arm noise levels
        - Each arm gets a different amount of noise
        - Model learns to not over-rely on any single arm
        
        Mathematical insight:
        - x_t[arm_i] = t_i * noise[arm_i] + (1-t_i) * actions[arm_i]
        - u_t = noise - actions (target, SAME for all, independent of t!)
        - Model sees mixed-noise input, predicts same target
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=True
        )
        bsize = actions.shape[0]
        device = actions.device

        if noise is None:
            noise = self.sample_noise(actions.shape, device)

        # Decide whether to use per-arm noise (Diffusion Forcing) or uniform noise
        use_per_arm = torch.rand(1).item() < self.per_arm_noise_prob

        if use_per_arm and time is None:
            # === DIFFUSION FORCING: per-arm noise levels ===
            time_per_arm, focal_arm = self.sample_time_per_arm(bsize, device)
            x_t = self.create_per_arm_noisy_actions(actions, noise, time_per_arm)
            
            # Add noise to non-focal arms' state observations
            noisy_state = self.add_noise_to_state(state, focal_arm)
            
            # Use mean time for the model's time conditioning
            # (the per-arm info is already in x_t)
            time = time_per_arm.mean(dim=1)  # [B]
        else:
            # === ORIGINAL: uniform noise level ===
            if time is None:
                time = self.sample_time(bsize, device)
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            noisy_state = state  # No state noise in uniform mode

        # Target velocity: u_t = noise - actions (same for all cases!)
        u_t = noise - actions

        # Forward through model (SAME as original PI0)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            noisy_state, x_t, time  # Use noisy_state instead of state
        )

        if (self.paligemma_with_expert.paligemma.language_model.layers[0]
                .self_attn.q_proj.weight.dtype == torch.bfloat16):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
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

        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    # ========== INFERENCE WITH PER-ARM SUPPORT ==========

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, focal_arm=None, use_ensemble=False) -> Tensor:
        """
        Inference with optional per-arm noise matching training distribution.
        
        Args:
            device: torch device
            observation: input observation
            noise: optional initial noise
            num_steps: number of denoising steps (recommend 20-25 for stability)
            focal_arm: Optional[int] - which arm to prioritize (lower noise).
                      If None, uses uniform noise (original behavior).
                      If specified, matches training distribution with focal arm.
            use_ensemble: If True, run inference with each arm as focal and average results.
                         Most stable but slower (num_arms times inference).
        
        Returns:
            Denoised actions tensor
        """
        # If ensemble mode, delegate to ensemble method
        if use_ensemble:
            return self._sample_actions_ensemble_internal(device, observation, noise, num_steps)
        
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

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
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        base_time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Setup per-arm time offsets if focal_arm is specified
        if focal_arm is not None:
            if isinstance(focal_arm, int):
                focal_arm_tensor = torch.full((bsize,), focal_arm, dtype=torch.long, device=device)
            else:
                focal_arm_tensor = focal_arm.to(device)
            
            # Create time offset: non-focal arms start with extra noise (matching training)
            # Use half of training offset for inference stability
            inference_offset = self.noise_offset * 0.5
            time_offset = torch.full((bsize, self.num_arms), inference_offset, device=device)
            focal_mask = F.one_hot(focal_arm_tensor, self.num_arms).bool()
            time_offset = torch.where(focal_mask, torch.zeros_like(time_offset), time_offset)
        else:
            time_offset = None

        while base_time >= -dt_tensor / 2:
            if time_offset is not None:
                # Per-arm time: focal arm uses base_time, others use base_time + offset
                time_per_arm = (base_time + time_offset).clamp(0.001, 1.0)
                effective_time = time_per_arm.mean(dim=1)  # [B]
            else:
                effective_time = base_time.expand(bsize)
            
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, effective_time)
            x_t = x_t + dt_tensor * v_t
            base_time = base_time + dt_tensor

        return x_t
    
    @torch.no_grad()
    def _sample_actions_ensemble_internal(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """
        Internal ensemble inference: each arm takes turn as focal, average results.
        
        More stable but slower (num_arms times inference).
        Use this if single focal_arm inference still shows jitter.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        all_actions = []
        for focal_arm_idx in range(self.num_arms):
            # Call with use_ensemble=False to avoid infinite recursion
            actions = self.sample_actions(
                device, observation, 
                noise=noise.clone(),
                num_steps=num_steps,
                focal_arm=focal_arm_idx,
                use_ensemble=False,
            )
            all_actions.append(actions)
        
        # Average across all focal arm choices
        return torch.stack(all_actions, dim=0).mean(dim=0)
    
    # Alias for backward compatibility
    sample_actions_ensemble = _sample_actions_ensemble_internal

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Single denoising step (same as original PI0)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

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
        suffix_out = suffix_out[:, -self.config.action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

