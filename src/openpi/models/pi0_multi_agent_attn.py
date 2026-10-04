"""Pi0 Multi-Agent Attention model with per-agent selective attention.

Each agent's action tokens only attend to:
- Global view image tokens
- Their own wrist view image tokens  
- All language tokens (shared across agents)

They do NOT attend to other agents' wrist view tokens.
"""

import dataclasses
import logging
from typing import Sequence

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class TokenBoundaries:
    """Tracks token positions for selective attention.
    
    This dataclass tracks where different types of tokens are located
    in the concatenated sequence, enabling per-agent attention masking.
    
    Prefix structure (with per-agent prompts and LCT):
        [global_image] [wrist0_image] [wrist1_image] [agent0_prompt] [agent1_prompt] [LCT_tokens]
        Each agent's prompt is independently padded to max_token_len_per_agent.
    
    Suffix structure (for 2 agents, horizon=50):
        [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
        Total: num_agents + num_agents * action_horizon tokens
    """
    # Image name -> (start_idx, end_idx) in the prefix sequence
    image_boundaries: dict[str, tuple[int, int]]
    # Global image token indices (start, end)
    global_image_range: tuple[int, int]
    # Total prefix length (images + language + LCT)
    prefix_length: int
    # Language tokens range (overall range covering all agent prompts)
    language_range: tuple[int, int]
    # Latent Collaboration Token range (start, end) in prefix
    latent_collab_range: tuple[int, int] = (0, 0)
    # Agent ID -> state token index in suffix (relative to suffix start)
    # e.g., {0: 0, 1: 1} means agent 0's state is at suffix[0], agent 1's at suffix[1]
    state_token_indices: dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of state tokens (one per agent)
    num_state_tokens: int = 1
    # Agent ID -> (start_idx, end_idx) for that agent's action tokens in suffix
    # e.g., {0: (2, 52), 1: (52, 102)} means agent 0's actions at suffix[2:52], agent 1's at suffix[52:102]
    action_token_boundaries: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)
    # Action horizon per agent
    action_horizon_per_agent: int = 50
    # Per-agent language token ranges (agent_id -> (start, end) in prefix)
    # Each agent's prompt is independently tokenized and padded
    # e.g., {0: (256, 280), 1: (280, 304)} means agent 0's prompt at [256:280], agent 1's at [280:304]
    per_agent_language_ranges: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)


def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def make_attn_mask(input_mask, mask_ar):
    """Standard attention mask from big_vision.
    
    Creates attention mask based on cumulative ar_mask values.
    Token i can attend to token j if cumsum(ar_mask)[i] >= cumsum(ar_mask)[j].
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_per_agent_prefix_attn_mask(
    prefix_mask: jnp.ndarray,      # [B, prefix_len]
    ar_mask: jnp.ndarray,          # [prefix_len]
    boundaries: TokenBoundaries,
    num_agents: int,
    agent_to_wrist_image: dict[int, str],
    apply_prefix_mask: bool = True,  # Whether to apply arm-wise masking to prefix tokens
) -> jnp.ndarray:
    """Create per-agent attention mask for prefix tokens.
    
    Prefix structure (with per-agent prompts):
        [global_image] [wrist0_image] [wrist1_image] [agent0_prompt] [agent1_prompt] [LCT_tokens]
    
    Attention rules (when apply_prefix_mask=True):
    - Global image tokens: attend to ALL prefix tokens (information hub)
    - Wrist image tokens: attend to global + own wrist + OWN prompt (arm-wise, NOT LCT)
    - Language tokens: each agent's prompt attends to global + own wrist + own prompt
    - LCT tokens: attend to ALL prefix tokens (including all wrist images) - this is the key for collaboration
    
    When apply_prefix_mask=False: all prefix tokens can attend to all prefix tokens (full attention)
    
    Args:
        prefix_mask: Padding mask [B, prefix_len]
        ar_mask: Auto-regressive mask [prefix_len]
        boundaries: Token position boundaries (with per_agent_language_ranges)
        num_agents: Number of agents
        agent_to_wrist_image: Mapping from agent_id to wrist image key
        apply_prefix_mask: Whether to apply arm-wise masking (True) or full attention (False)
    
    Returns:
        Attention mask [B, prefix_len, prefix_len]
    """
    batch_size = prefix_mask.shape[0]
    prefix_len = prefix_mask.shape[1]
    
    # Start with standard attention mask
    base_mask = make_attn_mask(prefix_mask, ar_mask)  # [B, prefix_len, prefix_len]
    
    # If not applying prefix mask, return the base mask (full attention for prefix)
    if not apply_prefix_mask:
        return base_mask
    
    global_start, global_end = boundaries.global_image_range
    lct_start, lct_end = boundaries.latent_collab_range
    
    # Helper function to create allowed key mask for an agent
    # Each agent can see: global image + own wrist image + OWN prompt tokens
    # NOTE: wrist images do NOT see LCT (LCT info flows through action tokens)
    def create_agent_allowed_mask(agent_id: int) -> jnp.ndarray:
        allowed_key_mask = jnp.zeros(prefix_len, dtype=jnp.bool_)
        
        # 1. Allow global image tokens
        allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
        
        # 2. Allow own wrist image tokens
        wrist_image_name = agent_to_wrist_image.get(agent_id)
        if wrist_image_name and wrist_image_name in boundaries.image_boundaries:
            wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
            allowed_key_mask = allowed_key_mask.at[wrist_start:wrist_end].set(True)
        
        # 3. Allow OWN prompt tokens only (per-agent prompt separation)
        if agent_id in boundaries.per_agent_language_ranges:
            agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
            allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
        
        # NOTE: wrist images do NOT see LCT tokens - LCT info flows through suffix tokens
        
        return allowed_key_mask
    
    # Process each wrist image: restrict to global + own wrist + own prompt
    for agent_id in range(num_agents):
        wrist_image_name = agent_to_wrist_image.get(agent_id)
        if wrist_image_name is None or wrist_image_name not in boundaries.image_boundaries:
            continue
        
        wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
        
        # Create allowed key mask for this agent's wrist image tokens
        allowed_key_mask = create_agent_allowed_mask(agent_id)
        
        # Respect padding mask - apply per-batch (each sample may have different prompt length)
        allowed_key_mask_batched = jnp.logical_and(allowed_key_mask[None, :], prefix_mask)  # [B, prefix_len]
        
        # Apply mask to all wrist image query positions
        for query_idx in range(wrist_start, wrist_end):
            if query_idx >= prefix_len:
                break
            base_mask = base_mask.at[:, query_idx, :].set(allowed_key_mask_batched)
    
    # Process each agent's prompt tokens: restrict to global + own wrist + own prompt
    for agent_id in range(num_agents):
        if agent_id not in boundaries.per_agent_language_ranges:
            continue
        
        agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
        
        # Create allowed key mask for this agent's prompt tokens
        allowed_key_mask = create_agent_allowed_mask(agent_id)
        
        # Respect padding mask
        allowed_key_mask_batched = jnp.logical_and(allowed_key_mask[None, :], prefix_mask)
        
        # Apply mask to all prompt query positions for this agent
        for query_idx in range(agent_lang_start, agent_lang_end):
            if query_idx >= prefix_len:
                break
            base_mask = base_mask.at[:, query_idx, :].set(allowed_key_mask_batched)
    
    # LCT tokens can attend to ALL prefix tokens (including all wrist images and all prompts)
    # This is the key mechanism for cross-arm information aggregation
    if lct_start < lct_end:
        full_prefix_mask = jnp.ones(prefix_len, dtype=jnp.bool_)
        for query_idx in range(lct_start, lct_end):
            if query_idx >= prefix_len:
                break
            # LCT can see everything in prefix
            full_prefix_mask_batched = jnp.logical_and(full_prefix_mask[None, :], prefix_mask)
            base_mask = base_mask.at[:, query_idx, :].set(full_prefix_mask_batched)
    
    # Global image tokens can attend to ALL prefix tokens (already in base_mask, no change needed)
    
    return base_mask


def make_per_agent_attn_mask(
    input_mask: jnp.ndarray,  # [B, total_seq_len]
    ar_mask: jnp.ndarray,     # [total_seq_len]
    boundaries: TokenBoundaries,
    num_agents: int,
    action_horizon: int,
    agent_to_wrist_image: dict[int, str],
    actions_per_agent: int | None = None,
    apply_prefix_mask: bool = True,  # whether to apply arm-wise prefix masking
    apply_suffix_mask: bool = True,  # whether to apply arm-wise suffix masking
) -> jnp.ndarray:
    """Create per-agent attention mask for both prefix and suffix tokens.
    
    Prefix structure (with per-agent prompts):
        [global_image] [wrist0_image] [wrist1_image] [agent0_prompt] [agent1_prompt] [LCT_tokens]
    
    Suffix structure (for 2 agents, horizon=50):
        [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
    
    Prefix attention rules (when apply_prefix_mask=True):
    - Global image tokens: attend to ALL prefix tokens (information hub)
    - Wrist image tokens: attend to global + own wrist + OWN prompt (NOT LCT)
    - Prompt tokens: attend to global + own wrist + OWN prompt (per agent)
    - LCT tokens: attend to ALL prefix tokens (key for cross-arm collaboration)
    
    Suffix attention rules:
    - State tokens: attend to global + own wrist + OWN prompt + LCT + own state
    - Action tokens: attend to global + own wrist + OWN prompt + LCT + own state + own actions (causally)
    
    Args:
        input_mask: Padding mask [B, total_seq_len]
        ar_mask: Auto-regressive mask [total_seq_len]
        boundaries: Token position boundaries (with action_token_boundaries and per_agent_language_ranges)
        num_agents: Number of robot agents
        action_horizon: Total action tokens (num_agents * horizon_per_agent)
        agent_to_wrist_image: Mapping from agent_id to wrist image key name
        actions_per_agent: Actions per agent (horizon per agent)
        apply_prefix_mask: Whether to apply arm-wise masking to prefix tokens
        apply_suffix_mask: Whether to apply arm-wise masking to suffix (state/action expert) tokens
    
    Returns:
        Attention mask [B, total_seq_len, total_seq_len]
    """
    batch_size = input_mask.shape[0]
    total_len = input_mask.shape[1]
    
    # Start with standard attention mask
    base_mask = make_attn_mask(input_mask, ar_mask)  # [B, total_len, total_len]
    
    prefix_len = boundaries.prefix_length
    global_start, global_end = boundaries.global_image_range
    lct_start, lct_end = boundaries.latent_collab_range
    
    # Helper function to create allowed key mask for an agent (for prefix keys)
    # Each agent can see: global image + own wrist image + OWN prompt tokens + LCT tokens
    def create_agent_prefix_allowed_mask(agent_id: int) -> jnp.ndarray:
        allowed_key_mask = jnp.zeros(total_len, dtype=jnp.bool_)
        
        # 1. Allow global image tokens
        allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
        
        # 2. Allow own wrist image tokens
        wrist_image_name = agent_to_wrist_image.get(agent_id)
        if wrist_image_name and wrist_image_name in boundaries.image_boundaries:
            wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
            allowed_key_mask = allowed_key_mask.at[wrist_start:wrist_end].set(True)
        
        # 3. Allow OWN prompt tokens only (per-agent prompt separation)
        if agent_id in boundaries.per_agent_language_ranges:
            agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
            allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
        
        # 4. Allow LCT tokens (cross-arm collaboration bridge)
        if lct_start < lct_end:
            allowed_key_mask = allowed_key_mask.at[lct_start:lct_end].set(True)
        
        return allowed_key_mask
    
    # ========== STEP 1: Apply arm-wise masking to prefix tokens ==========
    if apply_prefix_mask:
        # Process each wrist image: restrict to global + own wrist + own prompt (NOT LCT)
        for agent_id in range(num_agents):
            wrist_image_name = agent_to_wrist_image.get(agent_id)
            if wrist_image_name is None or wrist_image_name not in boundaries.image_boundaries:
                continue
            
            wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
            
            # Create allowed key mask for wrist image tokens (without LCT)
            allowed_key_mask = jnp.zeros(total_len, dtype=jnp.bool_)
            allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
            if wrist_image_name in boundaries.image_boundaries:
                ws, we = boundaries.image_boundaries[wrist_image_name]
                allowed_key_mask = allowed_key_mask.at[ws:we].set(True)
            # Only allow OWN prompt tokens (per-agent prompt separation)
            if agent_id in boundaries.per_agent_language_ranges:
                agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
                allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
            # NOTE: wrist images do NOT see LCT - LCT info flows through suffix tokens
            
            # Respect padding mask - apply per-batch (each sample may have different prompt length)
            allowed_key_mask_batched = jnp.logical_and(allowed_key_mask[None, :], input_mask)  # [B, total_len]
            
            # Apply mask to all wrist image query positions
            for query_idx in range(wrist_start, wrist_end):
                if query_idx >= prefix_len:
                    break
                base_mask = base_mask.at[:, query_idx, :].set(allowed_key_mask_batched)
        
        # Process each agent's prompt tokens: restrict to global + own wrist + own prompt
        for agent_id in range(num_agents):
            if agent_id not in boundaries.per_agent_language_ranges:
                continue
            
            agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
            
            # Create allowed key mask for this agent's prompt tokens (without LCT)
            allowed_key_mask = jnp.zeros(total_len, dtype=jnp.bool_)
            allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
            
            wrist_image_name = agent_to_wrist_image.get(agent_id)
            if wrist_image_name and wrist_image_name in boundaries.image_boundaries:
                ws, we = boundaries.image_boundaries[wrist_image_name]
                allowed_key_mask = allowed_key_mask.at[ws:we].set(True)
            
            # Only allow OWN prompt tokens
            allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
            
            # Respect padding mask
            allowed_key_mask_batched = jnp.logical_and(allowed_key_mask[None, :], input_mask)
            
            # Apply mask to all prompt query positions for this agent
            for query_idx in range(agent_lang_start, agent_lang_end):
                if query_idx >= prefix_len:
                    break
                base_mask = base_mask.at[:, query_idx, :].set(allowed_key_mask_batched)
        
        # LCT tokens can attend to ALL prefix tokens (including all wrist images and all prompts)
        if lct_start < lct_end:
            full_prefix_mask = jnp.zeros(total_len, dtype=jnp.bool_)
            full_prefix_mask = full_prefix_mask.at[:prefix_len].set(True)
            for query_idx in range(lct_start, lct_end):
                if query_idx >= prefix_len:
                    break
                full_prefix_mask_batched = jnp.logical_and(full_prefix_mask[None, :], input_mask)
                base_mask = base_mask.at[:, query_idx, :].set(full_prefix_mask_batched)
    
    # ========== STEP 2: Apply arm-wise masking to suffix tokens ==========
    
    if apply_suffix_mask:
        # Calculate token positions
        prefix_len = boundaries.prefix_length
        num_state_tokens = boundaries.num_state_tokens
        
        if actions_per_agent is None:
            actions_per_agent = action_horizon // num_agents
        
        # For each agent, determine which key tokens they can attend to
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
            
            # Determine allowed key token ranges for this agent
            allowed_key_mask = jnp.zeros(total_len, dtype=jnp.bool_)
            
            # 1. Always allow global image tokens
            global_start, global_end = boundaries.global_image_range
            allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
            
            # 2. Allow this agent's own wrist image tokens
            if agent_id in agent_to_wrist_image:
                wrist_image_name = agent_to_wrist_image[agent_id]
                if wrist_image_name in boundaries.image_boundaries:
                    wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
                    allowed_key_mask = allowed_key_mask.at[wrist_start:wrist_end].set(True)
            
            # 3. Allow OWN prompt tokens only (per-agent prompt separation)
            if agent_id in boundaries.per_agent_language_ranges:
                agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
                allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
            
            # 4. Allow LCT tokens (cross-arm collaboration bridge)
            if lct_start < lct_end:
                allowed_key_mask = allowed_key_mask.at[lct_start:lct_end].set(True)
            
            # 5. Allow ONLY this agent's own state token (NOT other agents' state tokens)
            if agent_id in boundaries.state_token_indices:
                agent_state_idx = prefix_len + boundaries.state_token_indices[agent_id]
                allowed_key_mask = allowed_key_mask.at[agent_state_idx].set(True)
            else:
                # Fallback: if no per-agent state tracking, allow all state tokens
                for state_idx in range(num_state_tokens):
                    allowed_key_mask = allowed_key_mask.at[prefix_len + state_idx].set(True)
            
            # 6. Allow this agent's own action tokens (will apply causality below)
            allowed_key_mask = allowed_key_mask.at[agent_action_start:agent_action_end].set(True)
            
            # Apply the mask to each action token in this agent's range
            for query_idx in range(agent_action_start, agent_action_end):
                if query_idx >= total_len:
                    break
                    
                # Create the attention mask for this query position
                query_mask = allowed_key_mask.copy()
                
                # Apply causality for action tokens: can only attend to previous actions
                for key_idx in range(agent_action_start, agent_action_end):
                    if key_idx > query_idx:
                        query_mask = query_mask.at[key_idx].set(False)
                
                # Also respect padding mask - apply per-batch (each sample may have different prompt length)
                query_mask_batched = jnp.logical_and(query_mask[None, :], input_mask)  # [B, total_len]
                
                # Update the base mask for this query position
                base_mask = base_mask.at[:, query_idx, :].set(query_mask_batched)
        
        # Also update attention for state tokens: each state token should only see its agent's stuff + LCT
        for agent_id in range(num_agents):
            if agent_id not in boundaries.state_token_indices:
                continue
                
            state_query_idx = prefix_len + boundaries.state_token_indices[agent_id]
            if state_query_idx >= total_len:
                continue
            
            # State token can attend to: global image + own wrist + OWN prompt + LCT
            allowed_key_mask = jnp.zeros(total_len, dtype=jnp.bool_)
            
            # Global image
            global_start, global_end = boundaries.global_image_range
            allowed_key_mask = allowed_key_mask.at[global_start:global_end].set(True)
            
            # Own wrist image
            if agent_id in agent_to_wrist_image:
                wrist_image_name = agent_to_wrist_image[agent_id]
                if wrist_image_name in boundaries.image_boundaries:
                    wrist_start, wrist_end = boundaries.image_boundaries[wrist_image_name]
                    allowed_key_mask = allowed_key_mask.at[wrist_start:wrist_end].set(True)
            
            # OWN prompt tokens only (per-agent prompt separation)
            if agent_id in boundaries.per_agent_language_ranges:
                agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
                allowed_key_mask = allowed_key_mask.at[agent_lang_start:agent_lang_end].set(True)
            
            # LCT tokens (cross-arm collaboration bridge)
            lct_start, lct_end = boundaries.latent_collab_range
            if lct_start < lct_end:
                allowed_key_mask = allowed_key_mask.at[lct_start:lct_end].set(True)
            
            # Own state token (self-attention)
            allowed_key_mask = allowed_key_mask.at[state_query_idx].set(True)
            
            # Respect padding - apply per-batch (each sample may have different prompt length)
            allowed_key_mask_batched = jnp.logical_and(allowed_key_mask[None, :], input_mask)  # [B, total_len]
            
            base_mask = base_mask.at[:, state_query_idx, :].set(allowed_key_mask_batched)
    
    return base_mask


@dataclasses.dataclass(frozen=True)
class Pi0MultiAgentAttnConfig(_model.BaseModelConfig):
    """Configuration for Pi0 with per-agent attention masking."""
    
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    
    # Standard Pi0 parameters
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    pi05: bool = False
    discrete_state_input: bool = None  # type: ignore
    
    # Multi-agent attention parameters
    num_agents: int = 2
    action_dim_per_agent: int = 8
    
    # Mapping from agent ID to wrist image key
    # Default for 2-arm setup
    agent_to_wrist_image: tuple[tuple[int, str], ...] = (
        (0, "left_wrist_0_rgb"),
        (1, "right_wrist_0_rgb"),
    )
    
    # Global image key (will be accessible to all agents)
    global_image_key: str = "base_0_rgb"
    
    # Whether to use arm-wise attention for prefix tokens (images + language)
    # If True: each wrist image only sees global + own wrist + all language (arm-wise)
    # If False: all prefix tokens can attend to all prefix tokens (full attention)
    use_arm_wise_prefix_attn: bool = True

    # Whether to use arm-wise attention for suffix tokens (state + action expert side)
    # If True: each agent's state/action tokens only see their own prefix slice + own tokens
    # If False: suffix tokens fall back to the standard ar_mask-based attention (full attention)
    use_arm_wise_suffix_attn: bool = True
    
    # Latent Collaboration Token (LCT) parameters
    # LCT tokens act as a bridge for cross-arm information exchange
    # They can see all images (global + all wrists) and language, and all agents can attend to them
    use_latent_collab_tokens: bool = False  # Whether to use LCT
    num_latent_collab_tokens: int = 4       # Number of LCT tokens (recommended: 4-8)
    
    # Per-agent prompt parameters
    # If True: prompt is split by "|" separator, each agent gets its own prompt portion
    # Each portion is independently tokenized and padded to max_token_len_per_agent
    # Format: "agent0_instruction | agent1_instruction | ..."
    use_per_agent_prompt: bool = False
    # Max token length per agent (used when use_per_agent_prompt=True)
    # If None, defaults to max_token_len // num_agents
    max_token_len_per_agent: int | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
    
    def get_max_token_len_per_agent(self) -> int:
        """Get max token length per agent, with fallback to max_token_len // num_agents."""
        if self.max_token_len_per_agent is not None:
            return self.max_token_len_per_agent
        return self.max_token_len // self.num_agents

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MultiAgentAttn":
        return Pi0MultiAgentAttn(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # Build image specs based on configuration
        images = {self.global_image_key: image_spec}
        image_masks = {self.global_image_key: image_mask_spec}
        
        for agent_id, wrist_key in self.agent_to_wrist_image:
            images[wrist_key] = image_spec
            image_masks[wrist_key] = image_mask_spec

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=images,
                image_masks=image_masks,
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_agent_to_wrist_dict(self) -> dict[int, str]:
        """Convert tuple config to dict for easier access."""
        return dict(self.agent_to_wrist_image)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        return nnx.Nothing

    def load_pytorch(self, train_config, weight_path: str):
        """Load PI0MultiAgentAttnPytorch model."""
        import logging
        import safetensors.torch
        from openpi.models_pytorch.pi0_multi_agent_attn_pytorch import PI0MultiAgentAttnPytorch
        
        logger = logging.getLogger(__name__)
        logger.info(f"train_config: {train_config}")
        print("  [load_pytorch] Creating PI0MultiAgentAttnPytorch model...", flush=True)
        
        model = PI0MultiAgentAttnPytorch(config=train_config.model)
        print(f"  [load_pytorch] Model created, loading weights from {weight_path}...", flush=True)
        print("  [load_pytorch] This may take several minutes for large models...", flush=True)
        
        safetensors.torch.load_model(model, weight_path)
        print("  [load_pytorch] Weights loaded successfully!", flush=True)
        return model


class Pi0MultiAgentAttn(_model.BaseModel):
    """Pi0 model with per-agent selective attention.
    
    Each agent's action tokens only attend to:
    - Global view image tokens
    - Their own wrist view image tokens
    - Their own prompt portion
    - State token
    - Their own previous action tokens (causally)
    """
    
    def __init__(self, config: Pi0MultiAgentAttnConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Shared vision-language model
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # Action processing layers - SHARED across all agents (same as original Pi0)
        # This allows loading pretrained Pi0 weights directly
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # Per-agent state projections (these are new, randomly initialized)
            self.state_projs = {
                f"agent{i}": nnx.Linear(config.action_dim_per_agent, action_expert_config.width, rngs=rngs)
                for i in range(config.num_agents)
            }
            # Shared action time MLP (same as original Pi0, can load pretrained weights)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        
        # Store agent-to-image mapping
        self._agent_to_wrist_image = config.get_agent_to_wrist_dict()
        
        # Latent Collaboration Tokens (LCT) - learnable tokens for cross-arm information exchange
        if config.use_latent_collab_tokens and config.num_latent_collab_tokens > 0:
            # Small random initialization (similar to learnable positional embeddings)
            # Use paligemma_config.width (2048) to match prefix token dimensions
            self.latent_collab_tokens = nnx.Param(
                jax.random.normal(rngs(), (config.num_latent_collab_tokens, paligemma_config.width)) * 0.02
            )
        else:
            self.latent_collab_tokens = None
        
        self.deterministic = True

    @at.typecheck
    def embed_prefix_with_boundaries(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        TokenBoundaries,
    ]:
        """Embed prefix tokens and track token boundaries.
        
        Returns tokens, masks, and boundary information for per-agent attention.
        
        When use_per_agent_prompt=True:
            - tokenized_prompt is expected to be [B, num_agents * max_token_len_per_agent]
            - Each agent's prompt is stored consecutively: [agent0_tokens, agent1_tokens, ...]
            - Each agent's prompt portion is independently padded to max_token_len_per_agent
            - per_agent_language_ranges tracks each agent's prompt position in the prefix
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Initialize boundaries tracking
        image_boundaries = {}
        global_image_range = (0, 0)
        per_agent_language_ranges = {}
        current_pos = 0
        
        # Embed images and track their positions
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            num_tokens = image_tokens.shape[1]
            
            start_pos = current_pos
            end_pos = current_pos + num_tokens
            
            # Track which image this is
            if name == self.config.global_image_key or "base" in name or "global" in name:
                global_image_range = (start_pos, end_pos)
            else:
                image_boundaries[name] = (start_pos, end_pos)
            
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=num_tokens,
                )
            )
            ar_mask += [False] * num_tokens
            current_pos = end_pos

        # Add language tokens and track position
        language_start = current_pos
        
        if obs.tokenized_prompt is not None:
            if self.config.use_per_agent_prompt:
                # Per-agent prompt mode: split tokenized_prompt into per-agent portions
                # tokenized_prompt: [B, num_agents * max_token_len_per_agent]
                # tokenized_prompt_mask: [B, num_agents * max_token_len_per_agent]
                max_len_per_agent = self.config.get_max_token_len_per_agent()
                
                for agent_id in range(self.config.num_agents):
                    # Extract this agent's prompt portion
                    start_in_prompt = agent_id * max_len_per_agent
                    end_in_prompt = start_in_prompt + max_len_per_agent
                    
                    agent_prompt = obs.tokenized_prompt[:, start_in_prompt:end_in_prompt]
                    agent_prompt_mask = obs.tokenized_prompt_mask[:, start_in_prompt:end_in_prompt]
                    
                    # Embed this agent's prompt
                    agent_tokens = self.PaliGemma.llm(agent_prompt, method="embed")
                    
                    tokens.append(agent_tokens)
                    input_mask.append(agent_prompt_mask)
                    ar_mask += [False] * max_len_per_agent
                    
                    # Track this agent's language token range in prefix
                    per_agent_language_ranges[agent_id] = (current_pos, current_pos + max_len_per_agent)
                    current_pos += max_len_per_agent
            else:
                # Original mode: shared prompt for all agents
                tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                tokens.append(tokenized_inputs)
                input_mask.append(obs.tokenized_prompt_mask)
                ar_mask += [False] * tokenized_inputs.shape[1]
                current_pos += tokenized_inputs.shape[1]
                
                # In shared mode, all agents see the same language range
                for agent_id in range(self.config.num_agents):
                    per_agent_language_ranges[agent_id] = (language_start, current_pos)
        
        language_end = current_pos
        
        # Add Latent Collaboration Tokens (LCT) if enabled
        latent_collab_start = current_pos
        if self.config.use_latent_collab_tokens and self.latent_collab_tokens is not None:
            batch_size = list(obs.images.values())[0].shape[0]
            # Get LCT embeddings and broadcast to batch
            lct_tokens = self.latent_collab_tokens.value  # [num_lct, emb]
            lct_tokens = einops.repeat(lct_tokens, "n e -> b n e", b=batch_size)
            
            num_lct = self.config.num_latent_collab_tokens
            tokens.append(lct_tokens)
            input_mask.append(jnp.ones((batch_size, num_lct), dtype=jnp.bool_))
            ar_mask += [False] * num_lct  # Non-autoregressive
            current_pos += num_lct
        latent_collab_end = current_pos
        
        # Create boundaries object
        boundaries = TokenBoundaries(
            image_boundaries=image_boundaries,
            global_image_range=global_image_range,
            prefix_length=current_pos,
            language_range=(language_start, language_end),
            latent_collab_range=(latent_collab_start, latent_collab_end),
            per_agent_language_ranges=per_agent_language_ranges,
        )
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, boundaries

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
        dict[int, int],  # state_token_indices: agent_id -> index in suffix
        dict[int, tuple[int, int]],  # action_token_boundaries: agent_id -> (start, end) in suffix
    ]:
        """Embed suffix tokens with per-agent decoupled action tokens.
        
        Pi0 suffix (for 2 agents, horizon=50):
            [state_0, state_1, action_0_t0...action_0_t49, action_1_t0...action_1_t49]
            Total: num_agents + num_agents * action_horizon tokens
        
        Pi0.5 suffix (for 2 agents, horizon=50):
            [action_0_t0...action_0_t49, action_1_t0...action_1_t49]
            Total: num_agents * action_horizon tokens (no state tokens, state is discrete in prefix)
        
        Each agent has action_horizon action tokens using the SHARED Pi0 projection.
        
        Args:
            obs: Observation containing state
            noisy_actions: Noisy action sequence [B, action_horizon, action_dim]
            timestep: Diffusion timestep [B]
        
        Returns:
            tokens: Embedded suffix tokens
            input_mask: Padding mask
            ar_mask: Auto-regressive mask
            adarms_cond: AdaRMS conditioning (for pi05, None for pi0)
            state_token_indices: Mapping from agent_id to state token index in suffix
            action_token_boundaries: Mapping from agent_id to (start, end) in suffix
        """
        input_mask = []
        ar_mask = []
        tokens = []
        state_token_indices = {}
        action_token_boundaries = {}
        
        batch_size = obs.state.shape[0]
        current_suffix_idx = 0
        
        if self.pi05:
            # Pi0.5 mode: no state tokens (state is discrete in prefix), uses AdaRMS
            # Still create per-agent action token blocks for multi-agent attention masking
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
            
            for agent_id in range(self.config.num_agents):
                agent_action_tokens = self.action_in_proj(noisy_actions)  # [B, horizon, width]
                
                tokens.append(agent_action_tokens)
                input_mask.append(jnp.ones((batch_size, self.action_horizon), dtype=jnp.bool_))
                ar_mask += [True] + ([False] * (self.action_horizon - 1))
                
                action_start = current_suffix_idx
                action_end = current_suffix_idx + self.action_horizon
                action_token_boundaries[agent_id] = (action_start, action_end)
                current_suffix_idx = action_end
        else:
            # Pi0 mode: per-agent state tokens + per-agent action tokens
            
            # 1. Create per-agent state tokens
            for agent_id in range(self.config.num_agents):
                start_dim = agent_id * self.config.action_dim_per_agent
                end_dim = start_dim + self.config.action_dim_per_agent
                agent_state = obs.state[:, start_dim:end_dim]  # [B, action_dim_per_agent]
                
                state_proj = self.state_projs[f"agent{agent_id}"]
                agent_state_token = state_proj(agent_state)[:, None, :]  # [B, 1, emb]
                
                tokens.append(agent_state_token)
                input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
                ar_mask += [True]
                
                state_token_indices[agent_id] = current_suffix_idx
                current_suffix_idx += 1
            
            # 2. Create per-agent action tokens using SHARED projection
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            
            for agent_id in range(self.config.num_agents):
                agent_action_tokens = self.action_in_proj(noisy_actions)  # [B, horizon, width]
                
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
                action_time_tokens = jnp.concatenate([agent_action_tokens, time_tokens], axis=-1)
                
                action_time_tokens = self.action_time_mlp_in(action_time_tokens)
                action_time_tokens = nnx.swish(action_time_tokens)
                action_time_tokens = self.action_time_mlp_out(action_time_tokens)
                
                tokens.append(action_time_tokens)
                input_mask.append(jnp.ones((batch_size, self.action_horizon), dtype=jnp.bool_))
                ar_mask += [True] + ([False] * (self.action_horizon - 1))
                
                action_start = current_suffix_idx
                action_end = current_suffix_idx + self.action_horizon
                action_token_boundaries[agent_id] = (action_start, action_end)
                current_suffix_idx = action_end
            
            adarms_cond = None
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, adarms_cond, state_token_indices, action_token_boundaries

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=image_keys)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix with boundary tracking
        prefix_tokens, prefix_mask, prefix_ar_mask, boundaries = self.embed_prefix_with_boundaries(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, state_token_indices, action_token_boundaries = self.embed_suffix(observation, x_t, time)
        
        # Update boundaries with state token and action token information
        boundaries.state_token_indices = state_token_indices
        boundaries.num_state_tokens = self.config.num_agents if not self.pi05 else 0
        boundaries.action_token_boundaries = action_token_boundaries
        boundaries.action_horizon_per_agent = self.action_horizon
        
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        
        # Create per-agent attention mask
        attn_mask = make_per_agent_attn_mask(
            input_mask,
            ar_mask,
            boundaries,
            num_agents=self.config.num_agents,
            action_horizon=self.action_horizon * self.config.num_agents,  # Total action tokens
            agent_to_wrist_image=self._agent_to_wrist_image,
            actions_per_agent=self.action_horizon,  # Each agent has action_horizon tokens
            apply_prefix_mask=self.config.use_arm_wise_prefix_attn,
            apply_suffix_mask=self.config.use_arm_wise_suffix_attn,
        )
        
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        
        # Decode per-agent action outputs
        v_t = self._decode_action_outputs(suffix_out, boundaries)
        
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    def _decode_action_outputs(
        self, 
        suffix_out: jnp.ndarray, 
        boundaries: TokenBoundaries
    ) -> jnp.ndarray:
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
        num_agents = self.config.num_agents
        dim_per_agent = self.config.action_dim_per_agent
        
        for agent_id in range(num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                raise ValueError(f"Missing action_token_boundaries for agent {agent_id}")
            
            action_start, action_end = boundaries.action_token_boundaries[agent_id]
            agent_action_out = suffix_out[:, action_start:action_end, :]  # [B, horizon, width]
            
            agent_v_full = self.action_out_proj(agent_action_out)  # [B, horizon, action_dim]
            
            if agent_id < num_agents - 1:
                agent_v = agent_v_full[:, :, :dim_per_agent]
            else:
                remaining_dims = self.action_dim - agent_id * dim_per_agent
                agent_v = agent_v_full[:, :, :remaining_dims]
            
            agent_velocities.append(agent_v)
        
        v_t = jnp.concatenate(agent_velocities, axis=-1)
        
        return v_t

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=image_keys)
        
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Embed prefix with boundary tracking
        prefix_tokens, prefix_mask, prefix_ar_mask, boundaries = self.embed_prefix_with_boundaries(observation)
        
        # First fill KV cache with prefix using per-agent arm-wise masking (or full attention based on config)
        prefix_attn_mask = make_per_agent_prefix_attn_mask(
            prefix_mask,
            prefix_ar_mask,
            boundaries,
            num_agents=self.config.num_agents,
            agent_to_wrist_image=self._agent_to_wrist_image,
            apply_prefix_mask=self.config.use_arm_wise_prefix_attn,  # arm-wise or full attention for prefix
        )
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond, state_token_indices, action_token_boundaries = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            # Update boundaries with state token and action token info
            boundaries.state_token_indices = state_token_indices
            boundaries.num_state_tokens = self.config.num_agents if not self.pi05 else 0
            boundaries.action_token_boundaries = action_token_boundaries
            boundaries.action_horizon_per_agent = self.action_horizon
            
            suffix_len = suffix_tokens.shape[1]
            prefix_len = prefix_tokens.shape[1]
            
            # Create suffix self-attention mask with per-agent masking
            suffix_attn_mask = self._make_suffix_self_attn_mask(
                suffix_mask, suffix_ar_mask, boundaries, suffix_len
            )
            
            # Create cross-attention mask (suffix attending to prefix)
            cross_attn_mask = self._make_cross_attn_mask(
                prefix_mask, suffix_mask, boundaries, suffix_len
            )
            
            # Combine masks
            full_attn_mask = jnp.concatenate([cross_attn_mask, suffix_attn_mask], axis=-1)
            
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            
            # Decode per-agent action outputs
            v_t = self._decode_action_outputs(suffix_out, boundaries)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
    
    def _make_suffix_self_attn_mask(
        self,
        suffix_mask: jnp.ndarray,
        suffix_ar_mask: jnp.ndarray,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> jnp.ndarray:
        """Create suffix self-attention mask with per-agent decoupled masking.
        
        Pi0 suffix:  [state_0, state_1, action_0_t0...t49, action_1_t0...t49]
        Pi0.5 suffix: [action_0_t0...t49, action_1_t0...t49] (no state tokens)
        
        Each agent's action tokens can only see:
        - Their own state token (pi0 only, not present in pi0.5)
        - Their own previous action tokens (causally)
        
        When use_arm_wise_suffix_attn is False, returns the standard ar_mask-based mask.
        """
        batch_size = suffix_mask.shape[0]
        
        base_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        
        if not self.config.use_arm_wise_suffix_attn:
            return base_mask
        
        for agent_id in range(self.config.num_agents):
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
                
                allowed_mask = jnp.zeros(suffix_len, dtype=jnp.bool_)
                
                # Allow own state token if present (pi0 only)
                if agent_id in boundaries.state_token_indices:
                    agent_state_idx = boundaries.state_token_indices[agent_id]
                    allowed_mask = allowed_mask.at[agent_state_idx].set(True)
                
                # Allow own action tokens (causally)
                for key_idx in range(agent_action_start, query_idx + 1):
                    allowed_mask = allowed_mask.at[key_idx].set(True)
                
                allowed_mask_batched = jnp.logical_and(allowed_mask[None, :], suffix_mask)
                base_mask = base_mask.at[:, query_idx, :].set(allowed_mask_batched)
        
        # State token self-attention: each state only sees itself (pi0 only)
        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.state_token_indices:
                continue
            
            state_idx = boundaries.state_token_indices[agent_id]
            if state_idx >= suffix_len:
                continue
            
            allowed_mask = jnp.zeros(suffix_len, dtype=jnp.bool_)
            allowed_mask = allowed_mask.at[state_idx].set(True)
            allowed_mask_batched = jnp.logical_and(allowed_mask[None, :], suffix_mask)
            base_mask = base_mask.at[:, state_idx, :].set(allowed_mask_batched)
        
        return base_mask

    def _make_cross_attn_mask(
        self,
        prefix_mask: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        boundaries: TokenBoundaries,
        suffix_len: int,
    ) -> jnp.ndarray:
        """Create cross-attention mask for suffix attending to prefix with per-agent masking.
        
        Pi0 suffix:  [state_0, state_1, action_0_t0...t49, action_1_t0...t49]
        Pi0.5 suffix: [action_0_t0...t49, action_1_t0...t49]
        
        Each agent's state (if present) and action tokens can only attend to:
        - Global image tokens
        - Their own wrist image tokens
        - Their OWN prompt tokens (per-agent prompt separation)
        - LCT tokens (cross-arm collaboration bridge)
        
        When use_arm_wise_suffix_attn is False, all suffix tokens can attend to all prefix tokens.
        """
        batch_size = prefix_mask.shape[0]
        prefix_len = prefix_mask.shape[1]
        
        # Start with full cross attention (suffix can attend to all prefix)
        base_cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        
        if not self.config.use_arm_wise_suffix_attn:
            return base_cross_mask
        
        # Create per-agent prefix masks (reusable for state and action tokens)
        def create_agent_prefix_mask(agent_id):
            allowed_prefix_mask = jnp.zeros(prefix_len, dtype=jnp.bool_)
            
            # Allow global image
            global_start, global_end = boundaries.global_image_range
            allowed_prefix_mask = allowed_prefix_mask.at[global_start:global_end].set(True)
            
            # Allow own wrist image
            if agent_id in self._agent_to_wrist_image:
                wrist_name = self._agent_to_wrist_image[agent_id]
                if wrist_name in boundaries.image_boundaries:
                    wrist_start, wrist_end = boundaries.image_boundaries[wrist_name]
                    allowed_prefix_mask = allowed_prefix_mask.at[wrist_start:wrist_end].set(True)
            
            # Allow OWN prompt tokens only (per-agent prompt separation)
            if agent_id in boundaries.per_agent_language_ranges:
                agent_lang_start, agent_lang_end = boundaries.per_agent_language_ranges[agent_id]
                allowed_prefix_mask = allowed_prefix_mask.at[agent_lang_start:agent_lang_end].set(True)
            
            # Allow LCT tokens (cross-arm collaboration bridge)
            lct_start, lct_end = boundaries.latent_collab_range
            if lct_start < lct_end:
                allowed_prefix_mask = allowed_prefix_mask.at[lct_start:lct_end].set(True)
            
            return allowed_prefix_mask
        
        # Apply per-agent masking for state tokens (cross-attention to prefix)
        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.state_token_indices:
                continue
            
            state_idx = boundaries.state_token_indices[agent_id]
            if state_idx >= suffix_len:
                continue
            
            allowed_prefix_mask = create_agent_prefix_mask(agent_id)
            # Apply per-batch (each sample may have different prompt length)
            combined_mask_batched = jnp.logical_and(allowed_prefix_mask[None, :], prefix_mask)  # [B, prefix_len]
            base_cross_mask = base_cross_mask.at[:, state_idx, :].set(combined_mask_batched)
        
        # Apply per-agent masking for action tokens using action_token_boundaries
        for agent_id in range(self.config.num_agents):
            if agent_id not in boundaries.action_token_boundaries:
                continue
                
            agent_action_start, agent_action_end = boundaries.action_token_boundaries[agent_id]
            
            if agent_action_end > suffix_len:
                agent_action_end = suffix_len
            if agent_action_start >= suffix_len:
                continue
            
            allowed_prefix_mask = create_agent_prefix_mask(agent_id)
            # Apply per-batch (each sample may have different prompt length)
            combined_mask_batched = jnp.logical_and(allowed_prefix_mask[None, :], prefix_mask)  # [B, prefix_len]
            
            # Apply mask to this agent's action positions in cross attention
            for suffix_idx in range(agent_action_start, agent_action_end):
                base_cross_mask = base_cross_mask.at[:, suffix_idx, :].set(combined_mask_batched)
        
        return base_cross_mask

