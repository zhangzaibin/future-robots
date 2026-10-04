"""Pi0 Multi-Arm model with decoupled action experts."""

import dataclasses

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at


def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Sinusoidal position embedding."""
    sinusoid_input = jnp.outer(
        pos,
        jnp.power(
            jnp.exp(-jnp.log(max_period / min_period) / (embedding_dim // 2)),
            jnp.arange(embedding_dim // 2),
        ),
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def make_attn_mask(
    input_mask: at.Bool[at.Array, "b s"], ar_mask: at.Bool[at.Array, " s"]
) -> at.Bool[at.Array, "b s s"]:
    """Create attention mask."""
    return jnp.logical_and(
        input_mask[:, None, :],  # key mask
        jnp.logical_and(
            input_mask[:, :, None],  # query mask
            ar_mask[None, None, :],  # ar mask - fix indexing
        ),
    )


@dataclasses.dataclass(frozen=True)
class Pi0MultiConfig(_model.BaseModelConfig):
    """Configuration for Pi0 Multi-Arm model with decoupled action experts."""

    # Multi-arm specific parameters
    num_arms: int = 3  # Number of robot arms
    action_dim_per_arm: int = 8  # Action dimensions per arm

    # Model parameters (inherited from base)
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Keep action_dim = 32 to align with original settings
    # Each expert outputs 32 dims, but only uses its responsible slice
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 220

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_MULTI

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Multi":
        return Pi0Multi(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    "third_wrist_0_rgb": image_spec,  # Additional image for 3rd arm
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "third_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        if "lora" in self.paligemma_variant:
            filters.append(nnx.PathContains("PaliGemma.llm.layers.*.attn.*.lora"))
            filters.append(nnx.PathContains("PaliGemma.llm.layers.*.mlp.*.lora"))
        if "lora" in self.action_expert_variant:
            filters.append(nnx.PathContains("arm_experts.*.*.lora"))

        return nnx.All(*filters) if filters else nnx.Nothing()


class ArmActionExpert(nnx.Module):
    """Independent flow matching expert for each robot arm."""

    def __init__(self, config: Pi0MultiConfig, arm_id: int, rngs: nnx.Rngs):
        self.arm_id = arm_id
        self.config = config
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Each expert has complete flow matching pipeline (same as original pi0)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    def embed_suffix_for_arm(
        self,
        state_token: at.Float[at.Array, "b 1 d"],
        arm_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Generate suffix tokens for this arm using its own flow matching."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Add the shared state token
        tokens.append(state_token)
        input_mask.append(jnp.ones((state_token.shape[0], 1), dtype=jnp.bool_))
        # Image/language inputs do not attend to state or actions
        ar_mask += [True]

        # This expert's own flow matching processing
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # Each expert processes actions with its own layers
        action_tokens = self.action_in_proj(arm_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.config.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)

        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # Image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.config.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask


class Pi0Multi(_model.BaseModel):
    """Pi0 Multi-Arm model with decoupled action experts."""

    def __init__(self, config: Pi0MultiConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        # Shared vision-language encoder (same as original pi0)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

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

        # Global state projection (shared across all arms)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        # Multiple independent flow matching experts (one per arm)
        self.arm_experts = {f"arm{i+1}": ArmActionExpert(config, arm_id=i, rngs=rngs) for i in range(config.num_arms)}

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed the prefix (shared across all arms)."""
        # Same as original pi0 - process images and prompt
        batch_size = obs.state.shape[0]
        paligemma_config = _gemma.get_config(self.config.paligemma_variant)

        # Process images (same as original pi0)
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Add language (tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute loss using decoupled action experts."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # Use the correct image keys for pi0_multi (including third_wrist_0_rgb)
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=image_keys)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Shared prefix processing (vision-language understanding)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Generate shared state token once
        state_token = self.state_proj(observation.state)[:, None, :]

        # Each expert has independent flow matching
        total_loss = 0
        for arm_key, arm_expert in self.arm_experts.items():
            # Extract arm_id from key (e.g., "arm1" -> 0, "arm2" -> 1)
            arm_id = int(arm_key[3:]) - 1

            # Each expert does its own suffix embedding with its own flow matching layers
            suffix_tokens, suffix_mask, suffix_ar_mask = arm_expert.embed_suffix_for_arm(state_token, x_t, time)

            # Each expert has its own LLM forward pass
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
            )

            # Each expert predicts 32-dim but only uses first 8 dims
            v_t_full = arm_expert.action_out_proj(suffix_out[:, -self.action_horizon :])
            v_t_expert = v_t_full[..., : self.config.action_dim_per_arm]  # Take first 8 dims

            # Compare with corresponding slice from ground truth
            start_idx = arm_id * self.config.action_dim_per_arm
            end_idx = start_idx + self.config.action_dim_per_arm
            u_t_slice = u_t[..., start_idx:end_idx]

            # Compute loss for this expert
            arm_loss = jnp.mean(jnp.square(v_t_expert - u_t_slice), axis=-1)
            total_loss += arm_loss

        # Average loss across all arms
        return total_loss / len(self.arm_experts)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """Sample actions using decoupled experts."""
        # Use the correct image keys for pi0_multi (including third_wrist_0_rgb)
        image_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=image_keys)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # Initialize noise with full 32 dimensions (same as original pi0)
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Shared prefix processing (same as original pi0)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Generate shared state token once (same as in compute_loss)
        state_token = self.state_proj(observation.state)[:, None, :]

        def step(carry):
            x_t, time = carry

            # Each expert does independent flow matching (same as training)
            expert_velocities = []

            for arm_key, arm_expert in self.arm_experts.items():
                # Each expert processes the full x_t (same as training)
                suffix_tokens, suffix_mask, suffix_ar_mask = arm_expert.embed_suffix_for_arm(
                    state_token, x_t, jnp.broadcast_to(time, batch_size)
                )

                # Each expert has its own attention computation
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                # Each expert has its own LLM forward pass
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
                )
                assert prefix_out is None

                # Each expert predicts 32-dim but only uses first 8 dims (same as training)
                v_t_full = arm_expert.action_out_proj(suffix_out[:, -self.action_horizon :])
                v_t_expert = v_t_full[..., : self.config.action_dim_per_arm]  # Take first 8 dims
                expert_velocities.append(v_t_expert)

            # Combine velocities from all experts (each contributes 8 dims)
            combined_v_t = jnp.concatenate(expert_velocities, axis=-1)
            # Pad to full 32 dimensions if needed
            if combined_v_t.shape[-1] < self.config.action_dim:
                padding_size = self.config.action_dim - combined_v_t.shape[-1]
                padding = jnp.zeros((*combined_v_t.shape[:-1], padding_size))
                combined_v_t = jnp.concatenate([combined_v_t, padding], axis=-1)

            return x_t + dt * combined_v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        # Run sampling loop (same as original pi0)
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
