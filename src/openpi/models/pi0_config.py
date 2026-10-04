import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

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
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
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
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0DiffusionForcingConfig(Pi0Config):
    """
    Configuration for PI0 with Diffusion Forcing for multi-arm robustness.
    
    This extends Pi0Config with parameters for per-arm noise injection,
    which teaches the model to be robust to noisy observations from other arms.
    """
    
    # Multi-arm configuration
    num_arms: int = 4  # Number of robot arms
    arm_action_dim: int = 8  # Action dimension per arm
    
    # Diffusion Forcing parameters for ACTIONS
    # How much extra noise (time offset) for non-focal arms
    # 0 = original behavior, 0.3 = recommended
    noise_offset: float = 0.3
    # Probability of using per-arm noise vs uniform noise
    # 0 = original behavior, 0.7 = recommended
    per_arm_noise_prob: float = 0.7
    
    # State noise parameters (for observation robustness)
    # Whether to add noise to other arms' state observations
    add_state_noise: bool = True
    # Noise std for other arms' states (0 = no noise)
    state_noise_std: float = 0.05

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Use PI0 model type for compatibility with transforms
        return _model.ModelType.PI0

    def __post_init__(self):
        super().__post_init__()
        # Validate that action_dim is compatible with num_arms * arm_action_dim
        if self.num_arms * self.arm_action_dim > self.action_dim:
            import warnings
            warnings.warn(
                f"num_arms ({self.num_arms}) * arm_action_dim ({self.arm_action_dim}) = "
                f"{self.num_arms * self.arm_action_dim} > action_dim ({self.action_dim}). "
                f"Some arm dimensions may be truncated."
            )

    def load_pytorch(self, train_config, weight_path: str):
        """Load PI0DiffusionForcing model instead of PI0Pytorch."""
        import logging
        import safetensors.torch
        from openpi.models_pytorch.pi0_diffusion_forcing import PI0DiffusionForcing
        
        logger = logging.getLogger(__name__)
        logger.info(f"train_config: {train_config}")
        print("  [load_pytorch] Creating PI0DiffusionForcing model...", flush=True)
        
        model = PI0DiffusionForcing(config=train_config.model)
        print(f"  [load_pytorch] Model created, loading weights from {weight_path}...", flush=True)
        print("  [load_pytorch] This may take several minutes for large models...", flush=True)
        
        safetensors.torch.load_model(model, weight_path)
        print("  [load_pytorch] Weights loaded successfully!", flush=True)
        return model
