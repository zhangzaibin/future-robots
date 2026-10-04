import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class Pi0SkillLoRAComposerWeightLoader(WeightLoader):
    """Loads weights for Pi0SkillLoRAComposer from base pi0.5 checkpoints.

    Keeps randomly initialized: skill-LoRA banks, composer modules, LCT, time MLP, etc.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        missing_pattern = (
            ".*(?:lora|state_projs|latent_collab_tokens|time_mlp_in|time_mlp_out|final_norm_1|"
            "skill_lora|skill_router|skill_bank|w_a_bank|w_b_bank|"
            "lora_a_bank|lora_b_bank|gating_einsum_skill_lora|linear_skill_lora).*"
        )
        return _merge_params(loaded_params, params, missing_regex=missing_pattern)


@dataclasses.dataclass(frozen=True)
class MultiAgentAttnWeightLoader(WeightLoader):
    """Loads weights for Pi0MultiAgentAttn from base pi0/pi0.5 checkpoints.

    Handles loading from base pi0 checkpoint into either pi0 or pi0.5 multi-agent model.
    Shared projections (PaliGemma, action_in_proj, action_out_proj, etc.) are loaded
    from the pretrained checkpoint. Parameters not present in the base checkpoint are
    kept as randomly initialized:
    - state_projs: per-agent state projections (pi0 multi-agent)
    - latent_collab_tokens: LCT tokens for cross-arm collaboration
    - time_mlp_in, time_mlp_out: AdaRMS time MLP (pi0.5)
    - final_norm_1: additional RMSNorm in pi0.5 Gemma LLM
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        missing_pattern = ".*(?:lora|state_projs|latent_collab_tokens|time_mlp_in|time_mlp_out|final_norm_1).*"
        return _merge_params(loaded_params, params, missing_regex=missing_pattern)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0MoEWeightLoader(WeightLoader):
    """Loads weights for Pi0MoE model, copying action_out_proj weights to each expert.

    This loader:
    1. Loads the shared vision-language components (PaliGemma, state_proj, action_in_proj, etc.) from pretrained pi0
    2. Copies the original action_out_proj weights to each expert in action_out_proj_experts
    3. Handles proper initialization for multi-expert architecture
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load pretrained pi0 parameters
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Check if this is a Pi0MoE model by looking for action_out_proj_experts structure
        if "action_out_proj_experts" in params:
            # Handle Pi0MoE model specifically
            result_params = {}

            # First copy all non-expert parameters (shared components)
            for key, value in params.items():
                if key != "action_out_proj_experts":
                    if key in loaded_params:
                        result_params[key] = self._copy_nested_params(loaded_params[key], value)
                    else:
                        result_params[key] = value

            # Now handle action_out_proj_experts - copy action_out_proj weights to each expert
            if "action_out_proj_experts" in params:
                result_params["action_out_proj_experts"] = {}

                # Find the original action_out_proj weights from the loaded pi0 model
                if "action_out_proj" in loaded_params:
                    original_action_out_proj = loaded_params["action_out_proj"]

                    # Copy to each expert
                    for expert_key in params["action_out_proj_experts"]:
                        result_params["action_out_proj_experts"][expert_key] = self._copy_nested_params(
                            original_action_out_proj, params["action_out_proj_experts"][expert_key]
                        )
                        logger.info(f"Copied action_out_proj weights to {expert_key}")
                else:
                    # Fallback to keeping original expert weights if no action_out_proj found
                    result_params["action_out_proj_experts"] = params["action_out_proj_experts"]
                    logger.warning("No action_out_proj found in loaded params, keeping original expert weights")

            return result_params

        # Fall back to regular merge for non-Pi0MoE models
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")

    def _copy_nested_params(self, source, target):
        """Recursively copy parameters with proper shape handling."""
        if isinstance(target, dict):
            result = {}
            for key, target_value in target.items():
                if key in source:
                    result[key] = self._copy_nested_params(source[key], target_value)
                else:
                    result[key] = target_value
            return result

        if hasattr(target, "shape") and hasattr(source, "shape"):
            # Handle array-like objects with shapes
            if target.shape == source.shape:
                return source.astype(target.dtype)
            # Handle dimension mismatch by truncating or tiling
            logger.info(f"Shape mismatch: source {source.shape} vs target {target.shape}, adjusting...")
            return self._adjust_param_shape(source, target)

        # For other types, return target as fallback
        return target

    def _adjust_param_shape(self, source, target):
        """Adjust parameter shape when source and target have different shapes."""
        if len(source.shape) != len(target.shape):
            return target  # Cannot handle different number of dimensions

        # Handle each dimension
        adjusted = source
        for dim in range(len(source.shape)):
            if source.shape[dim] > target.shape[dim]:
                # Truncate if source is larger
                slices = [slice(None)] * len(source.shape)
                slices[dim] = slice(0, target.shape[dim])
                adjusted = adjusted[tuple(slices)]
            elif source.shape[dim] < target.shape[dim]:
                # Tile if source is smaller
                repeats = [1] * len(source.shape)
                repeats[dim] = (target.shape[dim] + source.shape[dim] - 1) // source.shape[dim]
                adjusted = np.tile(adjusted, repeats)
                # Truncate to exact size if needed
                slices = [slice(None)] * len(adjusted.shape)
                slices[dim] = slice(0, target.shape[dim])
                adjusted = adjusted[tuple(slices)]

        return adjusted.astype(target.dtype)


@dataclasses.dataclass(frozen=True)
class Pi0MultiWeightLoader(WeightLoader):
    """Loads weights for Pi0Multi model, copying action expert weights to each arm expert.

    This loader:
    1. Loads the shared vision-language components (PaliGemma, state_proj) from pretrained pi0
    2. Copies the original action expert weights to each arm expert
    3. Handles dimension mismatches by truncating or tiling weights appropriately
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load pretrained pi0 parameters
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # For Pi0Multi, we need to manually copy weights to avoid flattening issues with arm_experts list
        # Check if this is a Pi0Multi model by looking for arm_experts structure
        if "arm_experts" in params:
            # Handle Pi0Multi model specifically
            result_params = {}

            # First copy all non-arm_experts parameters (shared components)
            for key, value in params.items():
                if key != "arm_experts":
                    if key in loaded_params:
                        result_params[key] = self._copy_nested_params(loaded_params[key], value)
                    else:
                        result_params[key] = value

            # Now handle arm_experts - copy action expert weights to each arm expert
            if "arm_experts" in params and isinstance(params["arm_experts"], list):
                result_params["arm_experts"] = []

                # Find the original action expert weights from the loaded pi0 model
                action_expert_weights = {}
                action_expert_keys = ["action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj"]

                # The original pi0 model has these as top-level keys
                for key in action_expert_keys:
                    if key in loaded_params:
                        action_expert_weights[key] = loaded_params[key]

                # Copy to each arm expert
                for arm_idx, arm_expert_params in enumerate(params["arm_experts"]):
                    new_arm_expert = {}

                    for param_name, param_value in arm_expert_params.items():
                        if param_name in action_expert_weights:
                            # Copy from original action expert
                            new_arm_expert[param_name] = self._copy_nested_params(
                                action_expert_weights[param_name], param_value
                            )
                        else:
                            # Keep original parameter if not found in action expert
                            new_arm_expert[param_name] = param_value

                    result_params["arm_experts"].append(new_arm_expert)
            else:
                # If arm_experts is not a list or doesn't exist, keep original
                result_params["arm_experts"] = params.get("arm_experts", [])

            # Add any missing LoRA parameters from target
            for key, value in params.items():
                if key not in result_params:
                    if "lora" in key.lower():
                        result_params[key] = value

            return result_params

        # Fallback to standard loading for non-Pi0Multi models
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")

    def _copy_nested_params(self, source, target):
        """Recursively copy parameters from source to target, preserving structure."""
        if isinstance(target, dict) and isinstance(source, dict):
            result = {}
            for key, value in target.items():
                if key in source:
                    result[key] = self._copy_nested_params(source[key], value)
                else:
                    result[key] = value
            return result
        if hasattr(target, "shape") and hasattr(source, "shape"):
            # Both are arrays, check shape compatibility
            if source.shape == target.shape:
                return source.astype(target.dtype)
            return target
        # For other types, return target as fallback
        return target


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
