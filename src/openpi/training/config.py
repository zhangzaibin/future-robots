"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
from openpi.models.pi0_config import Pi0DiffusionForcingConfig
from openpi.models.pi0_fast import Pi0FASTConfig
from openpi.models.pi0_multi_agent_attn import Pi0MultiAgentAttnConfig
from openpi.models.pi0_skill_lora_composer import Pi0SkillLoRAComposerConfig
from openpi.training.weight_loaders import MultiAgentAttnWeightLoader
from openpi.training.weight_loaders import Pi0SkillLoRAComposerWeightLoader
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.umi_policy as umi_policy
import openpi.policies.so101_policy as so101_policy
import openpi.policies.bi_so100_policy as bi_so100_policy
import openpi.policies.ur_policy as _ur_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.paths as _paths
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

import matplotlib

matplotlib.use("Agg")  # 无界面安全绘图


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Video decoding backend for LeRobot datasets that store video files.
    # Set to "pyav" when torchcodec is unavailable (e.g. FFmpeg not installed).
    # If None, the lerobot default is used ("torchcodec" when available, else "pyav").
    video_backend: str | None = None

    # Timestamp tolerance (seconds) for video frame matching in LeRobot video datasets.
    # The default lerobot value (0.0001s) is too tight for real video files; set to
    # at least half the frame interval (e.g. 0.02 for 60 FPS, 0.04 for 30 FPS).
    video_tolerance_s: float | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class SO101TransformFactory(GroupFactory):
    """Creates data transforms for SO101 single-arm robot."""
    
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[so101_policy.SO101Inputs(model_type=model_config.model_type)],
            outputs=[so101_policy.SO101Outputs()],
        )


@dataclasses.dataclass(frozen=True)
class BiSO100TransformFactory(GroupFactory):
    """Creates data transforms for Bi-SO100 dual-arm robot (12 DoF).
    
    Bi-SO100 has 12 DoF total:
        - Left arm: 5 joint positions + 1 gripper = 6 dims
        - Right arm: 5 joint positions + 1 gripper = 6 dims
    
    Args:
        use_delta_joint_actions: If True, convert joint dimensions to delta actions
            (relative to current state). Gripper dimensions remain absolute.
    """
    
    use_delta_joint_actions: bool = False
    
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        base_group = _transforms.Group(
            inputs=[bi_so100_policy.BiSO100Inputs(model_type=model_config.model_type)],
            outputs=[bi_so100_policy.BiSO100Outputs()],
        )
        
        if self.use_delta_joint_actions:
            # Bi-SO100: 2 arms × (5 joints + 1 gripper)
            # Mask: 5 delta joints + 1 absolute gripper per arm
            delta_action_mask = _transforms.make_bool_mask(5, -1, 5, -1)
            base_group = base_group.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        return base_group


@dataclasses.dataclass(frozen=True)
class URTransformFactory(GroupFactory):
    """Creates data transforms for UR dual-arm EEF-space dataset.

    The UR dataset uses EEF-space (pos + axis_angle + gripper) state and relative
    rot6d actions. It has two wrist cameras and no global camera; the global camera
    slot is filled with a zero image and masked out.
    """

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[_ur_policy.URInputs()],
            outputs=[_ur_policy.UROutputs()],
        )


@dataclasses.dataclass(frozen=True)
class DroidMultiAgentTransformFactory(GroupFactory):
    """Creates data transforms for multi-agent Droid scenarios (pickle-friendly).
    
    This factory supports four modes:
    - use_atom_version=True, use_robotwin_format=False: DroidMultiInputs_atom + DroidMultiOutputs_atom (default)
    - use_atom_version=False, use_robotwin_format=False: DroidDualInputs + DroidMultiOutputs
    - use_robotwin_format=True: DroidDualInputs_atom_robotwin + DroidDualOutputs_atom_robotwin
    - use_biso100_format=True: BiSO100MultiAgentInputs + BiSO100Outputs (flat LeRobot keys)
    
    Args:
        num_agents: Number of robot agents (2-4)
        use_atom_version: If True, use DroidMultiInputs_atom/DroidMultiOutputs_atom; 
                          If False, use DroidDualInputs/DroidMultiOutputs
        use_robotwin_format: If True, use RobotWin data format (Aloha-style nested structure)
                            This overrides use_atom_version when True.
        shuffle_prob: Probability to shuffle agent order (for atom version and robotwin)
        image_mask_prob: Probability to mask images to zeros (for atom version and robotwin).
        mask_all_images: If True, each image is independently masked with image_mask_prob;
                         if False (default), only the global image can be masked.
        mask_random_one: If True, randomly select exactly one image to mask (global or any wrist)
                         instead of always masking the global image. Takes priority over mask_all_images.
        image_noise_prob: Probability to add Gaussian noise (for atom version and robotwin)
        task_instruction: Task instruction string (for non-atom version and robotwin)
        single_arm_id: ID of single arm to train (only for atom version)
        mask_multiple_images: Whether to allow masking multiple images at once (only for robotwin)
        max_masked_images: Maximum number of images to mask (only for robotwin)
        noise_std: Standard deviation of Gaussian noise (only for robotwin)
        use_delta_joint_actions: If True, convert joint dimensions to delta actions (relative to current state).
                                 Gripper dimensions remain absolute. This is important for "wait" behavior where
                                 delta_action = 0 means "stay still". Each agent has (joints_per_agent - 1) joints
                                 converted to delta, and 1 gripper kept absolute.
        joints_per_agent: Number of joint dimensions per agent (default: 7 for 7-DOF arm)
        use_phase_aware_delta: If True, use phase-aware delta action computation. For wait data (phase=wait_phase),
                               this ensures all actions in horizon are set to current state before computing delta,
                               resulting in delta=0. This handles wait data with dynamic states across frames.
        wait_phase: Phase value indicating wait data (default: 0). Used when use_phase_aware_delta=True.
        wait_augment_prob: Probability to randomly make one agent "wait" during training (data augmentation).
                          When triggered, one agent's instruction is changed to wait_instruction and its
                          action is set to current state (delta = 0 after DeltaActions transform).
        wait_instruction: Instruction text for the waiting agent (default: "Wait for the other robots.")
        action_dim_per_agent: Action dimension per agent (default: 8 = 7 joints + 1 gripper)
    """
    
    num_agents: int = 2
    use_atom_version: bool = True
    use_robotwin_format: bool = False  # New parameter for RobotWin data format
    use_biso100_format: bool = False  # Bi-SO100 flat LeRobot key format
    
    # Atom version parameters
    shuffle_prob: float = 0.0
    image_mask_prob: float = 0.0
    mask_all_images: bool = False  # If True, each image is independently masked; if False, only global image
    mask_random_one: bool = False  # If True, randomly select exactly one image to mask
    image_noise_prob: float = 0.0
    single_arm_id: int | None = None
    
    # Non-atom version parameters  
    task_instruction: str | None = "Put the object in the target location."
    
    # Override model_type (None means use model_config.model_type)
    model_type_override: _model.ModelType | None = None
    
    # RobotWin-specific parameters
    mask_multiple_images: bool = True  # Whether to allow masking multiple images at once
    max_masked_images: int = 3  # Maximum number of images to mask (1-3)
    noise_std: float = 0.02  # Standard deviation of Gaussian noise
    
    # Delta action parameters
    use_delta_joint_actions: bool = False  # Convert joint actions to delta (relative to current state)
    joints_per_agent: int = 7  # Number of joint dimensions per agent (gripper is +1, so total 8 per agent)
    
    # Phase-aware delta action parameters
    use_phase_aware_delta: bool = False  # If True, use phase-aware delta action computation for wait data
    wait_phase: int = 0  # Phase value that indicates wait data (default: 0)
    
    # Wait augmentation parameters (data augmentation to generate wait samples on-the-fly)
    wait_augment_prob: float = 0.0  # Probability to randomly make one agent "wait"
    wait_instruction: str = "Wait for the other robots."  # Instruction for waiting agent
    action_dim_per_agent: int = 8  # Action dimension per agent (7 joints + 1 gripper)

    # BiSO100-to-atom preprocessing (for SO101 flat format datasets)
    # When True, prepends BiSO100ToAtomTransform to the data_transforms inputs so the
    # conversion from flat state (image/wrist_image/state) → atom keys
    # (global_image/image_agent0/joint_position_agent{N}/gripper_position_agent{N})
    # runs during BOTH training and inference, instead of only during training via repack_transforms.
    needs_biso100_to_atom: bool = False
    biso100_joints_per_agent: int = 5  # joints per arm for SO101/SO100 (5 joints + 1 gripper = 6 DoF)

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        model_type = self.model_type_override if self.model_type_override is not None else model_config.model_type
        
        # Bi-SO100 format: flat LeRobot keys with shuffle support
        if self.use_biso100_format:
            base_group = _transforms.Group(
                inputs=[
                    bi_so100_policy.BiSO100MultiAgentInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_type,
                        shuffle_prob=self.shuffle_prob,
                        image_mask_prob=self.image_mask_prob,
                        image_noise_prob=self.image_noise_prob,
                        noise_std=self.noise_std,
                        mask_all_images=self.mask_all_images,
                        mask_random_one=self.mask_random_one,
                        action_dim_per_agent=self.action_dim_per_agent,
                        wait_augment_prob=self.wait_augment_prob,
                        wait_instruction=self.wait_instruction,
                    )
                ],
                outputs=[
                    bi_so100_policy.BiSO100Outputs()
                ],
            )
        # RobotWin format: use DroidDualInputs_atom_robotwin + DroidDualOutputs_atom_robotwin
        elif self.use_robotwin_format:
            base_group = _transforms.Group(
                inputs=[
                    droid_policy.DroidDualInputs_atom_robotwin(
                        action_dim=model_config.action_dim,
                        model_type=model_type,
                        shuffle_prob=self.shuffle_prob,
                        image_mask_prob=self.image_mask_prob,
                        image_noise_prob=self.image_noise_prob,
                        mask_multiple_images=self.mask_multiple_images,
                        max_masked_images=self.max_masked_images,
                        noise_std=self.noise_std,
                        mask_all_images=self.mask_all_images,
                        mask_random_one=self.mask_random_one,
                        task_instruction=self.task_instruction,
                        # Wait augmentation parameters
                        wait_augment_prob=self.wait_augment_prob,
                        wait_instruction=self.wait_instruction,
                        action_dim_per_agent=self.action_dim_per_agent,
                    )
                ],
                outputs=[
                    droid_policy.DroidDualOutputs_atom_robotwin()
                ],
            )
        elif self.use_atom_version:
            # Original atom version: DroidMultiInputs_atom + DroidMultiOutputs_atom
            base_group = _transforms.Group(
                inputs=[
                    droid_policy.DroidMultiInputs_atom(
                        action_dim=model_config.action_dim,
                        num_agents=self.num_agents,
                        model_type=model_type,
                        shuffle_prob=self.shuffle_prob,
                        image_mask_prob=self.image_mask_prob,
                        image_noise_prob=self.image_noise_prob,
                        mask_all_images=self.mask_all_images,
                        mask_random_one=self.mask_random_one,
                        single_arm_id=self.single_arm_id,
                        # Wait augmentation parameters
                        wait_augment_prob=self.wait_augment_prob,
                        wait_instruction=self.wait_instruction,
                        action_dim_per_agent=self.action_dim_per_agent,
                    )
                ],
                outputs=[
                    droid_policy.DroidMultiOutputs_atom(
                        num_agents=self.num_agents,
                        single_arm_id=self.single_arm_id,
                    )
                ],
            )
        else:
            # Non-atom version: DroidDualInputs + DroidMultiOutputs
            base_group = _transforms.Group(
                inputs=[
                    droid_policy.DroidMultiInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_type,
                        task_instruction=self.task_instruction,
                    )
                ],
                outputs=[
                    droid_policy.DroidMultiOutputs(num_agents=self.num_agents)
                ],
            )
        
        # Prepend BiSO100ToAtomTransform when the dataset stores flat SO101 keys
        # (image, wrist_image, image_agent1, state) instead of atom keys.
        # This must be part of data_transforms (not repack_transforms) so it runs
        # during both training AND inference.
        if self.needs_biso100_to_atom and self.use_atom_version:
            base_group = _transforms.Group(
                inputs=[bi_so100_policy.BiSO100ToAtomTransform(joints_per_agent=self.biso100_joints_per_agent), *base_group.inputs],
                outputs=base_group.outputs,
            )

        # Add delta action transform if requested
        # For each agent: joints_per_agent joints (delta) + 1 gripper (absolute)
        # Example for 2 agents with 7 joints each: make_bool_mask(7, -1, 7, -1)
        # Result: [T,T,T,T,T,T,T,F, T,T,T,T,T,T,T,F] - 7 delta joints + 1 abs gripper per agent
        if self.use_delta_joint_actions:
            # Build mask: for each agent, joints_per_agent True + 1 False (for gripper)
            mask_args = []
            for _ in range(self.num_agents):
                mask_args.extend([self.joints_per_agent, -1])  # 7 joints delta, 1 gripper absolute
            delta_action_mask = _transforms.make_bool_mask(*mask_args)
            
            # Choose between phase-aware or standard delta actions
            if self.use_phase_aware_delta:
                # Use phase-aware transforms that handle wait data (phase=wait_phase) specially
                base_group = base_group.push(
                    inputs=[_transforms.DeltaActionsWithPhase(
                        mask=delta_action_mask,
                        wait_phase=self.wait_phase
                    )],
                    outputs=[_transforms.AbsoluteActionsWithPhase(
                        mask=delta_action_mask,
                        wait_phase=self.wait_phase
                    )],
                )
            else:
                # Use standard delta action transforms
                base_group = base_group.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
        
        return base_group


@dataclasses.dataclass(frozen=True)
class SkillLoRAModelTransformFactory(GroupFactory):
    """Model transforms for Pi0 Skill-LoRA Composer with per-agent prompts.

    Args:
        default_prompt: Optional default prompt injected when data has no prompt.
        skill_labels_path: Path to a ``skill_labels.jsonl`` file that maps
            ``task_index`` → per-agent ``skill_ids``.  When provided, a
            ``SkillLabelFromTaskIndex`` transform is prepended so that the router
            receives real supervision labels instead of the ``-1`` fallback.
    """

    default_prompt: str | None = None
    skill_labels_path: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        from openpi.models.pi0_skill_lora_composer import Pi0SkillLoRAComposerConfig

        if not isinstance(model_config, Pi0SkillLoRAComposerConfig):
            raise ValueError(
                f"SkillLoRAModelTransformFactory requires Pi0SkillLoRAComposerConfig, got {type(model_config)}"
            )
        if not model_config.use_per_agent_prompt:
            raise ValueError("SkillLoRAModelTransformFactory requires use_per_agent_prompt=True")

        max_len_per_agent = model_config.get_max_token_len_per_agent()
        num_agents = model_config.num_agents

        inputs: list = [_transforms.InjectDefaultPrompt(self.default_prompt)]

        if self.skill_labels_path is not None:
            inputs.append(_transforms.SkillLabelFromTaskIndex.from_jsonl(self.skill_labels_path))

        inputs += [
            _transforms.EnsureSkillFields(
                num_agents=num_agents,
                num_skills=model_config.num_skills,
            ),
            _transforms.TokenizePerAgentPrompt(
                _tokenizer.PaligemmaPerAgentTokenizer(
                    max_len_per_agent=max_len_per_agent,
                    num_agents=num_agents,
                    action_dim_per_agent=model_config.action_dim_per_agent,
                ),
                discrete_state_input=model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(model_config.action_dim),
        ]

        return _transforms.Group(inputs=inputs)


@dataclasses.dataclass(frozen=True)
class PerAgentPromptModelTransformFactory(GroupFactory):
    """Creates model transforms for Pi0 Multi-Agent Attention models with per-agent prompts.
    
    This factory uses the PaligemmaPerAgentTokenizer which splits prompts by "|"
    and tokenizes each agent's portion independently.
    
    Example prompt format: "Pick up red block | Place blue cube"
    Each agent's portion is padded to max_token_len_per_agent independently.
    """
    
    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # Import here to avoid circular imports
        from openpi.models.pi0_multi_agent_attn import Pi0MultiAgentAttnConfig
        
        if not isinstance(model_config, Pi0MultiAgentAttnConfig):
            raise ValueError(
                f"PerAgentPromptModelTransformFactory requires Pi0MultiAgentAttnConfig, "
                f"got {type(model_config)}"
            )
        
        if not model_config.use_per_agent_prompt:
            raise ValueError(
                "PerAgentPromptModelTransformFactory requires use_per_agent_prompt=True in model config"
            )
        
        max_len_per_agent = model_config.get_max_token_len_per_agent()
        num_agents = model_config.num_agents
        
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.TokenizePerAgentPrompt(
                    _tokenizer.PaligemmaPerAgentTokenizer(
                        max_len_per_agent=max_len_per_agent,
                        num_agents=num_agents,
                        action_dim_per_agent=model_config.action_dim_per_agent,
                    ),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(
                    model_config,
                    (pi0_config.Pi0Config, Pi0MultiAgentAttnConfig, Pi05TwinVLAConfig),
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _model.ModelType.PI0_MULTI:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            # TODO: This part means we are using the same model config for PI0_MOE and PI0, but they are different models.
            # TODO: We need to change this part to use different model configs for PI0_MOE and PI0.
            case _model.ModelType.PI0:  # Handle PI0_MOE same as PI0
                if hasattr(model_config, "num_arms"):  # Check if it's PI0_MOE
                    return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            _transforms.TokenizePrompt(
                                _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            ),
                        ],
                    )
                # Otherwise fall through to regular PI0 handling


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        # If asset_id is explicitly set (even if empty string), use it; otherwise use repo_id
        asset_id = self.assets.asset_id if self.assets.asset_id is not None else repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class UmiBimanualDataConfig(DataConfigFactory):
    """Data config for bimanual UMI datasets converted from dataset.zarr.zip.

    The LeRobot dataset stores absolute axis-angle poses (camera0_rgb, camera1_rgb,
    state(26,), actions(14,)).  UmiInputs converts these to the model format:
      - state  → 32D (grips + wrt_start rotations + cross-robot relative poses)
      - actions → (H,20) relative SE3 pose10d matching HarMI's bimanual DP output
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack: rename LeRobot dataset keys to inference-side keys.
        # The LeRobot dataset uses plain names; we keep them identical so the
        # UmiInputs transform can address them directly (no renaming needed
        # beyond mapping 'task' → 'prompt').
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "camera0_rgb": "camera0_rgb",
                        "camera1_rgb": "camera1_rgb",
                        "state": "state",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputs(model_type=model_config.model_type)],
            outputs=[umi_policy.UmiOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPlaceMeatDataConfig(DataConfigFactory):
    """
    Config for place_meat_order dataset with 2 agents.
    This dataset has keys: image_agent0, image_agent1, global_image, joint_position_agent0, etc.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform: 将数据集的键映射到标准键
        # 数据集键 -> 标准键（用于后续处理）
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # RepackTransform 映射方向：新键 -> 旧键路径
                        # 新键 "image" 从数据集的 "global_image" 获取
                        "image": "global_image",
                        # 新键 "wrist_image" 从数据集的 "image_agent0" 获取
                        "wrist_image": "image_agent0",
                        # 保留 image_agent1 用于后续处理
                        "image_agent1": "image_agent1",
                        # 保留其他键
                        "joint_position_agent0": "joint_position_agent0",
                        "joint_position_agent1": "joint_position_agent1",
                        "gripper_position_agent0": "gripper_position_agent0",
                        "gripper_position_agent1": "gripper_position_agent1",
                        "actions": "actions",
                        # 新键 "prompt" 从数据集的 "task" 获取
                        "prompt": "task",
                    }
                )
            ]
        )

        # 使用自定义的 PlaceMeatInputs 和 PlaceMeatOutputs
        from openpi.policies.place_meat_policy import PlaceMeatInputs
        from openpi.policies.place_meat_policy import PlaceMeatOutputs

        data_transforms = _transforms.Group(
            inputs=[PlaceMeatInputs(model_type=model_config.model_type)],
            outputs=[PlaceMeatOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    # {
                    #     "observation/exterior_image_1_left": "observation/image",
                    #     "observation/wrist_image_left": "observation/wrist_image",
                    #     "observation/joint_position": "observation/joint_position",
                    #     "observation/gripper_position": "observation/gripper_position",
                    #     "actions": "actions",
                    #     "prompt": "prompt",
                    # }
                    {
                        "observation/exterior_image_1_left": "image",
                        "observation/wrist_image_left": "wrist_image",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )




@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = _paths.checkpoint_path("")

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # MACG
    # two_arms_stack_bowl
    TrainConfig(
        name="aevla_pi0_two_arms_stack_bowls",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_stack_bowl'),
                asset_id="two_arms_stack_bowl_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="aevla_pi05_two_arms_stack_bowls",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            # max_token_len auto-set by __post_init__ (200 for pi05)
            # For per-agent prompt: total = num_agents * max_token_len_per_agent = 2 * 100 = 200
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_stack_bowl'),
                asset_id="two_arms_stack_bowl_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi0_two_arms_stack_bowls",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_stack_bowl'),
                asset_id="two_arms_stack_bowl_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.3,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_two_arms_stack_bowls",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_stack_bowl'),
                asset_id="two_arms_stack_bowl_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # two_arms_stack_cube
    TrainConfig(
        name="aevla_pi05_two_arms_stack_cubes",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            # max_token_len auto-set by __post_init__ (200 for pi05)
            # For per-agent prompt: total = num_agents * max_token_len_per_agent = 2 * 100 = 200
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_handover_bottles/two_arms_handover_bottles_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_handover_bottles'),
                asset_id="two_arms_handover_bottles_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="aevla_pi05_skill_lora_two_arms_multitask",
        model=Pi0SkillLoRAComposerConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_arm_wise_suffix_attn=True,
            use_latent_collab_tokens=False,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
            num_skills=10,
            skill_lora_rank=4,
            skill_lora_alpha=16.0,
            router_loss_weight=0.1,
            router_use_text_only=True,
            router_per_arm=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI05,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=SkillLoRAModelTransformFactory(
                skill_labels_path=_paths.dataset_path('multi_task_combined/meta/skill_labels.jsonl'),
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=Pi0SkillLoRAComposerWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="aevla_pi05_skill_lora_two_arms_multitask_full_attn",
        model=Pi0SkillLoRAComposerConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=False,
            use_arm_wise_suffix_attn=False,
            use_latent_collab_tokens=False,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
            num_skills=10,
            skill_lora_rank=4,
            skill_lora_alpha=16.0,
            router_loss_weight=0.1,
            router_use_text_only=True,
            router_per_arm=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI05,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=SkillLoRAModelTransformFactory(
                skill_labels_path=_paths.dataset_path('multi_task_combined/meta/skill_labels.jsonl'),
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=Pi0SkillLoRAComposerWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="aevla_pi05_two_arms_multi_task_nowait_15k",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_two_arms_stack_cubes",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_stack_cube/two_arms_stack_cube_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_stack_cube'),
                asset_id="two_arms_stack_cube_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # two_arms_push_cube
    TrainConfig(
        name="aevla_pi05_two_arms_push_cubes",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            # max_token_len auto-set by __post_init__ (200 for pi05)
            # For per-agent prompt: total = num_agents * max_token_len_per_agent = 2 * 100 = 200
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_push_cube/two_arms_push_cube_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_push_cube'),
                asset_id="two_arms_push_cube_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_two_arms_push_cubes",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_push_cube/two_arms_push_cube_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_push_cube'),
                asset_id="two_arms_push_cube_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # multi_task (stack_bowls + stack_cubes + push_cube combined)
    # image_mask_prob=0.6 
    # wait_augment_prob=0.3
    TrainConfig(
        name="aevla_pi05_two_arms_multi_task",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # Ablation: full attention (prefix=False, suffix=False) — both sides use standard ar_mask-based attention
    TrainConfig(
        name="aevla_pi05_two_arms_multi_task_full_attn",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=False,
            use_arm_wise_suffix_attn=False,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # Ablation: prefix-only arm-wise attention (prefix=True, suffix=False)
    TrainConfig(
        name="aevla_pi05_two_arms_multi_task_prefix_only",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_arm_wise_suffix_attn=False,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # Ablation: suffix-only arm-wise attention (prefix=False, suffix=True)
    TrainConfig(
        name="aevla_pi05_two_arms_multi_task_suffix_only",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=False,
            use_arm_wise_suffix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # three_arms_place_objects_combined (pi05, per-agent prompt, delta joint actions)
    TrainConfig(
        name="aevla_pi05_three_arms_place_objects",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=3,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
                (2, "third_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('three_arms_place_objects_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="three_arms_place_objects_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=50,
        save_interval=20000,
        keep_period=20000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # three_arms_place_objects_combined (pi05, standard Pi0Config, delta joint actions)
    TrainConfig(
        name="pi05_three_arms_place_objects",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('three_arms_place_objects_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="three_arms_place_objects_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=50,
        save_interval=20000,
        keep_period=20000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # three_arms_place_objects_combined (pi0, standard Pi0Config, delta joint actions)
    TrainConfig(
        name="pi0_three_arms_place_objects",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('three_arms_place_objects_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="three_arms_place_objects_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=50,
        save_interval=20000,
        keep_period=20000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # three_arms_place_objects_combined (pi0fast, Pi0FASTConfig, delta joint actions)
    TrainConfig(
        name="pi0fast_three_arms_place_objects",
        model=Pi0FASTConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=300,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('three_arms_place_objects_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="three_arms_place_objects_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0_fast")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=20000,
        batch_size=32,
        log_interval=50,
        save_interval=20000,
        keep_period=20000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_two_arms_multi_task",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_two_arms_multi_task_mavla_shuffle06_mask04",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.6,
                image_mask_prob=0.4,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_single_arm_left_multi_task",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined_left",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=False,
                single_arm_id=0,
                model_type_override=_model.ModelType.PI05,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_single_arm_right_multi_task",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined_right",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=False,
                single_arm_id=1,
                model_type_override=_model.ModelType.PI05,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi0_two_arms_multi_task",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi0fast_two_arms_multi_task",
        model=Pi0FASTConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=250,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('multi_task_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="multi_task_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                use_delta_joint_actions=True,
                joints_per_agent=7,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0_fast")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),


    # two_arms_handover_bottles
    TrainConfig(
        name="macg_two_arms_handover_bottles_pi0_moa_wait00_mask06_delta",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_handover_bottles/two_arms_handover_bottles_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_handover_bottles'),
                asset_id="two_arms_handover_bottles_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.6,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.0,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # two_arms_handover_bottles
    TrainConfig(
        name="two_arm_infer",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_handover_bottles/two_arms_handover_bottles_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_handover_bottles'),
                asset_id="two_arms_handover_bottles_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.0,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="two_arm_infer_lct",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
            use_latent_collab_tokens=True,
            num_latent_collab_tokens=32,  
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('MACG/two_arms_handover_bottles/two_arms_handover_bottles_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('MACG/two_arms_handover_bottles'),
                asset_id="two_arms_handover_bottles_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.0,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # three_arms_stack_cubes
    TrainConfig(
        name="macg_three_arms_stack_cubes_pi0_moa_wait03_mask05_delta",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=30,
            # Total max_token_len = num_agents * max_token_len_per_agent = 3 * 48 = 144
            max_token_len=144,
            num_agents=3,
            action_dim_per_agent=8,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
                (2, "third_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('stack_three_cubes_3arms'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="stack_three_cubes_3arms",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.5,
                model_type_override=_model.ModelType.PI0,
                # Enable delta action for joints: delta_action = action - state
                # This makes "wait" data work correctly: delta = 0 means stay still
                use_delta_joint_actions=True,
                joints_per_agent=7,  # 7 joints + 1 gripper = 8 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.9,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=8,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # three_arms_place_objectsk
    TrainConfig(
        name="macg_three_arms_place_objects_pi0_moa_wait03_mask05_delta",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=250,
            num_agents=3,
            action_dim_per_agent=8,
            # Mapping: agent 0 -> left wrist, agent 1 -> right wrist, agent 2 -> third wrist
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
                (2, "third_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('placefactory_new/placefactory_combined'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('placefactory_new'),
                asset_id="placefactory_combined",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=3,
                use_atom_version=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI0,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=5000,
        keep_period=20000,
        num_workers=1,
        fsdp_devices=2,
    ),

    




    # ============================================================================
    # Pi0 Multi-Agent Attention with RobotWin Data Format
    # Uses Aloha-style nested data structure from RobotWin dataset
    # Action dim per agent: 7 (6 joints + 1 gripper)
    # ============================================================================
    TrainConfig(
        name="pi0_multi_agent_attn_2arms_per_agent_prompt_blocksrankingrobotwin",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=7,  # RobotWin uses 7 dims (6 joints + 1 gripper)
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('robotwin_lerobot/blocks_ranking_rgb_phase'),  # TODO: Update path
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('robotwin_lerobot'),  # TODO: Update path
                asset_id="blocks_ranking_rgb_phase",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_robotwin_format=True,  # Enable RobotWin data format
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=6,  # 6 joints + 1 gripper = 7 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.3,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=7,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
                # RobotWin dataset uses 'action' instead of 'actions'
                action_sequence_keys=("action",),
                # Repack flat LeRobot keys to nested structure for DroidDualInputs_atom_robotwin
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation": {
                                    "state": "observation.state",
                                    "images": {
                                        "cam_high": "observation.images.cam_high",
                                        "cam_left_wrist": "observation.images.cam_left_wrist",
                                        "cam_right_wrist": "observation.images.cam_right_wrist",
                                    },
                                },
                                "action": "action",
                            }
                        )
                    ]
                ),
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="robotwin_infer",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            # Total max_token_len = num_agents * max_token_len_per_agent = 2 * 48 = 96
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=7,  # RobotWin uses 7 dims (6 joints + 1 gripper)
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            # Per-agent prompt settings
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,  # Each agent gets 48 tokens
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('robotwin_lerobot/blocks_ranking_rgb_phase'),  # TODO: Update path
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('robotwin_lerobot'),  # TODO: Update path
                asset_id="blocks_ranking_rgb_phase",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_robotwin_format=True,  # Enable RobotWin data format
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=6,  # 6 joints + 1 gripper = 7 dims per agent
                # Wait augmentation: randomly make one agent "wait" during training
                # This generates wait samples on-the-fly without pre-generated wait data
                wait_augment_prob=0.0,  # 30% probability to make one agent wait
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=7,
            ),
            # Use PerAgentPromptModelTransformFactory for per-agent prompt tokenization
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
                # RobotWin dataset uses 'action' instead of 'actions'
                action_sequence_keys=("action",),
                # Repack flat LeRobot keys to nested structure for DroidDualInputs_atom_robotwin
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation": {
                                    "state": "observation.state",
                                    "images": {
                                        "cam_high": "observation.images.cam_high",
                                        "cam_left_wrist": "observation.images.cam_left_wrist",
                                        "cam_right_wrist": "observation.images.cam_right_wrist",
                                    },
                                },
                                "action": "action",
                            }
                        )
                    ]
                ),
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # ============================================================================
    # Pi0.5 Multi-Agent Attention with RobotWin Data Format
    # Pi0.5 variant: discrete state input, AdaRMS time conditioning
    # No per-agent state tokens in suffix; state is part of discrete language prefix
    # ============================================================================
    TrainConfig(
        name="pi05_multi_agent_attn_2arms_per_agent_prompt_robotwin",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            # max_token_len auto-set by __post_init__ (200 for pi05)
            # For per-agent prompt: total = num_agents * max_token_len_per_agent = 2 * 100 = 200
            max_token_len=200,
            num_agents=2,
            action_dim_per_agent=7,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=100,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('robotwin_lerobot/blocks_ranking_rgb_phase'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('robotwin_lerobot'),
                asset_id="blocks_ranking_rgb_phase",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_robotwin_format=True,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                model_type_override=_model.ModelType.PI05,
                use_delta_joint_actions=True,
                joints_per_agent=6,
                wait_augment_prob=0.3,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=7,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation": {
                                    "state": "observation.state",
                                    "images": {
                                        "cam_high": "observation.images.cam_high",
                                        "cam_left_wrist": "observation.images.cam_left_wrist",
                                        "cam_right_wrist": "observation.images.cam_right_wrist",
                                    },
                                },
                                "action": "action",
                            }
                        )
                    ]
                ),
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # realworld experiment - SO101 single-arm robot
    # NOTE: The dataset should be converted from LeRobot v3.0 to v2.1 format first using:
    #   python scripts/convert_lerobot_v3_to_v2.py \
    #     --src dataset/realworld/cup2basket \
    #     --dst dataset/realworld/so101/cup2basket_v2
    TrainConfig(
        name="pi0_so101",
        model=pi0_config.Pi0Config(
            action_dim=32,  # SO101 has 6 DoF (5 joints + 1 gripper)
            action_horizon=150,
        ),
        data=SimpleDataConfig(
            # Path to the converted v2.1 format dataset
            repo_id=_paths.dataset_path('realworld/so101/cup2basket_v2'),
            assets=AssetsConfig(
                # Assets directory containing norm_stats.json
                assets_dir=_paths.dataset_path('realworld/so101'),
                asset_id="cup2basket_v2",
            ),
            data_transforms=SO101TransformFactory(),
            # Set default_prompt since the dataset uses prompt_from_task
            model_transforms=ModelTransformFactory(
                default_prompt="Put the cup in the basket"
            ),
            base_config=DataConfig(
                # Repack transforms to map dataset keys to model keys
                # After conversion, the dataset has: image, wrist_image, state, actions
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                # Map image to observation/image
                                "observation/image": "image",
                                # Map wrist_image to observation/wrist_image
                                "observation/wrist_image": "wrist_image",
                                # Map state to observation/state
                                "observation/state": "state",
                                # Map actions
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,  # Use task from dataset metadata
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # ============================================================================
    # Bi-SO100 Dual-Arm Robot Config (12 DoF)
    # ============================================================================

    # Stackbowls task
    TrainConfig(
        name="pi0_bi_so100_stackbowls",
        model=pi0_config.Pi0Config(
            action_dim=32,  # Bi-SO100 has 12 DoF (6 per arm), padded to 32
            action_horizon=50,
        ),
        data=SimpleDataConfig(
            # Path to the converted v2.1 format dataset
            repo_id=_paths.dataset_path('realworld/stackbowls_v2'),
            assets=AssetsConfig(
                # Assets directory containing norm_stats.json
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="stackbowls_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,  # Predict relative/delta actions
            ),
            # Set default_prompt since the dataset uses prompt_from_task
            model_transforms=ModelTransformFactory(
                default_prompt="Stack the bowls"
            ),
            base_config=DataConfig(
                # Repack transforms to map dataset keys to model keys
                # After conversion, the dataset has: image, wrist_image, image_agent1, state, actions
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                # Map image (global camera) to observation/image
                                "observation/image": "image",
                                # Map wrist_image (left wrist) to observation/wrist_image
                                "observation/wrist_image": "wrist_image",
                                # Map image_agent1 (right wrist) to observation/image_agent1
                                "observation/image_agent1": "image_agent1",
                                # Map state to observation/state
                                "observation/state": "state",
                                # Map actions
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,  # Use task from dataset metadata
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi0_bi_so100_moa_shuffle",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
            num_agents=2,
            action_dim_per_agent=6,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),
                (1, "right_wrist_0_rgb"),
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=48,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('realworld/stackbowls_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="stackbowls_v2",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_biso100_format=True,
                shuffle_prob=1.0,
                image_mask_prob=0.2,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi0_bi_so100_baseline_shuffle",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('realworld/stackbowls_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="stackbowls_v2",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_biso100_format=True,
                shuffle_prob=1.0,
                image_mask_prob=0.2,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="Stack the bowls"
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=30000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    # Atom-action version of the bi-SO100 stackbowls config.
    # Uses DroidMultiInputs_atom (phase-aware per-agent instructions) while reading
    # the flat stackbowls_v2 dataset via BiSO100ToAtomTransform which:
    #   1. Renames image/wrist_image/image_agent1 → global_image/image_agent0/image_agent1
    #   2. Splits flat 12-dim state → per-agent joint_position/gripper_position keys
    # The phase column already written by set_phases_stackbowls.py is passed through
    # so prompt_from_task can pick up per-phase instructions from the task string.
    TrainConfig(
        name="pi0_bi_so100_atom_shuffle_stackbowls",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('realworld/stackbowls_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="stackbowls_v2",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                use_biso100_format=False,
                shuffle_prob=1.0,
                image_mask_prob=0.2,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="Stack the bowls"
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        bi_so100_policy.BiSO100ToAtomTransform(joints_per_agent=5)
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # Place two cubes task
    TrainConfig(
        name="pi0_bi_so100_placetwocubes",
        model=pi0_config.Pi0Config(
            action_dim=32,  # Bi-SO100 has 12 DoF (6 per arm), padded to 32
            action_horizon=50,
        ),
        data=SimpleDataConfig(
            # Path to the converted v2.1 format dataset
            repo_id=_paths.dataset_path('realworld/place_two_cubes_v2'),
            assets=AssetsConfig(
                # Assets directory containing norm_stats.json
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="place_two_cubes_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,  # Predict relative/delta actions
            ),
            # Set default_prompt since the dataset uses prompt_from_task
            model_transforms=ModelTransformFactory(
                default_prompt="Use both arms to put the blue cube and the green block into the bowl."
            ),
            base_config=DataConfig(
                # Repack transforms to map dataset keys to model keys
                # After conversion, the dataset has: image, wrist_image, image_agent1, state, actions
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                # Map image (global camera) to observation/image
                                "observation/image": "image",
                                # Map wrist_image (left wrist) to observation/wrist_image
                                "observation/wrist_image": "wrist_image",
                                # Map image_agent1 (right wrist) to observation/image_agent1
                                "observation/image_agent1": "image_agent1",
                                # Map state to observation/state
                                "observation/state": "state",
                                # Map actions
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,  # Use task from dataset metadata
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # SO101-compatible dual-arm test setup (12 DoF)
    TrainConfig(
        name="pi0_bi_so100_flodtowel",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('foldtowel_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="foldtowel_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="test",
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=30000,
        batch_size=48,
        log_interval=50,
        save_interval=30000,
        keep_period=30000,
        num_workers=1,
        fsdp_devices=2,
    ),

    TrainConfig(
        name="pi05_bi_so100_flodtowel",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('foldtowel_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="foldtowel_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="fold the towel",
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=30000,
        batch_size=48,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    TrainConfig(
        name="pi05_bi_so100_flodtowel_pretrained_3k",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('foldtowel_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="foldtowel_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="fold the towel",
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _paths.checkpoint_path('pi05_bi_so100_flodtowel_nomem/so101_test/14999/params')
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=3000,
            decay_lr=1e-6,
        ),
        num_train_steps=10000,
        batch_size=48,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # Single-GPU fine-tuning initialized from exported pretrained model parameters.
    # Optimizer state and the learning-rate schedule start from step zero.
    TrainConfig(
        name="pi05_bi_so100_flodtowel_pretrained_3k_1gpu",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('foldtowel_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="foldtowel_v2",
            ),
            data_transforms=BiSO100TransformFactory(
                use_delta_joint_actions=True,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="fold the towel",
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/image_agent1": "image_agent1",
                                "observation/state": "state",
                                "actions": "actions",
                            }
                        )
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _paths.checkpoint_path('pi05_bi_so100_flodtowel/so101_test/14999/params')
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=3000,
            decay_lr=1e-6,
        ),
        num_train_steps=10000,
        batch_size=32,
        log_interval=50,
        save_interval=1000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=1,
    ),

    TrainConfig(
        name="pi0_bi_so100_atom_shuffle_placecubes",
        model=pi0_config.Pi0Config(
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('realworld/place_two_cubes_v2'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('realworld'),
                asset_id="place_two_cubes_v2",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                use_biso100_format=False,
                shuffle_prob=1.0,
                image_mask_prob=0.4,
                mask_all_images=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
            ),
            model_transforms=ModelTransformFactory(
                default_prompt="Use both arms to put the blue cube and the green block into the bowl."
            ),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        bi_so100_policy.BiSO100ToAtomTransform(joints_per_agent=5)
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi0")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5000,
            decay_lr=1e-6,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # -----------------------------------------------------------------------
    # SO101 real-world multi-task: push cubes + stack bowls + stack cubes
    # Dataset: dataset/macg_so101/macg_multitask_phased
    #   91 episodes, 42 669 frames, 15 task strings (3 tasks × 5 phases)
    # Phase labeling: scripts/so101/multitask/set_phases_so101_v2.py
    # Merge:          scripts/set_atom/macg/merge_multitask_datasets.py
    # Pipeline:       scripts/so101/multitask/run_phase_and_merge.sh
    # -----------------------------------------------------------------------
    TrainConfig(
        name="aevla_pi05_so101_multitask",
        model=Pi0MultiAgentAttnConfig(
            action_dim=32,
            action_horizon=50,
            pi05=True,
            max_token_len=200,
            num_agents=2,
            # SO101/SO100: 5 joints + 1 gripper = 6 dims per agent
            action_dim_per_agent=6,
            agent_to_wrist_image=(
                (0, "left_wrist_0_rgb"),   # left wrist
                (1, "right_wrist_0_rgb"),  # right wrist
            ),
            global_image_key="base_0_rgb",
            use_arm_wise_prefix_attn=True,
            use_per_agent_prompt=True,
            max_token_len_per_agent=200,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('macg_so101/macg_multitask_phased'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('macg_so101'),
                asset_id="macg_multitask_phased",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                use_biso100_format=False,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
                needs_biso100_to_atom=True,
                biso100_joints_per_agent=5,
            ),
            model_transforms=PerAgentPromptModelTransformFactory(),
            base_config=DataConfig(
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=MultiAgentAttnWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=15000,
        keep_period=15000,
        num_workers=1,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_so101_multitask",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('macg_so101/macg_multitask_phased'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path('macg_so101'),
                asset_id="macg_multitask_phased",
            ),
            data_transforms=DroidMultiAgentTransformFactory(
                num_agents=2,
                use_atom_version=True,
                use_biso100_format=False,
                shuffle_prob=0.0,
                image_mask_prob=0.0,
                mask_random_one=True,
                model_type_override=_model.ModelType.PI0,
                use_delta_joint_actions=True,
                joints_per_agent=5,
                wait_augment_prob=0.0,
                wait_instruction="Wait for the other robots.",
                action_dim_per_agent=6,
            ),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                repack_transforms=_transforms.Group(
                    inputs=[
                        # Reshape flat 12-D state/action -> per-agent keys expected by
                        # DroidMultiInputs_atom: joint_position_agent{N}, gripper_position_agent{N}
                        bi_so100_policy.BiSO100ToAtomTransform(joints_per_agent=5)
                    ]
                ),
                action_sequence_keys=("actions",),
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=200000,
            decay_lr=5e-5,
        ),
        num_train_steps=8000,
        batch_size=32,
        log_interval=50,
        save_interval=8000,
        keep_period=8000,
        num_workers=1,
        fsdp_devices=2,
    ),

    # -------------------------------------------------------------------------
    # UR dual-arm EEF-space training
    # -------------------------------------------------------------------------
    TrainConfig(
        name="pi05_ur_two_arms",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,          # 20D UR action padded to 32
            action_horizon=50,
            max_token_len=200,
            discrete_state_input=True,
        ),
        data=SimpleDataConfig(
            repo_id=_paths.dataset_path('ur'),
            assets=AssetsConfig(
                assets_dir=_paths.dataset_path(''),
                asset_id="ur",
            ),
            data_transforms=URTransformFactory(),
            model_transforms=ModelTransformFactory(),
            base_config=DataConfig(
                prompt_from_task=True,
                # Use absolute target actions (14D) to avoid zero-variance rot6d dims in relative actions
                action_sequence_keys=("action.raw_abs",),
                # Use pyav for video decoding (torchcodec requires FFmpeg)
                video_backend="pyav",
                # 60 FPS → frame interval ~16.7ms; set tolerance to half that
                video_tolerance_s=0.02,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_paths.pretrained_params("pi05")),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=20000,
            decay_lr=5e-5,
        ),
        num_train_steps=15000,
        batch_size=32,
        log_interval=50,
        save_interval=10000,
        keep_period=10000,
        num_workers=1,
        fsdp_devices=1,
    ),

    # -------------------------------------------------------------------------
    # UMI bimanual (single Pi0, 20D relative pose10d action)
    # -------------------------------------------------------------------------
    # Dataset path: set to the LeRobot dataset produced by
    #   examples/umi/convert_umi_data_to_lerobot.py
    # Action: 20D = 2 robots × [pos(3) + rot6d(6) + gripper(1)], relative SE3
    # Matches HarMI bimanual DP output exactly; use HarMI's get_real_umi_action
    # with action_pose_repr='relative' to convert back to absolute at deploy time.
    TrainConfig(
        name="pi0_umi_bimanual",
        model=pi0_config.Pi0Config(
            action_dim=32,      # 20D UMI action padded to 32
            action_horizon=16,
            max_token_len=48,
        ),
        data=UmiBimanualDataConfig(
            repo_id=_paths.dataset_path('umi_ur_lerobot/local/umi_bimanual'),
            assets=AssetsConfig(
                # compute_norm_stats writes to <assets_dirs>/<repo_id>; since repo_id is
                # an absolute path, it ends up as <repo_id>/norm_stats.json.
                # Point assets_dir to the dataset root so _load_norm_stats resolves to
                # dataset/umi_bimanual/norm_stats.json (assets_dir / asset_id).
                assets_dir=_paths.dataset_path('umi_ur_lerobot/local'),
                asset_id="umi_bimanual",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _paths.pretrained_params("pi0")
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=30000,
            decay_lr=5e-6,
        ),
        num_train_steps=40000,
        batch_size=32,
        log_interval=100,
        save_interval=10000,
        keep_period=10000,
        num_workers=2,
        fsdp_devices=2,
    ),

]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, get_config(k)) for k in _CONFIGS_DICT})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    dataset_dir = os.environ.get("FUTURE_DATASET_DIR")
    assets_dir = os.environ.get("FUTURE_ASSETS_DIR")
    if dataset_dir or assets_dir:
        data = config.data
        if dataset_dir:
            data = dataclasses.replace(data, repo_id=str(pathlib.Path(dataset_dir).expanduser().resolve()))
            model_transforms = getattr(data, "model_transforms", None)
            if getattr(model_transforms, "skill_labels_path", None) is not None:
                data = dataclasses.replace(
                    data, model_transforms=dataclasses.replace(
                        model_transforms, skill_labels_path=str(pathlib.Path(data.repo_id) / "meta/skill_labels.jsonl")
                    )
                )
        if assets_dir:
            data = dataclasses.replace(
                data, assets=dataclasses.replace(data.assets, assets_dir=str(pathlib.Path(assets_dir).expanduser().resolve()))
            )
        config = dataclasses.replace(config, data=data)
    foldtowel_params = os.environ.get("FUTURE_FOLDTOWEL_PARAMS")
    if foldtowel_params and config_name.startswith("pi05_bi_so100_flodtowel_pretrained"):
        config = dataclasses.replace(
            config, weight_loader=weight_loaders.CheckpointWeightLoader(os.path.expanduser(foldtowel_params))
        )
    return config
