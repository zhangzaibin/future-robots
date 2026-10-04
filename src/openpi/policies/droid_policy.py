import dataclasses
import re

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # print(data.keys())
        data["gripper_position"] = np.array([data["gripper_position"]])
        state = np.concatenate([data["joint_position"], data["gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # base_image = _parse_image(data["observation/exterior_image_1_left"])
        # wrist_image = _parse_image(data["observation/wrist_image_left"])
        base_image = _parse_image(data["global_image"])
        # base_image_1 = _parse_image(data["global_image"])
        wrist_image = _parse_image(data["image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # if "actions" in data:
        #     inputs["actions"] = np.array(data["actions"])
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}


# ============================================================================
# Unified Multi-Agent Configurations (Recommended to use these)
# ============================================================================


@dataclasses.dataclass(frozen=True)
class DroidMultiInputs_atom(transforms.DataTransformFn):
    """
    Unified input transform for multi-agent systems with data augmentation.
    Supports 2-4 agents with full augmentation features from DroidTripleInputs_atom.

    Args:
        action_dim: The action dimension of the model. Will be used to pad state and actions.
        num_agents: Number of robot agents (2-4)
        model_type: Determines which model will be used.
        shuffle_prob: Probability to shuffle agent order (default: 0.0)
        image_mask_prob: Probability to mask the global image to zeros (default: 0.3). Only affects global image, not wrist images.
        image_noise_prob: Probability to add Gaussian noise to images (default: 0.2)
        noise_std: Standard deviation of Gaussian noise (default: 0.02)
        single_arm_id: ID of single arm to train (0 to num_agents-1). If None, use all agents (default: None)
        wait_augment_prob: Probability to randomly make one agent "wait" (set action=state, instruction=wait)
        wait_instruction: Instruction text for waiting agent
        action_dim_per_agent: Action dimension per agent (default: 8 = 7 joints + 1 gripper)
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Number of agents
    num_agents: int = 2

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Shuffle probability: 0.0 = no shuffle, 1.0 = always shuffle
    shuffle_prob: float = 0.0

    # Image augmentation parameters
    image_mask_prob: float = 0.3  # Probability to mask the global image to zeros (wrist images are never masked)
    image_noise_prob: float = 0.0  # Probability to add Gaussian noise to images
    noise_std: float = 0.02  # Standard deviation of Gaussian noise
    mask_all_images: bool = False  # If True, each image is independently masked with image_mask_prob; if False, only global image
    mask_random_one: bool = True  # If True, randomly select exactly one image to mask (global or any wrist)

    # Single arm training mode
    single_arm_id: int | None = None  # If set, only train on this specific arm

    # Wait augmentation parameters
    wait_augment_prob: float = 0.0  # Probability to randomly make one agent "wait"
    wait_instruction: str = "Wait for the other robots."  # Instruction for waiting agent
    action_dim_per_agent: int = 8  # Action dimension per agent (7 joints + 1 gripper)

    def __post_init__(self):
        if self.num_agents < 2 or self.num_agents > 4:
            raise ValueError(f"num_agents must be between 2 and 4, got {self.num_agents}")
        if self.single_arm_id is not None:
            if self.single_arm_id < 0 or self.single_arm_id >= self.num_agents:
                raise ValueError(f"single_arm_id must be between 0 and {self.num_agents-1}, got {self.single_arm_id}")
        if self.wait_augment_prob < 0.0 or self.wait_augment_prob > 1.0:
            raise ValueError(f"wait_augment_prob must be between 0.0 and 1.0, got {self.wait_augment_prob}")

    def _extract_single_arm_prompt(self, task: str, arm_id: int) -> str:
        """
        Extract the prompt for a specific arm from the multi-agent task string.

        Supports two formats:
        1. "|" separated format: "Pick up red bowl | Place blue bowl"
           - Split by "|", each part corresponds to an agent by index
        2. "agentX:" prefix format: "Place the object. agent1: Wait."
           - agent0's instruction is at the beginning without "agent0:" prefix
           - Other agents' instructions have "agentX:" prefix

        Args:
            task: Task string in either format
            arm_id: ID of the arm to extract (0 to num_agents-1)

        Returns:
            The extracted prompt for the specified arm

        Example:
            # "|" format
            _extract_single_arm_prompt("Pick up red bowl | Place blue bowl", 0) -> "Pick up red bowl"
            _extract_single_arm_prompt("Pick up red bowl | Place blue bowl", 1) -> "Place blue bowl"
            # "agentX:" format
            _extract_single_arm_prompt("Place the object. agent1: Wait.", 0) -> "Place the object"
            _extract_single_arm_prompt("Place the object. agent1: Wait.", 1) -> "Wait"
        """
        # Format 1: "|" separated format
        if "|" in task:
            parts = [p.strip() for p in task.split("|")]
            if arm_id < len(parts):
                return parts[arm_id].rstrip(".")
            # Fallback: return last part if arm_id exceeds available parts
            return parts[-1].rstrip(".")

        # Format 2: "agentX:" prefix format
        if arm_id == 0:
            # For agent0, extract the part before the first "agent1:", "agent2:", etc.
            pattern = r"^(.*?)(?:\s+agent[1-9]\d*:|$)"
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                return extracted.rstrip(".")
            return task
        # For agent1, agent2, etc., extract the part after "agentX:"
        pattern = rf"agent{arm_id}:\s*(.*?)(?:\s+agent\d+:|$)"
        match = re.search(pattern, task, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            return extracted.rstrip(".")
        # Fallback: return the whole task
        return task

    def _apply_image_masking(self, images: list[np.ndarray], skip_masking: bool = False) -> list[np.ndarray]:
        """
        Apply various image augmentation techniques including:
        1. Mask images to zeros based on image_mask_prob
        2. Add Gaussian noise to images based on image_noise_prob

        Masking modes (checked in priority order):
        - mask_random_one=True: randomly select exactly one image to mask.
        - mask_all_images=True: each image is independently masked with image_mask_prob.
        - Otherwise (default): only the global image (index 0) can be masked.

        Args:
            images: List of images [global, image_agent0, image_agent1, ...]
            skip_masking: If True, skip masking (e.g., when all agents have the same instruction)

        Returns:
            List of augmented images with potentially masking and/or noise applied
        """
        if skip_masking:
            return [img.copy() for img in images]

        augmented_images = [img.copy() for img in images]

        if self.mask_random_one:
            if np.random.random() < self.image_mask_prob:
                idx = np.random.randint(len(augmented_images))
                augmented_images[idx] = np.zeros_like(augmented_images[idx])
        elif self.mask_all_images:
            for i in range(len(augmented_images)):
                if np.random.random() < self.image_mask_prob:
                    augmented_images[i] = np.zeros_like(augmented_images[i])
        else:
            if np.random.random() < self.image_mask_prob:
                augmented_images[0] = np.zeros_like(augmented_images[0])

        if np.random.random() < self.image_noise_prob:
            for i in range(len(augmented_images)):
                if not np.all(augmented_images[i] == 0):
                    noise = np.random.normal(0, self.noise_std, augmented_images[i].shape).astype(
                        augmented_images[i].dtype
                    )
                    if augmented_images[i].dtype == np.uint8:
                        augmented_images[i] = np.clip(
                            augmented_images[i].astype(np.float32) + noise * 255, 0, 255
                        ).astype(np.uint8)
                    else:
                        augmented_images[i] = np.clip(augmented_images[i] + noise, 0, 1).astype(
                            augmented_images[i].dtype
                        )
        return augmented_images

    def _check_if_agents_have_same_instruction(self, task: str) -> bool:
        """
        Check if all agents have the same instruction in the task.
        If they do, shuffling is not necessary.

        Supports two task formats:
        1. Per-agent prompt format: "pick red cube | hold blue cube" (separated by "|")
        2. Agent prefix format: "agent0: pick red cube; agent1: hold blue cube"

        Args:
            task: Task string to check

        Returns:
            True if all agents have identical instructions, False otherwise
        """
        # Format 1: Per-agent prompt format using "|" separator
        if "|" in task:
            parts = task.split("|")
            parts = [p.strip() for p in parts]
            
            # If fewer parts than agents, can't determine - assume different
            if len(parts) < self.num_agents:
                return False
            
            # Check if all agent instructions are identical
            instructions = parts[:self.num_agents]
            return all(instr == instructions[0] for instr in instructions)
        
        # Format 2: Agent prefix format using "agent0:", "agent1:", etc.
        pattern_parts = [rf"agent{i}:\s*([^;]+);?\s*" for i in range(self.num_agents)]
        pattern = "".join(pattern_parts)
        match = re.search(pattern, task, re.IGNORECASE)

        if match:
            instructions = [match.group(i + 1).strip() for i in range(self.num_agents)]
            # Check if all instructions are identical
            return all(instr == instructions[0] for instr in instructions)
        
        # If format doesn't match, assume they might be different (safe to shuffle)
        return False

    def _shuffle_task_prompt(self, task: str, shuffle_indices: np.ndarray) -> str:
        """
        Shuffle task prompt according to agent reordering.

        Supports two task formats:
        1. Per-agent prompt format: "pick red cube | hold blue cube" (separated by "|")
        2. Agent prefix format: "agent0: pick red cube; agent1: hold blue cube"

        Args:
            task: Original task string
            shuffle_indices: Permutation array, e.g., [1, 0] means agent0 and agent1 swap positions

        Returns:
            Shuffled task string with reordered agent instructions
        """
        # Format 1: Per-agent prompt format using "|" separator
        # Example: "pick red cube | hold blue cube" -> after shuffle [1,0] -> "hold blue cube | pick red cube"
        if "|" in task:
            parts = task.split("|")
            parts = [p.strip() for p in parts]
            
            # Ensure we have enough parts for all agents
            while len(parts) < self.num_agents:
                parts.append(parts[-1] if parts else "")
            
            # Reorder according to shuffle_indices
            # shuffle_indices[new_pos] = old_pos, so parts[shuffle_indices[i]] goes to position i
            shuffled_parts = [parts[shuffle_indices[i]] for i in range(self.num_agents)]
            
            return " | ".join(shuffled_parts)
        
        # Format 2: Agent prefix format using "agent0:", "agent1:", etc.
        # Example: "agent0: pick red cube; agent1: hold blue cube"
        pattern_parts = [rf"(agent{i}:\s*[^;]+;?\s*)" for i in range(self.num_agents)]
        pattern = "".join(pattern_parts)
        match = re.search(pattern, task, re.IGNORECASE)

        if match:
            # Extract agent instructions
            agent_instructions = [match.group(i + 1).strip() for i in range(self.num_agents)]

            # Reorder according to shuffle_indices
            shuffled_instructions = [agent_instructions[i] for i in shuffle_indices]

            # Rebuild with new agent numbers (agent0, agent1, agent2, ...)
            result = []
            for idx, instruction in enumerate(shuffled_instructions):
                # Replace old agentX: with new agent{idx}:
                old_agent_num = shuffle_indices[idx]
                new_instruction = re.sub(f"agent{old_agent_num}:", f"agent{idx}:", instruction, flags=re.IGNORECASE)
                result.append(new_instruction)

            return " ".join(result)
        
        # If format doesn't match, return original task (no shuffle)
        return task

    def _apply_wait_augment(
        self,
        task: str,
        actions: np.ndarray,
        state: np.ndarray,
        wait_agent: int,
    ) -> tuple[str, np.ndarray]:
        """
        Apply wait augmentation: make one agent "wait" by setting its action to current state
        and changing its instruction to the wait instruction.

        This works with delta action transform: when action = state, delta = state - state = 0.

        Args:
            task: Task string in format "instruction0 | instruction1" or similar
            actions: Action array, shape (horizon, num_agents * action_dim_per_agent) or (num_agents * action_dim_per_agent,)
            state: State array, shape (num_agents * action_dim_per_agent,) or padded
            wait_agent: Index of the agent to make wait (0 to num_agents-1)

        Returns:
            Tuple of (modified_task, modified_actions)
        """
        # 1. Modify the task prompt - replace the wait_agent's instruction
        # Handle different prompt formats:
        # Format 1: "instruction0 | instruction1" (per-agent prompt format)
        # Format 2: "agent0: instruction0; agent1: instruction1"
        
        if "|" in task:
            # Per-agent prompt format: "instruction0 | instruction1"
            parts = task.split("|")
            if wait_agent < len(parts):
                parts[wait_agent] = f" {self.wait_instruction} "
            modified_task = "|".join(parts)
        else:
            # Try agent0:, agent1: format
            pattern = rf"(agent{wait_agent}:\s*)([^;]+)(;?)"
            replacement = rf"\g<1>{self.wait_instruction}\g<3>"
            modified_task = re.sub(pattern, replacement, task, flags=re.IGNORECASE)
            if modified_task == task:
                # If no match found, just return original task
                modified_task = task

        # 2. Modify actions - set the wait_agent's actions to its current state
        # This ensures delta = action - state = 0 after DeltaActions transform
        modified_actions = actions.copy()
        
        start_idx = wait_agent * self.action_dim_per_agent
        end_idx = start_idx + self.action_dim_per_agent
        
        # Extract the wait agent's current state
        agent_state = state[start_idx:end_idx]
        
        if modified_actions.ndim == 1:
            # Single action: shape (num_agents * action_dim_per_agent,)
            modified_actions[start_idx:end_idx] = agent_state
        else:
            # Action sequence: shape (horizon, num_agents * action_dim_per_agent)
            # Set all horizon steps to the current state (robot stays still)
            modified_actions[:, start_idx:end_idx] = agent_state

        return modified_task, modified_actions

    def __call__(self, data: dict) -> dict:
        # ===== Single Arm Training Mode =====
        if self.single_arm_id is not None:
            # Extract only the specified arm's data
            arm_id = self.single_arm_id

            # Process state for single arm
            data[f"gripper_position_agent{arm_id}"] = np.array([data[f"gripper_position_agent{arm_id}"]])
            state = np.concatenate([data[f"joint_position_agent{arm_id}"], data[f"gripper_position_agent{arm_id}"]])
            state = transforms.pad_to_dim(state, self.action_dim)

            # Parse images for single arm
            global_image = _parse_image(data["global_image"])
            arm_image = _parse_image(data[f"image_agent{arm_id}"])

            # Apply image masking (same strategy as multi-agent: randomly mask global image)
            [global_image, arm_image] = self._apply_image_masking([global_image, arm_image])

            # Build image inputs for single arm (2 images: global + wrist)
            match self.model_type:
                case _model.ModelType.PI0 | _model.ModelType.PI05:
                    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                    images = (global_image, arm_image, np.zeros_like(global_image))
                    image_masks = (np.True_, np.True_, np.False_)
                case _model.ModelType.PI0_FAST:
                    names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                    # We don't mask out padding images for FAST models.
                    images = (global_image, np.zeros_like(global_image), arm_image)
                    image_masks = (np.True_, np.True_, np.True_)
                case _:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            inputs = {
                "state": state,
                "image": dict(zip(names, images, strict=True)),
                "image_mask": dict(zip(names, image_masks, strict=True)),
            }

            # Determine if this sample should be a "wait" sample
            is_waiting = (self.wait_augment_prob > 0 and np.random.random() < self.wait_augment_prob)

            # Extract single arm actions (8 dims)
            if "actions" in data:
                actions = data["actions"]
                # Ensure actions is numpy array
                if not isinstance(actions, np.ndarray):
                    actions = np.asarray(actions)

                # Extract actions for the specific arm
                # Handle both 1D and 2D actions
                if actions.ndim == 1:
                    # Single action: shape (num_agents * 8,)
                    single_arm_actions = actions[arm_id * 8 : (arm_id + 1) * 8]
                else:
                    # Action sequence: shape (action_horizon, num_agents * 8)
                    single_arm_actions = actions[:, arm_id * 8 : (arm_id + 1) * 8]

                # Wait augmentation: set actions to current state so delta = 0 (stay still)
                if is_waiting:
                    # Use unpadded state (first action_dim_per_agent dims)
                    arm_state = state[:self.action_dim_per_agent]
                    if single_arm_actions.ndim == 1:
                        single_arm_actions = arm_state.copy()
                    else:
                        single_arm_actions = np.tile(arm_state, (single_arm_actions.shape[0], 1))

                # Pad to action_dim
                inputs["actions"] = transforms.pad_to_dim(single_arm_actions, self.action_dim)

            # Extract single arm prompt (or use wait instruction)
            if is_waiting:
                inputs["prompt"] = self.wait_instruction
            elif "task" in data:
                task_str = data["task"]
                if isinstance(task_str, bytes):
                    task_str = task_str.decode("utf-8")
                inputs["prompt"] = self._extract_single_arm_prompt(task_str, arm_id)

            if "task_index" in data:
                inputs["task_index"] = data["task_index"]

            return inputs

        # ===== Multi-Agent Training Mode (original logic) =====
        # Process all agents' data dynamically
        states = []
        for i in range(self.num_agents):
            data[f"gripper_position_agent{i}"] = np.array([data[f"gripper_position_agent{i}"]])
            state = np.concatenate([data[f"joint_position_agent{i}"], data[f"gripper_position_agent{i}"]])
            states.append(state)

        # Parse images
        global_image = _parse_image(data["global_image"])
        agent_images = [_parse_image(data[f"image_agent{i}"]) for i in range(self.num_agents)]

        # Check if agents have the same instruction (if so, skip shuffle and masking)
        should_skip_shuffle = False
        if "task" in data:
            task_str = data["task"].decode("utf-8") if isinstance(data["task"], bytes) else data["task"]
            if self._check_if_agents_have_same_instruction(task_str):
                should_skip_shuffle = True

        # Apply image masking to all images (global + agent images)
        all_images = [global_image] + agent_images
        masked_images = self._apply_image_masking(all_images, skip_masking=should_skip_shuffle)
        global_image = masked_images[0]
        agent_images = masked_images[1:]

        # Decide whether to shuffle
        if not should_skip_shuffle and np.random.random() < self.shuffle_prob:
            # Generate random permutation: [0, 1, ..., num_agents-1] -> e.g., [2, 0, 1]
            shuffle_indices = np.random.permutation(self.num_agents)

            # Shuffle states according to shuffle_indices
            state = np.concatenate([states[i] for i in shuffle_indices])

            # Shuffle wrist images according to shuffle_indices
            agent_images_final = [agent_images[i] for i in shuffle_indices]

            # Shuffle actions if present (num_agents * action_dim_per_agent dims)
            if "actions" in data:
                actions = data["actions"]
                actions_per_agent = self.action_dim_per_agent

                # Handle both 1D and 2D actions (single action or action sequence)
                if actions.ndim == 1:
                    # Single action: shape (num_agents * action_dim_per_agent,)
                    action_chunks = [actions[i * actions_per_agent : (i + 1) * actions_per_agent] for i in range(self.num_agents)]
                    shuffled_actions = np.concatenate([action_chunks[i] for i in shuffle_indices])
                else:
                    # Action sequence: shape (action_horizon, num_agents * action_dim_per_agent)
                    action_chunks = [actions[:, i * actions_per_agent : (i + 1) * actions_per_agent] for i in range(self.num_agents)]
                    shuffled_actions = np.concatenate([action_chunks[i] for i in shuffle_indices], axis=1)

                data["actions"] = shuffled_actions

            # Shuffle task prompt if present
            if "task" in data:
                if isinstance(data["task"], bytes):
                    data["task"] = data["task"].decode("utf-8")
                data["task"] = self._shuffle_task_prompt(data["task"], shuffle_indices)
        else:
            # No shuffle: maintain original order
            state = np.concatenate(states)
            agent_images_final = agent_images

        # ===== Wait Augmentation =====
        # Randomly make one agent "wait" by setting its action = state (delta = 0)
        # and changing its instruction to the wait instruction
        if self.wait_augment_prob > 0.0 and np.random.random() < self.wait_augment_prob:
            if "actions" in data and "task" in data:
                # Randomly select one agent to wait
                wait_agent = np.random.randint(0, self.num_agents)
                
                # Decode task if needed
                task_str = data["task"]
                if isinstance(task_str, bytes):
                    task_str = task_str.decode("utf-8")
                
                # Get current actions
                actions = data["actions"]
                if not isinstance(actions, np.ndarray):
                    actions = np.asarray(actions)
                
                # Apply wait augmentation
                # Note: state here is already concatenated (before padding)
                modified_task, modified_actions = self._apply_wait_augment(
                    task=task_str,
                    actions=actions,
                    state=state,  # state is already concatenated from all agents
                    wait_agent=wait_agent,
                )
                
                data["task"] = modified_task
                data["actions"] = modified_actions

        # Pad state to model action_dim
        state = transforms.pad_to_dim(state, self.action_dim)

        # Build image inputs dynamically based on num_agents
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI0_FAST | _model.ModelType.PI05 | _model.ModelType.PI0_MULTI:
                # Dynamic naming: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, third_wrist_0_rgb, fourth_wrist_0_rgb
                wrist_names = ["left_wrist_0_rgb", "right_wrist_0_rgb", "third_wrist_0_rgb", "fourth_wrist_0_rgb"]
                names = ["base_0_rgb"] + wrist_names[: self.num_agents]
                images = (global_image,) + tuple(agent_images_final)
                image_masks = tuple([np.True_] * (self.num_agents + 1))
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            # Pad combined actions to model action dim
            # Actions should be (num_agents * 8) dims, padded to action_dim
            actions = data["actions"]
            # Ensure actions is numpy array
            if not isinstance(actions, np.ndarray):
                actions = np.asarray(actions)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "task" in data:
            # Decode task if it's still in bytes (might already be decoded in shuffle branch)
            task_str = data["task"]
            if isinstance(task_str, bytes):
                task_str = task_str.decode("utf-8")
            inputs["prompt"] = task_str

        if "task_index" in data:
            inputs["task_index"] = data["task_index"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidMultiOutputs_atom(transforms.DataTransformFn):
    """
    Unified output transform for multi-agent systems.

    Args:
        num_agents: Number of robot agents (2-4)
        single_arm_id: ID of single arm being trained (0 to num_agents-1). If None, use all agents (default: None)
    """

    num_agents: int = 2
    single_arm_id: int | None = None

    def __post_init__(self):
        if self.num_agents < 2 or self.num_agents > 4:
            raise ValueError(f"num_agents must be between 2 and 4, got {self.num_agents}")
        if self.single_arm_id is not None:
            if self.single_arm_id < 0 or self.single_arm_id >= self.num_agents:
                raise ValueError(f"single_arm_id must be between 0 and {self.num_agents-1}, got {self.single_arm_id}")

    def __call__(self, data: dict) -> dict:
        if self.single_arm_id is not None:
            # Return only 8 dimensions for single arm
            return {"actions": np.asarray(data["actions"][:, :8])}
        # Return (num_agents * 8) dimensions
        action_dims = self.num_agents * 8
        return {"actions": np.asarray(data["actions"][:, :action_dims])}


@dataclasses.dataclass(frozen=True)
class DroidMultiInputs(transforms.DataTransformFn):
    """
    Unified input transform for multi-agent systems WITHOUT data augmentation.
    Used for inference with a fixed task instruction.

    Args:
        action_dim: The action dimension of the model. Will be used to pad state and actions.
        num_agents: Number of robot agents (2-4)
        model_type: Determines which model will be used.
        task_instruction: Fixed task instruction for all episodes
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Number of agents
    num_agents: int = 2

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Fixed task instruction
    task_instruction: str = "complete the task"

    def __post_init__(self):
        if self.num_agents < 2 or self.num_agents > 4:
            raise ValueError(f"num_agents must be between 2 and 4, got {self.num_agents}")

    def __call__(self, data: dict) -> dict:
        # Process all agents' data dynamically
        states = []
        for i in range(self.num_agents):
            data[f"gripper_position_agent{i}"] = np.array([data[f"gripper_position_agent{i}"]])
            state = np.concatenate([data[f"joint_position_agent{i}"], data[f"gripper_position_agent{i}"]])
            states.append(state)

        # Combine states from all agents then pad once to model action_dim
        state = np.concatenate(states)
        state = transforms.pad_to_dim(state, self.action_dim)

        # Parse images
        global_image = _parse_image(data["global_image"])
        agent_images = [_parse_image(data[f"image_agent{i}"]) for i in range(self.num_agents)]

        # Build image inputs dynamically based on num_agents
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI0_FAST | _model.ModelType.PI05:
                # Dynamic naming: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, third_wrist_0_rgb, fourth_wrist_0_rgb
                wrist_names = ["left_wrist_0_rgb", "right_wrist_0_rgb", "third_wrist_0_rgb", "fourth_wrist_0_rgb"]
                names = ["base_0_rgb"] + wrist_names[: self.num_agents]
                images = (global_image,) + tuple(agent_images)
                image_masks = tuple([np.True_] * (self.num_agents + 1))
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            # Pad combined actions to model action dim
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        inputs["prompt"] = self.task_instruction

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidMultiOutputs(transforms.DataTransformFn):
    """
    Unified output transform for multi-agent systems (non-atom version).

    Args:
        num_agents: Number of robot agents (2-4)
    """

    num_agents: int = 2

    def __post_init__(self):
        if self.num_agents < 2 or self.num_agents > 4:
            raise ValueError(f"num_agents must be between 2 and 4, got {self.num_agents}")

    def __call__(self, data: dict) -> dict:
        # Return (num_agents * 8) dimensions
        action_dims = self.num_agents * 8
        return {"actions": np.asarray(data["actions"][:, :action_dims])}


@dataclasses.dataclass(frozen=True)
class DroidDualInputs_atom_robotwin(transforms.DataTransformFn):
    """
    Input transform for dual-agent RobotWin dataset using Aloha-style key names.
    Expects data with keys like:
    - observation.images.cam_high (global view)
    - observation.images.cam_left_wrist (agent0)
    - observation.images.cam_right_wrist (agent1)
    - observation.state (combined state for both agents)
    - action (combined actions)
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Shuffle probability
    shuffle_prob: float = 0.0

    # Image augmentation parameters
    image_mask_prob: float = 0.3  # Probability to mask the global image to zeros (wrist images are never masked)
    image_noise_prob: float = 0.0  # Probability to add Gaussian noise to images
    mask_multiple_images: bool = True  # (deprecated, no longer used) Whether to allow masking multiple images at once
    max_masked_images: int = 3  # (deprecated, no longer used) Maximum number of images to mask (1-3)
    noise_std: float = 0.02  # Standard deviation of Gaussian noise
    mask_all_images: bool = False  # If True, each image is independently masked with image_mask_prob; if False, only global image
    mask_random_one: bool = False  # If True, randomly select exactly one image to mask (global or any wrist)

    # Fixed task instruction. If provided, will use this instead of extracting from dataset.
    task_instruction: str | None = None

    # Wait augmentation parameters
    wait_augment_prob: float = 0.0  # Probability to randomly make one agent "wait"
    wait_instruction: str = "Wait for the other robots."  # Instruction for waiting agent
    action_dim_per_agent: int = 7  # Action dimension per agent (6 joints + 1 gripper for RobotWin)

    def _apply_image_masking(self, images: list[np.ndarray], skip_masking: bool = False) -> list[np.ndarray]:
        """
        Apply image augmentation techniques including masking and noise.

        Masking modes (checked in priority order):
        - mask_random_one=True: randomly select exactly one image to mask.
        - mask_all_images=True: each image is independently masked with image_mask_prob.
        - Otherwise (default): only the global image (index 0) can be masked.
        """
        if skip_masking:
            return [img.copy() for img in images]

        augmented_images = [img.copy() for img in images]

        if self.mask_random_one:
            if np.random.random() < self.image_mask_prob:
                idx = np.random.randint(len(augmented_images))
                augmented_images[idx] = np.zeros_like(augmented_images[idx])
        elif self.mask_all_images:
            for i in range(len(augmented_images)):
                if np.random.random() < self.image_mask_prob:
                    augmented_images[i] = np.zeros_like(augmented_images[i])
        else:
            if np.random.random() < self.image_mask_prob:
                augmented_images[0] = np.zeros_like(augmented_images[0])

        if np.random.random() < self.image_noise_prob:
            for i in range(len(augmented_images)):
                if not np.all(augmented_images[i] == 0):
                    noise = np.random.normal(0, self.noise_std, augmented_images[i].shape).astype(
                        augmented_images[i].dtype
                    )
                    if augmented_images[i].dtype == np.uint8:
                        augmented_images[i] = np.clip(
                            augmented_images[i].astype(np.float32) + noise * 255, 0, 255
                        ).astype(np.uint8)
                    else:
                        augmented_images[i] = np.clip(augmented_images[i] + noise, 0, 1).astype(
                            augmented_images[i].dtype
                        )

        return augmented_images

    def _apply_wait_augment(
        self,
        prompt: str,
        actions: np.ndarray,
        state: np.ndarray,
        wait_agent: int,
    ) -> tuple[str, np.ndarray]:
        """
        Apply wait augmentation: make one agent "wait" by setting its action to current state
        and changing its instruction to the wait instruction.

        This works with delta action transform: when action = state, delta = state - state = 0.

        Args:
            prompt: Prompt string in format "instruction0 | instruction1"
            actions: Action array, shape (horizon, 14) or (14,) for 2 agents × 7 dims
            state: State array, shape (14,) = [left_7, right_7]
            wait_agent: Index of the agent to make wait (0 or 1)

        Returns:
            Tuple of (modified_prompt, modified_actions)
        """
        # 1. Modify the prompt - replace the wait_agent's instruction
        if "|" in prompt:
            # Per-agent prompt format: "instruction0 | instruction1"
            parts = prompt.split("|")
            if wait_agent < len(parts):
                parts[wait_agent] = f" {self.wait_instruction} "
            modified_prompt = "|".join(parts)
        else:
            # Fallback: just return original prompt
            modified_prompt = prompt

        # 2. Modify actions - set the wait_agent's actions to its current state
        # This ensures delta = action - state = 0 after DeltaActions transform
        modified_actions = actions.copy()

        start_idx = wait_agent * self.action_dim_per_agent
        end_idx = start_idx + self.action_dim_per_agent

        # Extract the wait agent's current state
        agent_state = state[start_idx:end_idx]

        if modified_actions.ndim == 1:
            # Single action: shape (num_agents * action_dim_per_agent,)
            modified_actions[start_idx:end_idx] = agent_state
        else:
            # Action sequence: shape (horizon, num_agents * action_dim_per_agent)
            # Set all horizon steps to the current state (robot stays still)
            modified_actions[:, start_idx:end_idx] = agent_state

        return modified_prompt, modified_actions

    def _check_if_agents_have_same_instruction(self, prompt: str) -> bool:
        """Check if both agents have the same instruction."""
        # Format 1: Per-agent prompt format using "|" separator
        if "|" in prompt:
            parts = [p.strip() for p in prompt.split("|")]
            if len(parts) >= 2:
                return parts[0] == parts[1]
            return False

        # Format 2: Agent prefix format using "agent0:", "agent1:"
        pattern = r"agent0:\s*([^;]+);?\s*agent1:\s*([^;]+);?\s*"
        match = re.search(pattern, prompt, re.IGNORECASE)

        if match:
            instruction0 = match.group(1).strip()
            instruction1 = match.group(2).strip()
            return instruction0 == instruction1
        return False

    def _shuffle_task_prompt(self, prompt: str, shuffle_indices: np.ndarray) -> str:
        """Shuffle task prompt according to agent reordering."""
        pattern = r"(agent0:\s*[^;]+;?\s*)(agent1:\s*[^;]+;?\s*)"
        match = re.search(pattern, prompt, re.IGNORECASE)

        if match:
            agent_instructions = [match.group(1).strip(), match.group(2).strip()]
            shuffled_instructions = [agent_instructions[i] for i in shuffle_indices]

            result = []
            for idx, instruction in enumerate(shuffled_instructions):
                old_agent_num = shuffle_indices[idx]
                new_instruction = re.sub(f"agent{old_agent_num}:", f"agent{idx}:", instruction, flags=re.IGNORECASE)
                result.append(new_instruction)

            return " ".join(result)
        return prompt

    def __call__(self, data: dict) -> dict:
        # Extract state from observation.state (should already be combined for both agents)
        state = np.array(data["observation"]["state"])

        # Parse images from Aloha-style nested structure
        global_image = _parse_image(data["observation"]["images"]["cam_high"])
        image_agent0 = _parse_image(data["observation"]["images"]["cam_left_wrist"])
        image_agent1 = _parse_image(data["observation"]["images"]["cam_right_wrist"])

        # Check if agents have the same instruction
        should_skip_shuffle = False
        if "prompt" in data:
            prompt_str = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]
            if self._check_if_agents_have_same_instruction(prompt_str):
                should_skip_shuffle = True

        # Apply image masking
        agent_images = [global_image, image_agent0, image_agent1]
        masked_agent_images = self._apply_image_masking(agent_images, skip_masking=should_skip_shuffle)
        global_image, image_agent0, image_agent1 = masked_agent_images

        # Decide whether to shuffle
        if not should_skip_shuffle and np.random.random() < self.shuffle_prob:
            # Generate random permutation for 2 agents: [0, 1] -> e.g., [1, 0]
            shuffle_indices = np.random.permutation(2)

            # Shuffle state (assuming state is [agent0_state, agent1_state])
            # Split state into two halves
            state_dim_per_agent = len(state) // 2
            state_agent0 = state[:state_dim_per_agent]
            state_agent1 = state[state_dim_per_agent:]
            state_chunks = [state_agent0, state_agent1]
            state = np.concatenate([state_chunks[i] for i in shuffle_indices])

            # Shuffle images according to shuffle_indices
            image_chunks = [image_agent0, image_agent1]
            image_agent0_final = image_chunks[shuffle_indices[0]]
            image_agent1_final = image_chunks[shuffle_indices[1]]

            # Shuffle actions if present
            if "action" in data:
                actions = data["action"]
                actions_per_agent = 7  # 7 dims per agent (6 joints + 1 gripper)

                if actions.ndim == 1:
                    # Single action: 14-dim = 2 agents × 7 dims
                    action_chunks = [actions[i * actions_per_agent : (i + 1) * actions_per_agent] for i in range(2)]
                    shuffled_actions = np.concatenate([action_chunks[i] for i in shuffle_indices])
                else:
                    # Action sequence: (seq_len, 14)
                    action_chunks = [actions[:, i * actions_per_agent : (i + 1) * actions_per_agent] for i in range(2)]
                    shuffled_actions = np.concatenate([action_chunks[i] for i in shuffle_indices], axis=1)

                data["action"] = shuffled_actions

            # Shuffle task prompt if present
            if "prompt" in data:
                prompt_str = data["prompt"].decode("utf-8") if isinstance(data["prompt"], bytes) else data["prompt"]
                data["prompt"] = self._shuffle_task_prompt(prompt_str, shuffle_indices)
        else:
            # No shuffle
            image_agent0_final, image_agent1_final = image_agent0, image_agent1

        # ===== Wait Augmentation =====
        # Randomly make one agent "wait" by setting its action = state (delta = 0)
        # and changing its instruction to the wait instruction
        if self.wait_augment_prob > 0.0 and np.random.random() < self.wait_augment_prob:
            if "action" in data and "prompt" in data:
                # Randomly select one agent to wait
                wait_agent = np.random.randint(0, 2)

                # Decode prompt if needed
                prompt_str = data["prompt"]
                if isinstance(prompt_str, bytes):
                    prompt_str = prompt_str.decode("utf-8")

                # Get current actions
                actions = data["action"]
                if not isinstance(actions, np.ndarray):
                    actions = np.asarray(actions)

                # Apply wait augmentation
                # Note: state here is already combined [left_7, right_7] (before padding)
                modified_prompt, modified_actions = self._apply_wait_augment(
                    prompt=prompt_str,
                    actions=actions,
                    state=state,  # state is [left_7, right_7] = 14 dims
                    wait_agent=wait_agent,
                )

                data["prompt"] = modified_prompt
                data["action"] = modified_actions

        # Pad state to model action_dim
        state = transforms.pad_to_dim(state, self.action_dim)

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (global_image, image_agent0_final, image_agent1_final)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (global_image, image_agent0_final, image_agent1_final)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (global_image, image_agent0_final, image_agent1_final)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "action" in data:
            # Pad combined actions to model action dim
            inputs["actions"] = transforms.pad_to_dim(data["action"], self.action_dim)

        # Use fixed task_instruction if provided, otherwise extract from dataset
        if self.task_instruction is not None:
            inputs["prompt"] = self.task_instruction
        elif "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidDualOutputs_atom_robotwin(transforms.DataTransformFn):
    """
    Output transform for dual-agent RobotWin dataset.
    Returns 14 dimensions total (7 dims per agent: 6 joints + 1 gripper).
    """

    def __call__(self, data: dict) -> dict:
        # Return all 14 dimensions for dual agents (7 dims per agent)
        return {"actions": np.asarray(data["actions"][:, :14])}
