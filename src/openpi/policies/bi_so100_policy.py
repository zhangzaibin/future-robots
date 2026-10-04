"""Bi-SO100 dual-arm robot policy transforms with 12 DoF (6 per arm)."""

import dataclasses
import re

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bi_so100_example() -> dict:
    """Creates a random input example for the Bi-SO100 dual-arm policy."""
    return {
        "observation/state": np.random.rand(12),  # 6 per arm
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # Global camera
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # Left wrist
        "observation/image_agent1": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),  # Right wrist
        "prompt": "stack the bowls",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BiSO100Inputs(transforms.DataTransformFn):
    """
    Input transform for Bi-SO100 dual-arm robot dataset.
    
    Bi-SO100 has 12 DoF total:
        - Left arm: 5 joint positions + 1 gripper = 6 dims
        - Right arm: 5 joint positions + 1 gripper = 6 dims
    
    Dataset keys after conversion:
        - image: global camera (orbbec_rgb)
        - wrist_image: left wrist camera
        - image_agent1: right wrist camera
        - state: 12 dims (left 6 + right 6)
        - actions: 12 dims
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images - 3 cameras for dual-arm setup
        # Global/top camera -> base_0_rgb
        base_image = _parse_image(data.get("observation/image", np.zeros((480, 640, 3), dtype=np.uint8)))
        # Left wrist camera -> left_wrist_0_rgb
        left_wrist_image = _parse_image(data.get("observation/wrist_image", np.zeros((480, 640, 3), dtype=np.uint8)))
        # Right wrist camera -> right_wrist_0_rgb
        right_wrist_image = _parse_image(data.get("observation/image_agent1", np.zeros((480, 640, 3), dtype=np.uint8)))

        # Get state (12 dims for Bi-SO100: 6 left + 6 right)
        state = np.asarray(data.get("observation/state", np.zeros(12))).flatten()

        # Create inputs dict
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,  # All 3 cameras are valid for dual-arm
            },
        }

        # Actions are only available during training
        if "actions" in data:
            # Pad 12-dim actions to 32-dim to match model output dimension
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, 32, axis=-1)

        # Pass the prompt
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class BiSO100Outputs(transforms.DataTransformFn):
    """
    Output transform for Bi-SO100 dual-arm robot.
    Returns actions with 12 dimensions (6 per arm).
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 12 actions (Bi-SO100 has 12 DoF)
        return {"actions": np.asarray(data["actions"][:, :12])}


@dataclasses.dataclass(frozen=True)
class BiSO100MultiAgentInputs(transforms.DataTransformFn):
    """
    Multi-agent input transform for Bi-SO100 dual-arm robot with shuffle and augmentation.

    Reads bi_so100 flat-key format directly:
        - observation/image (global camera)
        - observation/wrist_image (left wrist camera, agent0)
        - observation/image_agent1 (right wrist camera, agent1)
        - observation/state (12 dims: left 6 + right 6)
        - actions (12 dims per timestep)

    Supports shuffle, image masking, noise, and wait augmentation.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    shuffle_prob: float = 0.0
    image_mask_prob: float = 0.0
    image_noise_prob: float = 0.0
    noise_std: float = 0.02
    mask_all_images: bool = False  # If True, each image is independently masked with image_mask_prob; if False, only global image
    mask_random_one: bool = False  # If True, randomly select exactly one image to mask (global or any wrist)

    action_dim_per_agent: int = 6  # 5 joints + 1 gripper
    wait_augment_prob: float = 0.0
    wait_instruction: str = "Wait for the other robots."

    def _apply_image_masking(self, images: list[np.ndarray], skip_masking: bool = False) -> list[np.ndarray]:
        if skip_masking:
            return [img.copy() for img in images]

        augmented = [img.copy() for img in images]

        if self.mask_random_one:
            if np.random.random() < self.image_mask_prob:
                idx = np.random.randint(len(augmented))
                augmented[idx] = np.zeros_like(augmented[idx])
        elif self.mask_all_images:
            for i in range(len(augmented)):
                if np.random.random() < self.image_mask_prob:
                    augmented[i] = np.zeros_like(augmented[i])
        else:
            if np.random.random() < self.image_mask_prob:
                augmented[0] = np.zeros_like(augmented[0])

        if np.random.random() < self.image_noise_prob:
            for i in range(len(augmented)):
                if not np.all(augmented[i] == 0):
                    noise = np.random.normal(0, self.noise_std, augmented[i].shape)
                    if augmented[i].dtype == np.uint8:
                        augmented[i] = np.clip(
                            augmented[i].astype(np.float32) + noise * 255, 0, 255
                        ).astype(np.uint8)
                    else:
                        augmented[i] = np.clip(
                            augmented[i] + noise, 0, 1
                        ).astype(augmented[i].dtype)

        return augmented

    def _check_if_agents_have_same_instruction(self, prompt: str) -> bool:
        if "|" in prompt:
            parts = [p.strip() for p in prompt.split("|")]
            return len(parts) >= 2 and parts[0] == parts[1]
        return False

    def _shuffle_prompt(self, prompt: str, shuffle_indices: np.ndarray) -> str:
        if "|" not in prompt:
            return prompt
        parts = [p.strip() for p in prompt.split("|")]
        while len(parts) < 2:
            parts.append(parts[-1] if parts else "")
        shuffled = [parts[shuffle_indices[i]] for i in range(2)]
        return " | ".join(shuffled)

    def _ensure_per_agent_prompt(self, prompt: str) -> str:
        """If prompt has no '|' separator, duplicate it for both agents."""
        if "|" not in prompt:
            return f"{prompt} | {prompt}"
        return prompt

    def _apply_wait_augment(
        self,
        prompt: str,
        actions: np.ndarray,
        state: np.ndarray,
        wait_agent: int,
    ) -> tuple[str, np.ndarray]:
        if "|" in prompt:
            parts = prompt.split("|")
            if wait_agent < len(parts):
                parts[wait_agent] = f" {self.wait_instruction} "
            modified_prompt = "|".join(parts)
        else:
            modified_prompt = prompt

        modified_actions = actions.copy()
        start_idx = wait_agent * self.action_dim_per_agent
        end_idx = start_idx + self.action_dim_per_agent
        agent_state = state[start_idx:end_idx]

        if modified_actions.ndim == 1:
            modified_actions[start_idx:end_idx] = agent_state
        else:
            modified_actions[:, start_idx:end_idx] = agent_state

        return modified_prompt, modified_actions

    def __call__(self, data: dict) -> dict:
        global_image = _parse_image(
            data.get("observation/image", np.zeros((480, 640, 3), dtype=np.uint8))
        )
        image_agent0 = _parse_image(
            data.get("observation/wrist_image", np.zeros((480, 640, 3), dtype=np.uint8))
        )
        image_agent1 = _parse_image(
            data.get("observation/image_agent1", np.zeros((480, 640, 3), dtype=np.uint8))
        )

        state = np.asarray(
            data.get("observation/state", np.zeros(self.action_dim_per_agent * 2))
        ).flatten()

        # Ensure per-agent prompt format
        prompt = ""
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            prompt = self._ensure_per_agent_prompt(prompt)

        should_skip_shuffle = self._check_if_agents_have_same_instruction(prompt) if prompt else False

        # Image masking
        all_images = [global_image, image_agent0, image_agent1]
        masked_images = self._apply_image_masking(all_images, skip_masking=should_skip_shuffle)
        global_image = masked_images[0]
        image_agent0_final, image_agent1_final = masked_images[1], masked_images[2]

        # Shuffle
        if not should_skip_shuffle and np.random.random() < self.shuffle_prob:
            shuffle_indices = np.random.permutation(2)

            state_per_agent = len(state) // 2
            state_chunks = [state[:state_per_agent], state[state_per_agent:]]
            state = np.concatenate([state_chunks[i] for i in shuffle_indices])

            wrist_chunks = [masked_images[1], masked_images[2]]
            image_agent0_final = wrist_chunks[shuffle_indices[0]]
            image_agent1_final = wrist_chunks[shuffle_indices[1]]

            if "actions" in data:
                actions = np.asarray(data["actions"])
                dim = self.action_dim_per_agent
                if actions.ndim == 1:
                    chunks = [actions[i * dim:(i + 1) * dim] for i in range(2)]
                    data["actions"] = np.concatenate([chunks[i] for i in shuffle_indices])
                else:
                    chunks = [actions[:, i * dim:(i + 1) * dim] for i in range(2)]
                    data["actions"] = np.concatenate([chunks[i] for i in shuffle_indices], axis=1)

            if prompt:
                prompt = self._shuffle_prompt(prompt, shuffle_indices)

        # Wait augmentation
        if self.wait_augment_prob > 0.0 and np.random.random() < self.wait_augment_prob:
            if "actions" in data and prompt:
                wait_agent = np.random.randint(0, 2)
                actions = np.asarray(data["actions"])
                prompt, modified_actions = self._apply_wait_augment(
                    prompt=prompt, actions=actions, state=state, wait_agent=wait_agent,
                )
                data["actions"] = modified_actions

        # Pad state
        state = transforms.pad_to_dim(state, self.action_dim)

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (global_image, image_agent0_final, image_agent1_final)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim, axis=-1)

        if prompt:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class BiSO100ToAtomTransform(transforms.DataTransformFn):
    """
    Converts flat BiSO100 dataset keys into atom-format keys expected by DroidMultiInputs_atom.

    Input keys (from stackbowls_v2 parquet):
        - image          : global camera
        - wrist_image    : left wrist camera  (agent 0)
        - image_agent1   : right wrist camera (agent 1)
        - state          : flat 12-dim [left_j0..j4, left_gripper, right_j0..j4, right_gripper]
        - actions        : flat 12-dim
        - task / prompt  : instruction string (passed through)
        - phase          : phase label (passed through if present)

    Output keys (atom format, consumed by DroidMultiInputs_atom):
        - global_image
        - image_agent0
        - image_agent1
        - joint_position_agent0  : state[0:5]
        - gripper_position_agent0: state[5:6]
        - joint_position_agent1  : state[6:11]
        - gripper_position_agent1: state[11:12]
        - actions, task/prompt, phase (passed through unchanged)

    Args:
        joints_per_agent: number of joint dims per arm (default 5 for SO100)
    """

    joints_per_agent: int = 5  # SO100 has 5 joints; gripper is +1 → total 6 per arm

    @staticmethod
    def _get(data: dict, *keys: str) -> object:
        """Return the value for the first key found in data."""
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"None of {keys} found in data. Available keys: {list(data.keys())}")

    def __call__(self, data: dict) -> dict:
        out: dict = {}

        # ── Images ──────────────────────────────────────────────────────────
        # Accept plain keys (training dataset) and observation/-prefixed keys (inference env).
        # Right wrist may arrive as "image_agent1", "observation/image_agent1", or
        # "observation/right_wrist_image" depending on the robot client.
        out["global_image"] = self._get(data, "image", "observation/image")
        out["image_agent0"] = self._get(data, "wrist_image", "observation/wrist_image")
        out["image_agent1"] = self._get(data, "image_agent1", "observation/image_agent1", "observation/right_wrist_image")

        # ── State split ─────────────────────────────────────────────────────
        # joint_position: 1D array (j,); gripper_position: scalar float.
        # droid_policy.py wraps gripper with np.array([scalar]) → shape (1,) for concatenation.
        # If we pass shape (1,) array instead, np.array([array]) → (1,1) which breaks concat.
        state = np.asarray(self._get(data, "state", "observation/state")).flatten()
        j = self.joints_per_agent
        out["joint_position_agent0"]   = state[0:j].astype(np.float32)
        out["gripper_position_agent0"] = float(state[j])
        out["joint_position_agent1"]   = state[j + 1 : 2 * j + 1].astype(np.float32)
        out["gripper_position_agent1"] = float(state[2 * j + 1])

        # ── Pass-through fields ──────────────────────────────────────────────
        if "actions" in data:
            out["actions"] = data["actions"]

        # DroidMultiInputs_atom reads "task", but LeRobot datasets may store "prompt"
        if "task" in data:
            out["task"] = data["task"]
        elif "prompt" in data:
            out["task"] = data["prompt"]

        if "phase" in data:
            out["phase"] = data["phase"]

        return out
