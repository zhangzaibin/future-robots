"""UR dual-arm robot policy transforms.

The UR dataset uses EEF-space state/actions with two wrist cameras and no global camera.

Dataset format (starvla_oxe / LeRobot v2.1):
  - observation.images.camera0: (H, W, 3) uint8 — right-arm wrist camera
  - observation.images.camera1: (H, W, 3) uint8 — left-arm wrist camera
  - observation.state:    14D float32 — (pos.xyz + axis_angle.xyz + gripper) × 2 robots
  - action.raw_abs:       14D float32 — (target_pos.xyz + target_rot.axis_angle.xyz + target_gripper) × 2 robots
  - action (unused):      20D float32 — relative rot6d actions (has zero-variance dims, not used)
  - language_instruction: str
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """Ensure image is uint8 (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_ur_example() -> dict:
    """Creates a random observation dict matching UR dataset keys (for testing)."""
    return {
        "observation.images.camera0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(14).astype(np.float32),
        "actions": np.random.rand(50, 20).astype(np.float32),
        "prompt": "Pick up the bottle with the right arm, pass it to the left arm, and place it down.",
    }


@dataclasses.dataclass(frozen=True)
class URInputs(transforms.DataTransformFn):
    """Converts UR LeRobot dataset keys to the model-expected format for pi05 training.

    Produces:
      image["base_0_rgb"]        — zeros (no global camera), mask=False
      image["left_wrist_0_rgb"]  — camera0 (right-arm wrist), mask=True
      image["right_wrist_0_rgb"] — camera1 (left-arm wrist),  mask=True
      state                      — 14D EEF state
      actions                    — 20D relative actions
      prompt                     — language instruction
    """

    def __call__(self, data: dict) -> dict:
        cam0 = _parse_image(data["observation.images.camera0"])
        cam1 = _parse_image(data["observation.images.camera1"])
        black = np.zeros_like(cam0)

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (black, cam0, cam1)
        # Global camera is masked out (no hardware global cam on UR setup)
        image_masks = (np.False_, np.True_, np.True_)

        out: dict = {
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "state": np.asarray(data["observation.state"], dtype=np.float32),
        }

        # Priority: "action.raw_abs" (14D absolute target) > "actions" > "action" (20D relative)
        raw_actions = data.get("action.raw_abs", data.get("actions", data.get("action")))
        if raw_actions is not None:
            out["actions"] = np.asarray(raw_actions, dtype=np.float32)

        # Accept either "prompt" or "language_instruction" from the dataset
        prompt = data.get("prompt", data.get("language_instruction", ""))
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        out["prompt"] = prompt

        return out


@dataclasses.dataclass(frozen=True)
class UROutputs(transforms.DataTransformFn):
    """Extracts the UR 14D absolute-target actions from model output (trims padding).

    14D = (target_pos.xyz + target_rot.axis_angle.xyz + target_gripper_width) × 2 robots
    """

    action_dim: int = 14  # 2 robots × 7D each

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
