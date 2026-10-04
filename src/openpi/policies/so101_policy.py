"""SO101 robot policy transforms for single-arm robot with 6 DoF."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy."""
    return {
        "observation/state": np.random.rand(6),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the cup and place it in the basket",
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
class SO101Inputs(transforms.DataTransformFn):
    """
    Input transform for SO101 single-arm robot dataset.
    
    SO101 has 6 DoF: 5 joint positions + 1 gripper position.
    Dataset keys:
        - observation.state: 6 dims
        - observation.images.front: front camera
        - observation.images.side: side camera  
        - action: 6 dims
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images - front camera as base image, side camera as wrist image
        base_image = _parse_image(data.get("observation/image", np.zeros((480, 640, 3), dtype=np.uint8)))
        wrist_image = _parse_image(data.get("observation/wrist_image", np.zeros((480, 640, 3), dtype=np.uint8)))

        # Get state (6 dims for SO101)
        state = np.asarray(data.get("observation/state", np.zeros(6))).flatten()

        # Create inputs dict
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training
        if "actions" in data:
            # Pad 6-dim actions to 32-dim to match model output dimension
            # The model predicts 32-dim but we only supervise the first 6-dim
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
class SO101Outputs(transforms.DataTransformFn):
    """
    Output transform for SO101 single-arm robot.
    Returns actions with 6 dimensions (5 joints + 1 gripper).
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 6 actions (SO101 has 6 DoF)
        return {"actions": np.asarray(data["actions"][:, :6])}



