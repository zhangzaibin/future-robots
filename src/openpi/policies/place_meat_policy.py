import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PlaceMeatInputs(transforms.DataTransformFn):
    """
    Input transform for place_meat_order dataset with 2 agents.
    Converts dataset keys to model input format.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 使用 image (从 repack transform 映射的 global_image) 作为 base image
        # 使用 wrist_image (从 repack transform 映射的 image_agent0) 作为 left wrist
        # 使用 image_agent1 作为 right wrist
        base_image = _parse_image(data.get("image", np.zeros((240, 320, 3), dtype=np.uint8)))
        wrist_image_0 = _parse_image(data.get("wrist_image", np.zeros((240, 320, 3), dtype=np.uint8)))
        wrist_image_1 = _parse_image(data.get("image_agent1", np.zeros_like(base_image)))

        # 合并 state: 需要组合两个 agent 的 state
        # 如果已经有合并的 state，直接使用；否则从各个键组合
        if "state" in data:
            combined_state = np.asarray(data["state"]).flatten()
        else:
            # 从各个键组合 state: 7 joints + 1 gripper per agent = 16 total
            state_agent0 = np.asarray(data.get("joint_position_agent0", np.zeros(7))).flatten()
            state_agent1 = np.asarray(data.get("joint_position_agent1", np.zeros(7))).flatten()
            gripper_agent0 = np.asarray(data.get("gripper_position_agent0", np.zeros(1))).flatten()
            gripper_agent1 = np.asarray(data.get("gripper_position_agent1", np.zeros(1))).flatten()
            combined_state = np.concatenate([state_agent0, gripper_agent0, state_agent1, gripper_agent1])

        inputs = {
            "state": combined_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image_0,
                "right_wrist_0_rgb": wrist_image_1,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PlaceMeatOutputs(transforms.DataTransformFn):
    """
    Output transform for place_meat_order dataset.
    Returns 16-dimensional actions (2 agents * 8 dimensions).
    """

    def __call__(self, data: dict) -> dict:
        # 返回 16 维 actions (2 agents * 8 dimensions)
        return {"actions": np.asarray(data["actions"][:, :16])}
