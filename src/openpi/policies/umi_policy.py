"""
UMI bimanual policy transforms for openpi / pi0.

The action representation matches HarMI's bimanual Diffusion Policy exactly:
  - 20D per step = 2 robots × [pos(3) + rot6d(6) + gripper(1)]
  - Poses are expressed **relative** to the current (last-obs-step) TCP pose
    via inv(T_current) @ T_future  (same as HarMI 'relative' pose_repr).

Stored in LeRobot (absolute axis-angle, 14D per frame):
  state   (26,) = [r0_pos(3), r0_rot_aa(3), r0_grip(1),
                   r1_pos(3), r1_rot_aa(3), r1_grip(1),
                   r0_demo_start(6), r1_demo_start(6)]
  actions (14,) = [r0_pos(3), r0_rot_aa(3), r0_grip(1),
                   r1_pos(3), r1_rot_aa(3), r1_grip(1)]

UmiInputs converts:
  state   → 32D model state (padded)
  actions → (H,20) relative pose10d (then padded to model action_dim=32)

Pose math is ported from:
  /18141169908/ur/HarMI/umi/common/pose_util.py
  /18141169908/ur/HarMI/diffusion_policy/common/pose_repr_util.py
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
import scipy.spatial.transform as st

from openpi import transforms
from openpi.models import model as _model


# ---------------------------------------------------------------------------
# Pose math (ported from HarMI)
# ---------------------------------------------------------------------------

def _normalize_vecs(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Extract 6D rotation representation from rotation matrix (...,3,3)."""
    batch_dim = mat.shape[:-2]
    return mat[..., :2, :].reshape(batch_dim + (6,)).copy()


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """Reconstruct rotation matrix from 6D representation (...,6)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize_vecs(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_vecs(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2)  # (...,3,3)


def _pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert 6D axis-angle pose (...,6) to 4×4 SE(3) matrix."""
    pos = pose[..., :3]
    aa = pose[..., 3:]
    shape = pose.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    rot = st.Rotation.from_rotvec(aa.reshape(-1, 3))
    mat[..., :3, :3] = rot.as_matrix().reshape(shape + (3, 3))
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat


def _mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """Extract 10D pose (pos(3) + rot6d(6) + gripper placeholder) from 4×4 mat.
    Returns (..., 9) — gripper is appended by the caller.
    """
    pos = mat[..., :3, 3]
    rot6d = _mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)  # (...,9)


def _convert_pose_mat_relative(
    pose_mat: np.ndarray, base_pose_mat: np.ndarray
) -> np.ndarray:
    """Express pose_mat relative to base_pose_mat: inv(base) @ pose."""
    # base_pose_mat is a single 4×4 matrix (the current TCP pose)
    return np.linalg.inv(base_pose_mat) @ pose_mat


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ---------------------------------------------------------------------------
# State layout helpers
# ---------------------------------------------------------------------------
# state (26,) layout indices
_R0_POS = slice(0, 3)
_R0_ROT = slice(3, 6)
_R0_GRIP = slice(6, 7)
_R1_POS = slice(7, 10)
_R1_ROT = slice(10, 13)
_R1_GRIP = slice(13, 14)
_R0_START = slice(14, 20)
_R1_START = slice(20, 26)

# actions (14,) layout indices
_A_R0_POS = slice(0, 3)
_A_R0_ROT = slice(3, 6)
_A_R0_GRIP = slice(6, 7)
_A_R1_POS = slice(7, 10)
_A_R1_ROT = slice(10, 13)
_A_R1_GRIP = slice(13, 14)


def _build_model_state(state: np.ndarray) -> np.ndarray:
    """Build a 32D model state from the raw 26D stored state.

    Mirrors HarMI bimanual obs construction (umi_dataset.py lines 273-325):
      [r0_grip(1), r1_grip(1),
       r0_rot6d_wrt_start(6), r1_rot6d_wrt_start(6),
       r0_pose10d_relative_to_r1(9), r1_pose10d_relative_to_r0(9)]
    = 1+1+6+6+9+9 = 32D  → padded to model action_dim (32).

    For inference the 'state' array is a single frame (26,).
    """
    # Parse current poses
    r0_pose = np.concatenate([state[_R0_POS], state[_R0_ROT]])  # (6,)
    r1_pose = np.concatenate([state[_R1_POS], state[_R1_ROT]])  # (6,)
    r0_mat = _pose_to_mat(r0_pose)  # (4,4)
    r1_mat = _pose_to_mat(r1_pose)

    # Gripper
    r0_grip = state[_R0_GRIP]  # (1,)
    r1_grip = state[_R1_GRIP]  # (1,)

    # Rotation w.r.t. episode start
    r0_start_mat = _pose_to_mat(state[_R0_START])
    r1_start_mat = _pose_to_mat(state[_R1_START])
    r0_wrt_start = _convert_pose_mat_relative(r0_mat, r0_start_mat)
    r1_wrt_start = _convert_pose_mat_relative(r1_mat, r1_start_mat)
    r0_rot6d_wrt_start = _mat_to_rot6d(r0_wrt_start[:3, :3])  # (6,)
    r1_rot6d_wrt_start = _mat_to_rot6d(r1_wrt_start[:3, :3])  # (6,)

    # Relative poses between robots (wrt each other's current pose)
    r0_wrt_r1 = _convert_pose_mat_relative(r0_mat, r1_mat)  # inv(r1) @ r0
    r1_wrt_r0 = _convert_pose_mat_relative(r1_mat, r0_mat)  # inv(r0) @ r1
    r0_pose10d_wrt_r1 = _mat_to_pose10d(r0_wrt_r1)  # (9,)
    r1_pose10d_wrt_r0 = _mat_to_pose10d(r1_wrt_r0)  # (9,)

    model_state = np.concatenate(
        [r0_grip, r1_grip,
         r0_rot6d_wrt_start, r1_rot6d_wrt_start,
         r0_pose10d_wrt_r1, r1_pose10d_wrt_r0],
        axis=-1,
    ).astype(np.float32)  # (32,)
    return model_state


def _abs_actions_to_relative_pose10d(
    state: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Convert stacked absolute action chunk (H,14) to relative pose10d (H,20).

    For each robot:
      1. Build the current base pose mat from state.
      2. Convert each future absolute pose to relative: inv(T_base) @ T_future.
      3. Flatten to pose10d (9,) and append gripper (1,) → 10D per robot.
    Concatenate both robots → 20D.

    This is an exact port of umi_dataset.py lines 334-366 with obs_pose_repr='relative'.
    """
    horizon = actions.shape[0]

    # ---- robot 0 ----
    r0_curr_pose = np.concatenate([state[_R0_POS], state[_R0_ROT]])  # (6,)
    r0_base_mat = _pose_to_mat(r0_curr_pose)  # (4,4) – single frame

    r0_fut_pose = np.concatenate(
        [actions[:, _A_R0_POS], actions[:, _A_R0_ROT]], axis=-1
    )  # (H,6)
    r0_fut_mat = _pose_to_mat(r0_fut_pose)  # (H,4,4)
    r0_rel_mat = _convert_pose_mat_relative(r0_fut_mat, r0_base_mat)  # (H,4,4)
    r0_pose10d = _mat_to_pose10d(r0_rel_mat)  # (H,9)
    r0_grip = actions[:, _A_R0_GRIP]  # (H,1)
    r0_action10d = np.concatenate([r0_pose10d, r0_grip], axis=-1)  # (H,10)

    # ---- robot 1 ----
    r1_curr_pose = np.concatenate([state[_R1_POS], state[_R1_ROT]])  # (6,)
    r1_base_mat = _pose_to_mat(r1_curr_pose)  # (4,4)

    r1_fut_pose = np.concatenate(
        [actions[:, _A_R1_POS], actions[:, _A_R1_ROT]], axis=-1
    )  # (H,6)
    r1_fut_mat = _pose_to_mat(r1_fut_pose)  # (H,4,4)
    r1_rel_mat = _convert_pose_mat_relative(r1_fut_mat, r1_base_mat)  # (H,4,4)
    r1_pose10d = _mat_to_pose10d(r1_rel_mat)  # (H,9)
    r1_grip = actions[:, _A_R1_GRIP]  # (H,1)
    r1_action10d = np.concatenate([r1_pose10d, r1_grip], axis=-1)  # (H,10)

    return np.concatenate([r0_action10d, r1_action10d], axis=-1).astype(np.float32)  # (H,20)


def _relative_pose10d_to_abs_actions(
    state: np.ndarray,
    rel_actions: np.ndarray,
) -> np.ndarray:
    """Inverse of _abs_actions_to_relative_pose10d.

    Converts predicted (H,20) relative pose10d back to (H,14) absolute axis-angle actions.
    Used during inference, mirroring HarMI's get_real_umi_action with 'relative' pose_repr.
    """
    def _decode_robot(curr_pose_6d: np.ndarray, rel_10d: np.ndarray):
        # rel_10d: (H,10)  → pos(3) + rot6d(6) + grip(1)
        base_mat = _pose_to_mat(curr_pose_6d)  # (4,4)
        h = rel_10d.shape[0]
        pos = rel_10d[:, :3]
        rot6d = rel_10d[:, 3:9]
        grip = rel_10d[:, 9:10]

        # Reconstruct absolute SE(3): base @ rel
        rot_mat = _rot6d_to_mat(rot6d)  # (H,3,3)
        rel_mat = np.zeros((h, 4, 4), dtype=np.float64)
        rel_mat[:, :3, :3] = rot_mat
        rel_mat[:, :3, 3] = pos
        rel_mat[:, 3, 3] = 1.0

        abs_mat = base_mat @ rel_mat  # (H,4,4)
        abs_pos = abs_mat[:, :3, 3]
        abs_rot = st.Rotation.from_matrix(abs_mat[:, :3, :3]).as_rotvec()
        return np.concatenate([abs_pos, abs_rot, grip], axis=-1).astype(np.float32)

    r0_abs = _decode_robot(
        np.concatenate([state[_R0_POS], state[_R0_ROT]]),
        rel_actions[:, :10],
    )  # (H,7)
    r1_abs = _decode_robot(
        np.concatenate([state[_R1_POS], state[_R1_ROT]]),
        rel_actions[:, 10:20],
    )  # (H,7)
    return np.concatenate([r0_abs, r1_abs], axis=-1)  # (H,14)


# ---------------------------------------------------------------------------
# DataTransformFn classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Convert raw UMI bimanual data into the format expected by pi0.

    Input dict keys (from LeRobot / inference environment):
      camera0_rgb   (H,W,3) or (3,H,W) uint8/float32
      camera1_rgb   (H,W,3) or (3,H,W) uint8/float32
      state         (26,) float32  – absolute axis-angle + demo_start_pose
      actions       (H,14) float32 – absolute axis-angle actions (training only)
      prompt        str             – task instruction

    Output dict keys (pi0 model format):
      state         (32,)    model state
      image         base_0_rgb        — zeros (no global camera), mask=False
                    left_wrist_0_rgb  — camera1_rgb, mask=True
                    right_wrist_0_rgb — camera0_rgb, mask=True
      image_mask    {same keys} bool
      actions       (H,32)   relative pose10d padded to model action_dim (training only)
      prompt        str
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state_raw = np.asarray(data["state"], dtype=np.float32)  # (26,)

        # Build model state
        model_state = _build_model_state(state_raw)  # (32,)

        # Images
        cam0 = _parse_image(data["camera0_rgb"])   # (H,W,3) uint8
        cam1 = _parse_image(data["camera1_rgb"])   # (H,W,3) uint8

        # No global/base camera on this UMI setup — only the two wrist cameras.
        # The zero-filled slot is masked out, matching ur_policy.py's convention
        # for wrist-only setups. camera1_rgb is the left wrist, camera0_rgb is
        # the right wrist.
        black = np.zeros_like(cam0)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (black, cam1, cam0)
                image_masks = (np.False_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (black, cam1, cam0)
                image_masks = (np.False_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs: dict = {
            "state": model_state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions: absolute (H,14) → relative pose10d (H,20) → pad to (H,32)
        if "actions" in data:
            actions_abs = np.asarray(data["actions"], dtype=np.float32)  # (H,14)
            actions_rel = _abs_actions_to_relative_pose10d(state_raw, actions_abs)  # (H,20)
            inputs["actions"] = transforms.pad_to_dim(actions_rel, 32)  # (H,32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """Convert pi0 action output back to 20D relative pose10d (and optionally to absolute).

    Returns the first 20 dimensions of the predicted action chunk as-is
    (relative SE3 pose10d in HarMI's format).  HarMI's `get_real_umi_action`
    with ``action_pose_repr='relative'`` handles the final conversion to
    absolute axis-angle at deploy time.
    """

    def __call__(self, data: dict) -> dict:
        # Strip padding: keep only the 20 real action dims.
        return {"actions": np.asarray(data["actions"][:, :20])}


def make_umi_example() -> dict:
    """Creates a random inference-time input dict for the UMI policy."""
    return {
        "camera0_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "camera1_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "state": np.random.randn(26).astype(np.float32),
        "prompt": "fold cloth",
    }
