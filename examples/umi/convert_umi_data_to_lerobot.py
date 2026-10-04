"""
Convert a bimanual UMI dataset.zarr.zip to LeRobot format.

The zarr replay buffer is produced by HarMI's SLAM pipeline step 07 and contains:
  data/camera{0,1}_rgb  – uint8 (T,224,224,3) JPEG-XL compressed
  data/robot{0,1}_eef_pos             – float32 (T,3)  absolute TCP position
  data/robot{0,1}_eef_rot_axis_angle  – float32 (T,3)  axis-angle rotation
  data/robot{0,1}_gripper_width       – float32 (T,1)  gripper width
  data/robot{0,1}_demo_start_pose     – float64 (T,6)  episode-start pose (pos+rotvec)
  data/robot{0,1}_demo_end_pose       – float64 (T,6)  episode-end pose
  meta/episode_ends                   – int64   (N_eps,) cumulative frame indices

Stored LeRobot features (absolute, downsampled every `--downsample` frames):
  camera0_rgb   image (224,224,3)
  camera1_rgb   image (224,224,3)
  state         float32 (26,)  = [r0_pos(3), r0_rot(3), r0_grip(1),
                                   r1_pos(3), r1_rot(3), r1_grip(1),
                                   r0_demo_start(6), r1_demo_start(6)]
  actions       float32 (14,)  = [r0_pos(3), r0_rot(3), r0_grip(1),
                                   r1_pos(3), r1_rot(3), r1_grip(1)]

Usage:
  uv run examples/umi/convert_umi_data_to_lerobot.py \\
      --zarr_path /path/to/dataset.zarr.zip \\
      --output_dir /path/to/output \\
      --repo_id my_org/umi_bimanual \\
      --task_prompt "fold cloth" \\
      --downsample 3
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tyro


def _interpolate_nans_per_episode(arr: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along the time axis, independently per episode.

    Some UMI sensor streams (e.g. gripper width) occasionally drop a few frames,
    leaving NaNs in the raw zarr arrays. These must not leak into the LeRobot
    dataset: they propagate through downstream pose/state transforms and poison
    the running mean/std used for normalization (turning whole feature columns
    into NaN, which then serializes as `null` in norm_stats.json).
    """
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr

    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    for start, end in zip(ep_starts, episode_ends, strict=True):
        seg = arr[start:end]
        seg_nan = nan_mask[start:end]
        if not seg_nan.any():
            continue
        idx = np.arange(len(seg))
        for col in range(seg.shape[1]):
            col_nan = seg_nan[:, col]
            if not col_nan.any():
                continue
            valid = ~col_nan
            if not valid.any():
                raise ValueError(
                    f"Episode [{start}:{end}] has an entirely-NaN column {col}; cannot interpolate."
                )
            seg[col_nan, col] = np.interp(idx[col_nan], idx[valid], seg[valid, col])
        arr[start:end] = seg
    return arr

# Register imagecodecs so zarr can decode JPEG-XL chunks
try:
    import imagecodecs.numcodecs as _ic_nc

    _ic_nc.register_codecs()
except ImportError:
    raise ImportError(
        "imagecodecs is required to decode the JPEG-XL camera arrays.\n"
        "Install with:  uv pip install imagecodecs"
    )

import zarr  # noqa: E402 – must come after register_codecs

# NOTE: lerobot must be imported AFTER setting HF_LEROBOT_HOME so that the
# output path is honoured.  The actual import is deferred into main().


def main(
    zarr_path: str,
    output_dir: str = "./datasets",
    repo_id: str = "local/umi_bimanual",
    task_prompt: str = "perform manipulation",
    downsample: int = 3,
    *,
    overwrite: bool = False,
) -> None:
    """
    Args:
        zarr_path:    Path to dataset.zarr.zip produced by HarMI step 07.
        output_dir:   Root directory where the LeRobot dataset will be written.
                      The dataset is saved to <output_dir>/<repo_id>.
        repo_id:      Repository / dataset identifier (e.g. "local/umi_bimanual").
        task_prompt:  Language instruction stored as the task string for every frame.
        downsample:   Keep every N-th frame (default 3, matching HarMI obs_down_sample_steps=3).
        overwrite:    Delete the output directory if it already exists.
    """
    import os

    # Set env var BEFORE importing lerobot so HF_LEROBOT_HOME is honoured.
    os.environ["HF_LEROBOT_HOME"] = str(output_dir)
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if overwrite:
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to delete it."
            )

    # ------------------------------------------------------------------ #
    # Open zarr replay buffer
    # ------------------------------------------------------------------ #
    print(f"Opening {zarr_path} ...")
    _zip_store = zarr.storage.ZipStore(zarr_path, mode="r")
    with _zip_store as zs:
        root = zarr.open_group(zs, mode="r")

        episode_ends: np.ndarray = root["meta"]["episode_ends"][:]  # (N_eps,)
        n_total = int(episode_ends[-1])

        # Load all arrays into memory (the dataset is ~38 k frames × small lowdim).
        r0_pos = root["data"]["robot0_eef_pos"][:]           # (T,3) float32
        r0_rot = root["data"]["robot0_eef_rot_axis_angle"][:]  # (T,3) float32
        r0_grip = root["data"]["robot0_gripper_width"][:]    # (T,1) float32
        r0_start = root["data"]["robot0_demo_start_pose"][:].astype(np.float32)  # (T,6)

        r1_pos = root["data"]["robot1_eef_pos"][:]
        r1_rot = root["data"]["robot1_eef_rot_axis_angle"][:]
        r1_grip = root["data"]["robot1_gripper_width"][:]
        r1_start = root["data"]["robot1_demo_start_pose"][:].astype(np.float32)

        # Sensor dropout occasionally leaves a handful of NaN frames in the
        # gripper-width streams; fill them in before they leak downstream.
        for name, arr in (("robot0_gripper_width", r0_grip), ("robot1_gripper_width", r1_grip)):
            n_nan = int(np.isnan(arr).sum())
            if n_nan:
                print(f"  Interpolating {n_nan} NaN frame(s) in {name} ...")
        r0_grip = _interpolate_nans_per_episode(r0_grip, episode_ends)
        r1_grip = _interpolate_nans_per_episode(r1_grip, episode_ends)

        # Camera arrays stay in zarr (lazy read per chunk) to keep RAM low.
        cam0 = root["data"]["camera0_rgb"]  # zarr array, read per-frame
        cam1 = root["data"]["camera1_rgb"]

        # ------------------------------------------------------------------ #
        # Build state and actions arrays
        # ------------------------------------------------------------------ #
        # state (26,) = [r0_pos, r0_rot, r0_grip, r1_pos, r1_rot, r1_grip,
        #                r0_demo_start, r1_demo_start]
        state_all = np.concatenate(
            [r0_pos, r0_rot, r0_grip, r1_pos, r1_rot, r1_grip, r0_start, r1_start],
            axis=-1,
        ).astype(np.float32)  # (T,26)

        # actions (14,) = [r0_pos, r0_rot, r0_grip, r1_pos, r1_rot, r1_grip]
        # These are the absolute TCP targets stored at each frame (identical to
        # the raw lowdim; HarMI's SequenceSampler uses them as future targets).
        actions_all = np.concatenate(
            [r0_pos, r0_rot, r0_grip, r1_pos, r1_rot, r1_grip], axis=-1
        ).astype(np.float32)  # (T,14)

        # ------------------------------------------------------------------ #
        # Create LeRobot dataset
        # ------------------------------------------------------------------ #
        fps = max(1, 60 // downsample)  # 60 Hz original → 20 Hz with downsample=3
        print(f"Creating LeRobot dataset '{repo_id}' at fps={fps} ...")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="umi_bimanual",
            fps=fps,
            features={
                "camera0_rgb": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "camera1_rgb": {
                    "dtype": "image",
                    "shape": (224, 224, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (26,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=8,
            image_writer_processes=4,
        )

        # ------------------------------------------------------------------ #
        # Write episodes
        # ------------------------------------------------------------------ #
        n_episodes = len(episode_ends)
        ep_start = 0

        for ep_idx in range(n_episodes):
            ep_end = int(episode_ends[ep_idx])
            ep_len = ep_end - ep_start

            # Collect downsampled frame indices (always include first frame)
            frame_indices = list(range(ep_start, ep_end, downsample))

            if len(frame_indices) == 0:
                ep_start = ep_end
                continue

            for t in frame_indices:
                img0 = cam0[t]  # (224,224,3) uint8
                img1 = cam1[t]

                dataset.add_frame(
                    {
                        "camera0_rgb": img0,
                        "camera1_rgb": img1,
                        "state": state_all[t],
                        "actions": actions_all[t],
                        "task": task_prompt,
                    }
                )

            dataset.save_episode()
            ep_start = ep_end

            if (ep_idx + 1) % 10 == 0 or ep_idx == n_episodes - 1:
                print(f"  Episode {ep_idx + 1}/{n_episodes} done  (frames: {ep_len} → {len(frame_indices)} downsampled)")

    print(f"\nDone. Dataset saved to {output_path}")
    print(f"  Total episodes : {n_episodes}")
    print(f"  Downsample     : every {downsample} frames  →  {fps} fps")


if __name__ == "__main__":
    tyro.cli(main)
