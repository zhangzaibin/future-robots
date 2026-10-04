#!/usr/bin/env python3
"""
Convert LeRobot v3.0 dataset format to v2.1 format for openpi training.

v3.0 format:
- Uses video files for image storage
- Uses parquet files for metadata (tasks.parquet, episodes.parquet)
- data_path format: data/chunk-{chunk_index}/file-{file_index}.parquet

v2.1 format:
- Images embedded in parquet files
- Uses jsonl files for metadata (tasks.jsonl, episodes.jsonl)
- data_path format: data/chunk-{episode_chunk}/episode_{episode_index}.parquet
- Requires norm_stats.json in root or assets directory

Supports both single-arm and dual-arm robots with auto-detection.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage
from tqdm import tqdm

# Use HuggingFace datasets for proper Image handling
from datasets import Dataset, Features, Image, Sequence, Value


# ============================================================================
# Robot Configuration Presets
# ============================================================================

@dataclass
class RobotConfig:
    """Robot configuration for dataset conversion."""
    robot_type: str
    num_arms: int
    state_dim: int
    action_dim: int
    cameras: list[str]
    camera_mapping: dict[str, str]  # src_key -> dst_key
    joint_names: list[str]
    pad_dim: int = 32


# Preset configurations for known robot types
ROBOT_CONFIGS = {
    # Single-arm SO101
    "so101_follower": RobotConfig(
        robot_type="so101_follower",
        num_arms=1,
        state_dim=6,
        action_dim=6,
        cameras=["observation.images.front", "observation.images.side"],
        camera_mapping={
            "observation.images.front": "image",
            "observation.images.side": "wrist_image",
        },
        joint_names=[
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ],
        pad_dim=32,
    ),
    # Dual-arm SO100 (bi_so100_follower)
    "bi_so100_follower": RobotConfig(
        robot_type="bi_so100_follower",
        num_arms=2,
        state_dim=12,
        action_dim=12,
        cameras=["observation.images.orbbec_rgb", "observation.images.left", "observation.images.right"],
        camera_mapping={
            "observation.images.orbbec_rgb": "image",       # Global camera -> main image
            "observation.images.left": "wrist_image",       # Left wrist -> wrist_image
            "observation.images.right": "image_agent1",     # Right wrist -> image_agent1
        },
        joint_names=[
            "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
            "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
            "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
            "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos"
        ],
        pad_dim=32,
    ),
}


def detect_robot_config(src_info: dict) -> RobotConfig:
    """
    Auto-detect robot configuration from source info.json.
    Falls back to building config from features if robot_type is unknown.
    """
    robot_type = src_info.get("robot_type", "unknown")
    
    # Check if we have a preset for this robot type
    if robot_type in ROBOT_CONFIGS:
        print(f"Using preset configuration for robot_type: {robot_type}")
        return ROBOT_CONFIGS[robot_type]
    
    # Build config dynamically from features
    print(f"Unknown robot_type '{robot_type}', building config from features...")
    features = src_info.get("features", {})
    
    # Detect cameras
    cameras = [k for k in features.keys() if k.startswith("observation.images.")]
    
    # Detect state/action dimensions
    state_shape = features.get("observation.state", {}).get("shape", [6])
    action_shape = features.get("action", {}).get("shape", [6])
    state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape
    action_dim = action_shape[0] if isinstance(action_shape, list) else action_shape
    
    # Infer number of arms from state dimension
    num_arms = 2 if state_dim > 8 else 1
    
    # Build camera mapping heuristically
    camera_mapping = {}
    for cam in cameras:
        cam_name = cam.split(".")[-1].lower()
        if any(k in cam_name for k in ["front", "high", "global", "orbbec", "top"]):
            camera_mapping[cam] = "image"
        elif any(k in cam_name for k in ["left", "side"]):
            camera_mapping[cam] = "wrist_image"
        elif "right" in cam_name:
            camera_mapping[cam] = "image_agent1"
        else:
            # Fallback: use camera name as key
            camera_mapping[cam] = f"image_{cam_name}"
    
    # Get joint names from features
    joint_names = features.get("observation.state", {}).get("names", [])
    if not joint_names:
        joint_names = [f"joint_{i}" for i in range(state_dim)]
    
    return RobotConfig(
        robot_type=robot_type,
        num_arms=num_arms,
        state_dim=state_dim,
        action_dim=action_dim,
        cameras=cameras,
        camera_mapping=camera_mapping,
        joint_names=joint_names,
        pad_dim=32,
    )

# Try to use PyAV for better AV1 support, fallback to OpenCV
try:
    import av
    USE_PYAV = True
except ImportError:
    import cv2
    USE_PYAV = False
    print("Warning: PyAV not available, using OpenCV (may have issues with AV1 codec)")


class MultiFileVideoReader:
    """
    Sequential reader for all video files belonging to one camera.

    Source rows are ordered by global frame index, so decoded frames can be
    discarded immediately instead of caching the entire dataset in RAM.
    """
    
    def __init__(self, video_paths: list[str]):
        self.video_paths = video_paths
        self._frames = self._iter_frames()
        self._current_index = -1
        self._current_frame: Optional[np.ndarray] = None
    
    def _iter_frames(self):
        """Decode video files lazily, retaining only the current frame."""
        for path in self.video_paths:
            frame_count = 0
            if USE_PYAV:
                with av.open(path) as container:
                    for frame in container.decode(video=0):
                        frame_count += 1
                        yield frame.to_ndarray(format="rgb24")
            else:
                cap = cv2.VideoCapture(path)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                finally:
                    cap.release()
            print(f"  Streamed {frame_count} frames from {path}")
    
    def get_frame(self, global_frame_index: int) -> np.ndarray:
        """Read forward to a global frame index across all video files."""
        if global_frame_index < self._current_index:
            raise ValueError(
                f"Cannot seek backwards from frame {self._current_index} "
                f"to frame {global_frame_index}"
            )
        
        while self._current_index < global_frame_index:
            try:
                self._current_frame = next(self._frames)
            except StopIteration as exc:
                raise ValueError(
                    f"Frame index {global_frame_index} exceeds available video frames"
                ) from exc
            self._current_index += 1

        if self._current_frame is None:
            raise ValueError(f"Failed to decode frame {global_frame_index}")
        return self._current_frame


def discover_video_files(src_path: Path, camera_key: str) -> list[Path]:
    """
    Discover all video files for a camera, handling multiple chunks/files.
    Returns sorted list of video file paths.
    """
    video_dir = src_path / "videos" / camera_key
    if not video_dir.exists():
        return []
    
    # Find all video files, sorted by chunk and file index
    video_files = sorted(video_dir.glob("chunk-*/file-*.mp4"))
    return video_files


def load_camera_videos(src_path: Path, robot_config: RobotConfig) -> dict[str, MultiFileVideoReader]:
    """
    Open lazy camera video readers based on robot configuration.
    Returns a dictionary mapping camera keys to video readers.
    """
    print("Opening video streams...")
    video_readers = {}
    
    for camera_key in robot_config.cameras:
        video_files = discover_video_files(src_path, camera_key)
        if video_files:
            print(f"  Camera '{camera_key}': streaming from {len(video_files)} video file(s)")
            video_readers[camera_key] = MultiFileVideoReader([str(f) for f in video_files])
        else:
            print(f"  Warning: No video files found for camera '{camera_key}'")
    
    return video_readers


def image_to_pil(image: np.ndarray) -> PILImage.Image:
    """Convert numpy image to PIL Image."""
    return PILImage.fromarray(image)


def read_parquet_parts(root: Path, index_column: str) -> pd.DataFrame:
    """Read all LeRobot shards, removing overlap at shard boundaries."""
    paths = sorted(root.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards found under {root}")
    data = pd.concat(
        [pq.read_table(path).to_pandas() for path in paths],
        ignore_index=True,
    )
    return (
        data.drop_duplicates(subset=index_column, keep="first")
        .sort_values(index_column)
        .reset_index(drop=True)
    )


def compute_norm_stats(data_df: pd.DataFrame, state_key: str = "observation.state", 
                       action_key: str = "action") -> dict:
    """Compute normalization statistics for states and actions."""
    
    def compute_stats(values: list) -> dict:
        arr = np.array(values)
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }
    
    # Extract state and action values
    states = data_df[state_key].tolist()
    actions = data_df[action_key].tolist()
    
    return {
        "norm_stats": {
            "state": compute_stats(states),
            "actions": compute_stats(actions),
        }
    }


def convert_dataset(
    src_dir: str,
    dst_dir: str,
    target_image_size: tuple = (224, 224),
    default_prompt: str = "Put the cup in the basket",
    robot_config: Optional[RobotConfig] = None,
):
    """
    Convert LeRobot v3.0 dataset to v2.1 format.
    
    Args:
        src_dir: Source dataset directory (v3.0 format)
        dst_dir: Destination dataset directory (v2.1 format)
        target_image_size: Target size for images (height, width)
        default_prompt: Default task prompt if not specified
        robot_config: Optional robot configuration. If None, will auto-detect.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Load source metadata
    print("Loading source metadata...")
    with open(src_path / "meta" / "info.json", "r") as f:
        src_info = json.load(f)
    
    # Auto-detect or use provided robot config
    if robot_config is None:
        robot_config = detect_robot_config(src_info)
    
    print(f"Robot configuration:")
    print(f"  Type: {robot_config.robot_type}")
    print(f"  Arms: {robot_config.num_arms}")
    print(f"  State dim: {robot_config.state_dim}")
    print(f"  Action dim: {robot_config.action_dim}")
    print(f"  Cameras: {robot_config.cameras}")
    print(f"  Camera mapping: {robot_config.camera_mapping}")
    
    # Load tasks from parquet
    tasks_parquet = pq.read_table(src_path / "meta" / "tasks.parquet").to_pandas()
    # tasks.parquet has task as index, task_index as column
    task_mapping = {}
    for task_name, row in tasks_parquet.iterrows():
        task_mapping[int(row['task_index'])] = str(task_name)
    
    print(f"Found {len(task_mapping)} tasks: {task_mapping}")
    
    # Load all episode metadata shards
    episodes_df = read_parquet_parts(
        src_path / "meta" / "episodes",
        index_column="episode_index",
    )
    
    # Load all source data shards
    print("Loading source data...")
    data_df = read_parquet_parts(src_path / "data", index_column="index")
    
    # Load camera videos using robot config
    video_readers = load_camera_videos(src_path, robot_config)
    
    # Create destination directories
    (dst_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_path / "meta").mkdir(parents=True, exist_ok=True)
    
    def compute_array_stats(arr: np.ndarray) -> dict:
        """Compute statistics for an array."""
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }
    
    # Process each episode
    print("Processing episodes...")
    episodes_info = []
    episodes_stats_list = []
    all_data_for_stats = []
    
    episode_indices = data_df['episode_index'].unique()
    
    for episode_idx in tqdm(sorted(episode_indices), desc="Converting episodes"):
        episode_data = data_df[data_df['episode_index'] == episode_idx].copy()
        episode_meta = episodes_df[episodes_df['episode_index'] == episode_idx].iloc[0]
        
        # Get task info for this episode
        task_list = episode_meta.get('tasks', [task_mapping.get(0, default_prompt)])
        if isinstance(task_list, (list, np.ndarray)) and len(task_list) > 0:
            primary_task = task_list[0] if isinstance(task_list[0], str) else default_prompt
        else:
            primary_task = default_prompt
        
        # Process each frame in the episode
        episode_records = []
        episode_states = []
        episode_actions = []
        
        for _, row in episode_data.iterrows():
            global_idx = int(row['index'])
            
            # Load images from all cameras using robot config mapping
            images_dict = {}
            for src_camera, dst_key in robot_config.camera_mapping.items():
                if src_camera in video_readers:
                    try:
                        image = video_readers[src_camera].get_frame(global_idx)
                        # Resize if needed
                        if image.shape[:2] != target_image_size:
                            image = np.array(PILImage.fromarray(image).resize(target_image_size[::-1]))
                        images_dict[dst_key] = image_to_pil(image)
                    except Exception as e:
                        print(f"Warning: Failed to load frame {global_idx} from {src_camera}: {e}")
                        # Use black image as fallback
                        black_image = np.zeros((*target_image_size, 3), dtype=np.uint8)
                        images_dict[dst_key] = image_to_pil(black_image)
                else:
                    # Camera not available, use black image
                    black_image = np.zeros((*target_image_size, 3), dtype=np.uint8)
                    images_dict[dst_key] = image_to_pil(black_image)
            
            # Get state and action
            state = np.array(row['observation.state']).astype(np.float32).tolist()
            action = np.array(row['action']).astype(np.float32).tolist()
            
            episode_states.append(state)
            episode_actions.append(action)
            
            # Build record with dynamic image keys
            record = {
                **images_dict,  # Unpack all camera images
                'state': state,
                'actions': action,
                'timestamp': float(row['timestamp']),
                'frame_index': int(row['frame_index']),
                'episode_index': int(episode_idx),
                'index': int(row['index']),
                'task_index': int(row['task_index']),
            }
            episode_records.append(record)
            all_data_for_stats.append({'state': np.array(state), 'actions': np.array(action)})
        
        # Save episode parquet using HuggingFace datasets for proper Image handling
        episode_parquet_path = dst_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        
        # Define features with proper Image type (dynamic based on camera mapping)
        feature_dict = {}
        # Add image features for each camera
        for dst_key in robot_config.camera_mapping.values():
            feature_dict[dst_key] = Image()
        # Add other features
        feature_dict.update({
            'state': Sequence(Value('float32')),
            'actions': Sequence(Value('float32')),
            'timestamp': Value('float32'),
            'frame_index': Value('int64'),
            'episode_index': Value('int64'),
            'index': Value('int64'),
            'task_index': Value('int64'),
        })
        features = Features(feature_dict)
        
        # Create HuggingFace dataset and save
        hf_dataset = Dataset.from_list(episode_records, features=features)
        hf_dataset.to_parquet(str(episode_parquet_path))
        
        # Compute episode statistics
        states_arr = np.array([np.array(s) for s in episode_states])
        actions_arr = np.array([np.array(a) for a in episode_actions])
        timestamps = np.array([r['timestamp'] for r in episode_records])
        frame_indices = np.array([r['frame_index'] for r in episode_records])
        indices = np.array([r['index'] for r in episode_records])
        task_indices = np.array([r['task_index'] for r in episode_records])
        ep_indices = np.array([r['episode_index'] for r in episode_records])
        
        episode_stats = {
            "episode_index": int(episode_idx),
            "stats": {
                "state": compute_array_stats(states_arr),
                "actions": compute_array_stats(actions_arr),
                "timestamp": compute_array_stats(timestamps.reshape(-1, 1)),
                "frame_index": compute_array_stats(frame_indices.reshape(-1, 1)),
                "episode_index": compute_array_stats(ep_indices.reshape(-1, 1)),
                "index": compute_array_stats(indices.reshape(-1, 1)),
                "task_index": compute_array_stats(task_indices.reshape(-1, 1)),
            }
        }
        episodes_stats_list.append(episode_stats)
        
        # Record episode info
        episodes_info.append({
            "episode_index": int(episode_idx),
            "tasks": [primary_task],
            "length": len(episode_records),
        })
    
    # Write episodes.jsonl
    print("Writing metadata files...")
    with open(dst_path / "meta" / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")
    
    # Write episodes_stats.jsonl
    with open(dst_path / "meta" / "episodes_stats.jsonl", "w") as f:
        for ep_stats in episodes_stats_list:
            f.write(json.dumps(ep_stats) + "\n")
    
    # Write tasks.jsonl
    with open(dst_path / "meta" / "tasks.jsonl", "w") as f:
        for task_idx, task_name in task_mapping.items():
            f.write(json.dumps({"task_index": task_idx, "task": task_name}) + "\n")
    
    # Write info.json in v2.1 format
    total_frames = sum(ep['length'] for ep in episodes_info)
    
    # Build features dynamically based on robot config
    output_features = {}
    
    # Add image features for each camera
    for dst_key in robot_config.camera_mapping.values():
        output_features[dst_key] = {
            "dtype": "image",
            "shape": [target_image_size[0], target_image_size[1], 3],
            "names": ["height", "width", "channel"]
        }
    
    # Add state and actions with proper dimensions from robot config
    output_features["state"] = {
        "dtype": "float32",
        "shape": [robot_config.state_dim],
        "names": robot_config.joint_names
    }
    output_features["actions"] = {
        "dtype": "float32",
        "shape": [robot_config.action_dim],
        "names": robot_config.joint_names
    }
    
    # Add standard metadata features
    output_features.update({
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        }
    })
    
    info_v2 = {
        "codebase_version": "v2.1",
        "robot_type": robot_config.robot_type,
        "total_episodes": len(episodes_info),
        "total_frames": total_frames,
        "total_tasks": len(task_mapping),
        "total_videos": 0,  # Images are embedded
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": src_info.get("fps", 30),
        "splits": {
            "train": f"0:{len(episodes_info)}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": output_features,
        # Additional info for dual-arm support
        "num_arms": robot_config.num_arms,
        "state_dim": robot_config.state_dim,
        "action_dim": robot_config.action_dim,
    }
    
    with open(dst_path / "meta" / "info.json", "w") as f:
        json.dump(info_v2, f, indent=4)
    
    # Compute and write norm_stats.json
    print("Computing normalization statistics...")
    stats_df = pd.DataFrame(all_data_for_stats)
    
    states = np.array(stats_df['state'].tolist())
    actions = np.array(stats_df['actions'].tolist())
    
    # Pad to target dimension (configurable, default 32 for pi0 model)
    pad_dim = robot_config.pad_dim
    
    def pad_stats(arr, target_dim):
        if arr.shape[1] < target_dim:
            padded = np.zeros((arr.shape[0], target_dim))
            padded[:, :arr.shape[1]] = arr
            return padded
        return arr
    
    states_padded = pad_stats(states, pad_dim)
    actions_padded = pad_stats(actions, pad_dim)
    
    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": states_padded.mean(axis=0).tolist(),
                "std": np.maximum(states_padded.std(axis=0), 1e-6).tolist(),  # Avoid div by zero
                "q01": np.percentile(states_padded, 1, axis=0).tolist(),
                "q99": np.percentile(states_padded, 99, axis=0).tolist(),
            },
            "actions": {
                "mean": actions_padded.mean(axis=0).tolist(),
                "std": np.maximum(actions_padded.std(axis=0), 1e-6).tolist(),
                "q01": np.percentile(actions_padded, 1, axis=0).tolist(),
                "q99": np.percentile(actions_padded, 99, axis=0).tolist(),
            }
        }
    }
    
    with open(dst_path / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Source: {src_dir}")
    print(f"  Destination: {dst_dir}")
    print(f"  Episodes: {len(episodes_info)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Tasks: {task_mapping}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 to v2.1 format (supports single-arm and dual-arm robots)"
    )
    parser.add_argument("--src", type=str, required=True, 
                       help="Source dataset directory (v3.0)")
    parser.add_argument("--dst", type=str, required=True, 
                       help="Destination dataset directory (v2.1)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224],
                       help="Target image size (height width)")
    parser.add_argument("--default-prompt", type=str, default="Put the cup in the basket",
                       help="Default task prompt")
    
    # Robot configuration options
    parser.add_argument("--robot-type", type=str, default=None,
                       choices=list(ROBOT_CONFIGS.keys()) + [None],
                       help="Robot type preset. If not specified, auto-detected from source info.json")
    parser.add_argument("--pad-dim", type=int, default=32,
                       help="Padding dimension for norm stats (default: 32 for pi0 model)")
    
    # Camera mapping override (advanced)
    parser.add_argument("--camera-mapping", type=str, nargs="+", default=None,
                       help="Override camera mapping as 'src_key:dst_key' pairs. "
                            "Example: 'observation.images.front:image observation.images.side:wrist_image'")
    
    args = parser.parse_args()
    
    # Prepare robot config
    robot_config = None
    if args.robot_type is not None:
        robot_config = ROBOT_CONFIGS.get(args.robot_type)
        if robot_config is None:
            print(f"Warning: Unknown robot type '{args.robot_type}', will auto-detect")
    
    # Override camera mapping if provided
    if robot_config is not None and args.camera_mapping is not None:
        camera_mapping = {}
        for mapping in args.camera_mapping:
            src_key, dst_key = mapping.split(":")
            camera_mapping[src_key] = dst_key
        # Create new config with overridden camera mapping
        robot_config = RobotConfig(
            robot_type=robot_config.robot_type,
            num_arms=robot_config.num_arms,
            state_dim=robot_config.state_dim,
            action_dim=robot_config.action_dim,
            cameras=list(camera_mapping.keys()),
            camera_mapping=camera_mapping,
            joint_names=robot_config.joint_names,
            pad_dim=args.pad_dim,
        )
    elif robot_config is not None:
        # Update pad_dim if provided
        robot_config = RobotConfig(
            robot_type=robot_config.robot_type,
            num_arms=robot_config.num_arms,
            state_dim=robot_config.state_dim,
            action_dim=robot_config.action_dim,
            cameras=robot_config.cameras,
            camera_mapping=robot_config.camera_mapping,
            joint_names=robot_config.joint_names,
            pad_dim=args.pad_dim,
        )
    
    convert_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        target_image_size=tuple(args.image_size),
        default_prompt=args.default_prompt,
        robot_config=robot_config,
    )


if __name__ == "__main__":
    main()

