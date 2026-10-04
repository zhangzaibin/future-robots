"""
Convert two_arms_place_food H5 dataset to LeRobot format (two agents).

Supports two modes:
  1. Regular mode: single task instruction for the whole trajectory
  2. Phase mode (--use_phase): per-phase atomic instructions from the phase-labeled H5 file

4-phase scheme (gripper + movement stability):
  Phase 1: A0: "Stay still."                       A1: "Grasp the pot lid."
  Phase 2: A0: "Stay still."                       A1: "Open the pot lid."
  Phase 3: A0: "Pick up the meat."                 A1: "Stay still."
  Phase 4: A0: "Place the meat into the pot."      A1: "Stay still."

Usage:
  # Regular mode
  python scripts/set_atom/macg/two_arms_place_food/convert_h5_lerobot_place_food.py \
    --h5_path dataset/MACG/two_arms_place_food/20260317_135714.h5 \
    --output_dir dataset/lerobot \
    --task_name two_arms_place_food \
    --task_instruction_agent0 "Pick up the meat and place it into the pot." \
    --task_instruction_agent1 "Open the pot lid and hold it."

  # Phase mode
  python scripts/set_atom/macg/two_arms_place_food/convert_h5_lerobot_place_food.py \
    --h5_path dataset/MACG/two_arms_place_food/20260317_135714_with_phases.h5 \
    --output_dir dataset/lerobot \
    --task_name two_arms_place_food \
    --use_phase
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


PHASE_INSTRUCTIONS = {
    1: {"agent0": "Wait for the other robots.",                    "agent1": "Pick up the pot lid."},
    2: {"agent0": "Wait for the other robots.",                    "agent1": "Open the pot lid."},
    3: {"agent0": "Pick up the meat.",              "agent1": "Wait for the other robots."},
    4: {"agent0": "Place the meat into the pot.",   "agent1": "Wait for the other robots."},
}


def _create_dataset(repo_id, use_phase=False):
    features = {
        "image_agent0": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "image_agent1": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "global_image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position_agent0": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "joint_position_agent1": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position_agent0": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "gripper_position_agent1": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (16,),
            "names": ["actions"],
        },
    }
    if use_phase:
        features["phase"] = {
            "dtype": "int32",
            "shape": (1,),
            "names": ["phase"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=30,
        features=features,
        image_writer_threads=4,
    )


def convert_to_lerobot(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    push_to_hub: bool = False,
):
    """Convert H5 dataset to LeRobot format with a single task instruction."""
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )

        repo_id = f"{task_name}_combined"
        print(f"\nCreating dataset '{repo_id}' ({len(traj_keys)} trajectories)...")
        dataset = _create_dataset(repo_id, use_phase=False)

        combined_instruction = f"{task_instruction_agent0} | {task_instruction_agent1}"

        for traj_key in traj_keys:
            try:
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]

                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image))

                for t in range(T):
                    s0 = actions_0[t].astype(np.float32)
                    s1 = actions_1[t].astype(np.float32)

                    dataset.add_frame({
                        "image_agent0": image_0[t],
                        "image_agent1": image_1[t],
                        "global_image": global_image[t],
                        "joint_position_agent0": s0[:7],
                        "joint_position_agent1": s1[:7],
                        "gripper_position_agent0": np.array([s0[7]], dtype=np.float32),
                        "gripper_position_agent1": np.array([s1[7]], dtype=np.float32),
                        "actions": np.concatenate([s0, s1]),
                        "task": combined_instruction,
                    })
                dataset.save_episode()
                print(f"  {traj_key}: {T} frames")

            except Exception as e:
                print(f"  [Warning] Failed to process {traj_key}: {e}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["place_food", "panda", "combined_agents", task_name],
            private=True, push_videos=True, license="apache-2.0",
        )
    print(f"\nDone! Dataset saved to: {Path(output_dir) / repo_id}")


def convert_to_lerobot_with_phase(
    h5_path: str,
    output_dir: str,
    task_name: str,
    phase_instructions: dict = None,
    push_to_hub: bool = False,
):
    """Convert phase-labeled H5 dataset to LeRobot format with per-phase instructions."""
    if phase_instructions is None:
        phase_instructions = PHASE_INSTRUCTIONS

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )

        repo_id = f"{task_name}_combined"
        print(f"\nCreating phase dataset '{repo_id}' ({len(traj_keys)} trajectories)...")
        dataset = _create_dataset(repo_id, use_phase=True)

        for traj_key in traj_keys:
            try:
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]
                phase_data = f[f"{traj_key}/phase"][:]

                T = min(len(actions_0), len(actions_1), len(image_0),
                        len(image_1), len(global_image), len(phase_data))

                for t in range(T):
                    s0 = actions_0[t].astype(np.float32)
                    s1 = actions_1[t].astype(np.float32)

                    current_phase = int(phase_data[t])
                    if current_phase not in phase_instructions:
                        raise ValueError(f"Phase {current_phase} not in phase_instructions")
                    instr = phase_instructions[current_phase]
                    combined_instruction = f"{instr['agent0']} | {instr['agent1']}"

                    dataset.add_frame({
                        "image_agent0": image_0[t],
                        "image_agent1": image_1[t],
                        "global_image": global_image[t],
                        "joint_position_agent0": s0[:7],
                        "joint_position_agent1": s1[:7],
                        "gripper_position_agent0": np.array([s0[7]], dtype=np.float32),
                        "gripper_position_agent1": np.array([s1[7]], dtype=np.float32),
                        "actions": np.concatenate([s0, s1]),
                        "phase": np.array([current_phase], dtype=np.int32),
                        "task": combined_instruction,
                    })
                dataset.save_episode()
                print(f"  {traj_key}: {T} frames")

            except Exception as e:
                print(f"  [Warning] Failed to process {traj_key}: {e}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["place_food", "panda", "combined_agents", task_name, "multi-phase"],
            private=True, push_videos=True, license="apache-2.0",
        )
    print(f"\nDone! Dataset saved to: {Path(output_dir) / repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert two_arms_place_food H5 to LeRobot format"
    )
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the converted dataset")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task")
    parser.add_argument("--use_phase", action="store_true", default=False,
                        help="Use phase-based instruction mapping")
    parser.add_argument("--task_instruction_agent0", type=str,
                        help="Instruction for agent 0 (required if not using phase)")
    parser.add_argument("--task_instruction_agent1", type=str,
                        help="Instruction for agent 1 (required if not using phase)")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push dataset to Hugging Face Hub")

    args = parser.parse_args()

    if args.use_phase:
        convert_to_lerobot_with_phase(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            push_to_hub=args.push_to_hub,
        )
    else:
        if not args.task_instruction_agent0 or not args.task_instruction_agent1:
            raise ValueError(
                "task_instruction_agent0 and task_instruction_agent1 are required "
                "when not using phase mode"
            )
        convert_to_lerobot(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            push_to_hub=args.push_to_hub,
        )
