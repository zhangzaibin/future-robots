import argparse
from pathlib import Path
import shutil

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def convert_robofactory_to_lerobot_combined(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    push_to_hub: bool = False,
):
    """
    Convert RoboFactory H5 dataset to LeRobot format combining both agents' data.

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'stack_cube', 'pick_place')
        task_instruction_agent0: Natural language instruction for agent 0
        task_instruction_agent1: Natural language instruction for agent 1
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Create a standardized dataset name
        repo_id = f"{task_name}_combined"
        output_path = Path(output_dir) / repo_id
        if output_path.exists():
            shutil.rmtree(output_path)

        print(f"\nCreating combined dataset for task '{task_name}'...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=30,
            features={
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
                    "shape": (16,),  # Combined actions from both agents
                    "names": ["actions"],
                },
            },
            image_writer_threads=4,
        )

        print("\nProcessing data for both agents...")

        for traj_key in traj_keys:
            try:
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/head_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/head_camera_agent1/rgb"][:]  # (T, H, W, 3)

                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image))

                # Combine task instructions
                combined_instruction = f"agent0: {task_instruction_agent0} agent1: {task_instruction_agent1}"

                for t in range(T):
                    # Process agent 0 data
                    full_state_0 = actions_0[t].astype(np.float32)
                    action_7d_0 = full_state_0[:7]
                    action_1d_0 = full_state_0[7]

                    # Process agent 1 data
                    full_state_1 = actions_1[t].astype(np.float32)
                    action_7d_1 = full_state_1[:7]
                    action_1d_1 = full_state_1[7]

                    # Combine actions into 16-dimensional vector
                    combined_actions = np.concatenate([full_state_0, full_state_1])

                    dataset.add_frame(
                        {
                            "image_agent0": image_0[t],
                            "image_agent1": image_1[t],
                            "global_image": global_image[t],
                            "joint_position_agent0": action_7d_0,
                            "joint_position_agent1": action_7d_1,
                            "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                            "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                            "actions": combined_actions,
                            "task": combined_instruction,
                        }
                    )
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_robofactory_to_lerobot_combined_with_phase(
    h5_path: str, output_dir: str, task_name: str, phase_instructions: dict = None, push_to_hub: bool = False
):
    """
    Convert RoboFactory H5 dataset with phase information to LeRobot format, combining both agents.

    Args:
        h5_path: Path to the input H5 file (merged file with phase field)
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'pass_shoes')
        phase_instructions: Dictionary mapping phase values to instruction dictionaries
                          Example: {
                              1: {"agent0": "Grasp the shoes with agent 0", "agent1": "Wait for agent 0"},
                              2: {"agent0": "Place the shoes for handover", "agent1": "Prepare to receive"},
                              3: {"agent0": "Wait for agent 1", "agent1": "Grasp the shoes with agent 1"},
                              4: {"agent0": "Complete", "agent1": "Place the shoes in final position"}
                          }
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """

    # Default phase instructions
    if phase_instructions is None:
        phase_instructions = {
            1: {"agent0": "Grasp the meat.", "agent1": "Grasp the lid."},
            2: {"agent0": "Place the meat in the pot.", "agent1": "Lift the lid."},
        }

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Create a standardized dataset name
        repo_id = f"{task_name}_combined"
        output_path = Path(output_dir) / repo_id
        # if output_path.exists():
        #     shutil.rmtree(output_path)

        print(f"\nCreating combined dataset for task '{task_name}'...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=30,
            features={
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
                    "shape": (16,),  # Combined actions from both agents
                    "names": ["actions"],
                },
                "phase": {
                    "dtype": "int32",
                    "shape": (1,),
                    "names": ["phase"],
                },
            },
            image_writer_threads=4,
        )

        print("\nProcessing data for both agents...")

        for traj_key in traj_keys:
            try:
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/head_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/head_camera_agent1/rgb"][:]  # (T, H, W, 3)

                # Load global image and phase data
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)
                phase_data = f[f"{traj_key}/phase"][:]  # (T,)

                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image), len(phase_data))

                for t in range(T):
                    # Process agent 0 data
                    full_state_0 = actions_0[t].astype(np.float32)
                    action_7d_0 = full_state_0[:7]
                    action_1d_0 = full_state_0[7]

                    # Process agent 1 data
                    full_state_1 = actions_1[t].astype(np.float32)
                    action_7d_1 = full_state_1[:7]
                    action_1d_1 = full_state_1[7]

                    # Combine actions into 16-dimensional vector
                    combined_actions = np.concatenate([full_state_0, full_state_1])

                    # Get current phase and instructions
                    current_phase = int(phase_data[t])
                    if current_phase in phase_instructions:
                        instruction_0 = phase_instructions[current_phase]["agent0"]
                        instruction_1 = phase_instructions[current_phase]["agent1"]
                        combined_instruction = f"agent0: {instruction_0} agent1: {instruction_1}"
                    else:
                        raise ValueError(f"Phase {current_phase} not found in phase_instructions")

                    dataset.add_frame(
                        {
                            "image_agent0": image_0[t],
                            "image_agent1": image_1[t],
                            "global_image": global_image[t],
                            "joint_position_agent0": action_7d_0,
                            "joint_position_agent1": action_7d_1,
                            "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                            "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                            "actions": combined_actions,
                            "phase": np.array([current_phase], dtype=np.int32),
                            "task": combined_instruction,
                        }
                    )
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name, "multi-phase"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboFactory H5 dataset to LeRobot format (Combined Agents)")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted dataset")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (e.g., stack_cube, pick_place)")
    parser.add_argument(
        "--use_phase",
        action="store_true",
        default=False,
        help="Use phase-based instruction mapping (for merged files with phase field)",
    )
    parser.add_argument(
        "--task_instruction_agent0",
        type=str,
        help="Natural language instruction for agent 0 (required if not using phase)",
    )
    parser.add_argument(
        "--task_instruction_agent1",
        type=str,
        help="Natural language instruction for agent 1 (required if not using phase)",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push the dataset to Hugging Face Hub"
    )

    args = parser.parse_args()

    if args.use_phase:
        # Use phase-based conversion
        convert_robofactory_to_lerobot_combined_with_phase(
            h5_path=args.h5_path, output_dir=args.output_dir, task_name=args.task_name, push_to_hub=args.push_to_hub
        )
    else:
        # Use regular conversion
        if not args.task_instruction_agent0 or not args.task_instruction_agent1:
            raise ValueError(
                "task_instruction_agent0 and task_instruction_agent1 are required when not using phase mode"
            )

        convert_robofactory_to_lerobot_combined(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            push_to_hub=args.push_to_hub,
        )
