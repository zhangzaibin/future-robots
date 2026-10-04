import argparse

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

"""
bash
uv run python scripts/tasks/convert_h5_lerobot_placemeat.py \
    --h5_path /18141169908/wyh/multivla-pi0/dataset/colla_vla/place_meat_order.h5 \
    --output_dir /18141169908/wyh/dataset/lerobot/place_meat_order \
    --task_name place_meat_order \
    --task_instruction_agent0 "Pick up meat pieces and place them in the correct order on the target location" \
    --task_instruction_agent1 "Assist agent 0 and coordinate placement sequence"

if has phase
uv run python scripts/tasks/convert_h5_lerobot_placemeat.py \
    --h5_path /path/to/your.h5 \
    --output_dir /output/dir \
    --task_name task_name \
    --use_phase
"""


def convert_robofactory_to_lerobot_combined_2agents(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    push_to_hub: bool = False,
):
    """
    Convert RoboFactory H5 dataset to LeRobot format combining TWO agents (without phase).

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'place_meat_colla')
        task_instruction_agent0: Natural language instruction for agent 0
        task_instruction_agent1: Natural language instruction for agent 1
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Create a standardized dataset name
        repo_id = f"{task_name}_combined_2agents"

        print(f"\nCreating combined dataset for task '{task_name}' with 2 agents...")
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
                    "shape": (16,),  # 2 agents * 8 dimensions
                    "names": ["actions"],
                },
            },
            image_writer_threads=4,
        )

        # Task instruction
        combined_instruction = f"agent0: {task_instruction_agent0} agent1: {task_instruction_agent1}"

        print("\nProcessing data for two agents...")
        successful_count = 0
        failed_count = 0

        for traj_key in traj_keys:
            try:
                # Validate trajectory structure first
                required_paths = [
                    f"{traj_key}/actions/panda-0",
                    f"{traj_key}/actions/panda-1",
                    f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb",
                    f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb",
                    f"{traj_key}/obs/sensor_data/head_camera_global/rgb",
                ]
                
                missing_paths = []
                for path in required_paths:
                    if path not in f:
                        missing_paths.append(path)
                
                if missing_paths:
                    print(f"[Warning] Skipping {traj_key}: Missing paths: {missing_paths}")
                    failed_count += 1
                    continue
                
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)

                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image))
                
                if T == 0:
                    print(f"[Warning] Skipping {traj_key}: No valid frames (T=0)")
                    failed_count += 1
                    continue

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
                successful_count += 1

            except KeyError as e:
                print(f"[Warning] Skipping {traj_key}: Missing key {e}")
                failed_count += 1
            except Exception as e:
                print(f"[Warning] Skipping {traj_key}: {type(e).__name__}: {e}")
                failed_count += 1

        print("\nConversion completed!")
        print(f"  ✓ Successfully processed: {successful_count} trajectories")
        print(f"  ✗ Failed: {failed_count} trajectories")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name, "two_agents"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_robofactory_to_lerobot_combined_with_phase_2agents(
    h5_path: str, output_dir: str, task_name: str, phase_instructions: dict = None, push_to_hub: bool = False
):
    """
    Convert RoboFactory H5 dataset with phase information to LeRobot format, combining TWO agents.

    Args:
        h5_path: Path to the input H5 file (merged file with phase field)
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'place_meat_colla')
        phase_instructions: Dictionary mapping phase values to instruction dictionaries
                          Example: {
                              1: {"agent0": "Pick up the meat.", "agent1": "Wait for agent0."},
                              2: {"agent0": "Place the meat.", "agent1": "Assist with placement."}
                          }
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """

    # Default phase instructions for place_meat task
    if phase_instructions is None:
        phase_instructions = {
            1: {"agent0": "Pick up the meat from the source location.", "agent1": "Wait and observe."},
            2: {"agent0": "Move towards the target location.", "agent1": "Prepare to assist."},
            3: {"agent0": "Place the meat on the target.", "agent1": "Help stabilize the placement."},
        }

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Create a standardized dataset name
        repo_id = f"{task_name}_combined_2agents"

        print(f"\nCreating combined dataset for task '{task_name}' with 2 agents...")
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
                    "shape": (16,),  # 2 agents * 8 dimensions (7 joints + 1 gripper)
                    "names": ["actions"],
                },
                "phase": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": None,
                },
            },
            image_writer_threads=4,
        )

        print("\nProcessing data for two agents...")
        successful_count = 0
        failed_count = 0

        for traj_key in traj_keys:
            try:
                # Validate trajectory structure first
                required_paths = [
                    f"{traj_key}/actions/panda-0",
                    f"{traj_key}/actions/panda-1",
                    f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb",
                    f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb",
                    f"{traj_key}/obs/sensor_data/head_camera_global/rgb",
                    f"{traj_key}/phase",
                ]
                
                missing_paths = []
                for path in required_paths:
                    if path not in f:
                        missing_paths.append(path)
                
                if missing_paths:
                    print(f"[Warning] Skipping {traj_key}: Missing paths: {missing_paths}")
                    failed_count += 1
                    continue
                
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)

                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                # Load phase information
                phase = f[f"{traj_key}/phase"][:]  # (T,)

                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image), len(phase))
                
                if T == 0:
                    print(f"[Warning] Skipping {traj_key}: No valid frames (T=0)")
                    failed_count += 1
                    continue

                for t in range(T):
                    # Process agent 0 data
                    full_state_0 = actions_0[t].astype(np.float32)
                    action_7d_0 = full_state_0[:7]
                    action_1d_0 = full_state_0[7]

                    # Process agent 1 data
                    full_state_1 = actions_1[t].astype(np.float32)
                    action_7d_1 = full_state_1[:7]
                    action_1d_1 = full_state_1[7]

                    # Combine actions into 16-dimensional vector (2 agents * 8)
                    combined_actions = np.concatenate([full_state_0, full_state_1])

                    # Get phase-specific instruction
                    current_phase = int(phase[t])
                    if current_phase in phase_instructions:
                        instruction_0 = phase_instructions[current_phase]["agent0"]
                        instruction_1 = phase_instructions[current_phase]["agent1"]
                    else:
                        instruction_0 = "Complete the task"
                        instruction_1 = "Complete the task"

                    combined_instruction = f"agent0: {instruction_0} agent1: {instruction_1}"

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
                            "phase": np.array([current_phase], dtype=np.int64),
                            "task": combined_instruction,
                        }
                    )
                dataset.save_episode()
                successful_count += 1

            except KeyError as e:
                print(f"[Warning] Skipping {traj_key}: Missing key {e}")
                failed_count += 1
            except Exception as e:
                print(f"[Warning] Skipping {traj_key}: {type(e).__name__}: {e}")
                failed_count += 1

        print("\nConversion completed!")
        print(f"  ✓ Successfully processed: {successful_count} trajectories")
        print(f"  ✗ Failed: {failed_count} trajectories")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name, "two_agents"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboFactory H5 dataset to LeRobot format (2 agents)")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted dataset")
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name of the task (e.g., place_meat_colla, place_meat_order)"
    )
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
        default="Put meat in the basket.",
    )
    parser.add_argument(
        "--task_instruction_agent1",
        type=str,
        help="Natural language instruction for agent 1 (required if not using phase)",
        default="",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push the dataset to Hugging Face Hub"
    )

    args = parser.parse_args()

    if args.use_phase:
        # Use phase-based conversion
        convert_robofactory_to_lerobot_combined_with_phase_2agents(
            h5_path=args.h5_path, output_dir=args.output_dir, task_name=args.task_name, push_to_hub=args.push_to_hub
        )
    else:
        # Use regular conversion (without phase)
        # if not args.task_instruction_agent0 or not args.task_instruction_agent1:
        #     raise ValueError(
        #         "task_instruction_agent0 and task_instruction_agent1 are required when not using phase mode"
        #     )

        convert_robofactory_to_lerobot_combined_2agents(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            push_to_hub=args.push_to_hub,
        )
