import h5py
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import shutil
import argparse


def convert_robofactory_to_lerobot_combined(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    task_instruction_agent2: str,
    push_to_hub: bool = False
):
    """
    Convert RoboFactory H5 dataset to LeRobot format combining three agents' data.
    
    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'stack_cubes')
        task_instruction_agent0: Natural language instruction for agent 0
        task_instruction_agent1: Natural language instruction for agent 1
        task_instruction_agent2: Natural language instruction for agent 2
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    with h5py.File(h5_path, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        
        # Create a standardized dataset name
        repo_id = f"{task_name}_combined"
        output_path = Path(output_dir) / repo_id

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
                "image_agent2": {
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
                "joint_position_agent2": {
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
                "gripper_position_agent2": {
                    "dtype": "float32",
                    "shape": (1,),  
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (24,),  # Combined actions from three agents (8 * 3 = 24)
                    "names": ["actions"],
                },
            },
            image_writer_threads=4,
        )

        print("\nProcessing data for three agents...")

        for traj_key in traj_keys:
            try:
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)
                
                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)
                
                # Load data for agent 2
                actions_2 = f[f"{traj_key}/actions/panda-2"][:]  # (T, 8)
                image_2 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent2/rgb"][:]  # (T, H, W, 3)
                
                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)
                
                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(actions_2), len(image_0), len(image_1), len(image_2), len(global_image))
                
                # Combine task instructions
                combined_instruction = f"agent0: {task_instruction_agent0} agent1: {task_instruction_agent1} agent2: {task_instruction_agent2}"
                
                for t in range(T):
                    # Process agent 0 data
                    full_state_0 = actions_0[t].astype(np.float32)
                    action_7d_0 = full_state_0[:7]
                    action_1d_0 = full_state_0[7]
                    
                    # Process agent 1 data
                    full_state_1 = actions_1[t].astype(np.float32)
                    action_7d_1 = full_state_1[:7]
                    action_1d_1 = full_state_1[7]
                    
                    # Process agent 2 data
                    full_state_2 = actions_2[t].astype(np.float32)
                    action_7d_2 = full_state_2[:7]
                    action_1d_2 = full_state_2[7]
                    
                    # Combine actions into 24-dimensional vector
                    combined_actions = np.concatenate([full_state_0, full_state_1, full_state_2])
                    
                    dataset.add_frame({
                        "image_agent0": image_0[t],
                        "image_agent1": image_1[t],
                        "image_agent2": image_2[t],
                        "global_image": global_image[t],
                        "joint_position_agent0": action_7d_0,
                        "joint_position_agent1": action_7d_1,
                        "joint_position_agent2": action_7d_2,
                        "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                        "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                        "gripper_position_agent2": np.array([action_1d_2], dtype=np.float32),
                        "actions": combined_actions,
                        "task": combined_instruction
                    })
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name, "three_agents"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_robofactory_to_lerobot_combined_with_phase(
    h5_path: str,
    output_dir: str,
    task_name: str,
    phase_instructions: dict = None,
    push_to_hub: bool = False
):
    """
    Convert RoboFactory H5 dataset with phase information to LeRobot format, combining three agents.
    
    Args:
        h5_path: Path to the input H5 file (merged file with phase field)
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'stack_cubes')
        phase_instructions: Dictionary mapping phase values to instruction dictionaries
                          Example: {
                              1: {"agent0": "Lift the object and place it on the target.", "agent1": "Lift the object and wait.", "agent2": "Lift the object and wait."},
                              2: {"agent0": "Return and wait for the other robots.", "agent1": "Place the object on the target.", "agent2": "Wait for the other robots."},
                              3: {"agent0": "Wait for the other robots.", "agent1": "Return and wait for the other robots.", "agent2": "Place the object on the target."}
                          }
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    
    # Default phase instructions for stack cubes task
    if phase_instructions is None:
        phase_instructions = {
            1: {
                "agent0": "Pick up the bottle.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
            2: {
                "agent0": "Place the bottle on the target.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Pick up the box.",
                "agent2": "Wait for the other robots."
            },
            4: {
                "agent0": "Wait for the other robots.",
                "agent1": "Place the box on the target.",
                "agent2": "Wait for the other robots."
            },
            5: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Pick up the food."
            },
            6: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Place the food on the target."
            },
            7: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
            11: {
                "agent0": "Pick up the bottle.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
            12: {
                "agent0": "Place the bottle on the target.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
            13: {
                "agent0": "Wait for the other robots.",
                "agent1": "Pick up the food.",
                "agent2": "Wait for the other robots."
            },
            14: {
                "agent0": "Wait for the other robots.",
                "agent1": "Place the food on the target.",
                "agent2": "Wait for the other robots."
            },
            15: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Pick up the box."
            },
            16: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Place the box on the target."
            },
            17: {
                "agent0": "Wait for the other robots.",
                "agent1": "Wait for the other robots.",
                "agent2": "Wait for the other robots."
            },
        }
    
    with h5py.File(h5_path, 'r') as f:
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
                "image_agent2": {
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
                "joint_position_agent2": {
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
                "gripper_position_agent2": {
                    "dtype": "float32",
                    "shape": (1,),  
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (24,),  # Combined actions from three agents (8 * 3 = 24)
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

        print("\nProcessing data for three agents...")

        for traj_key in traj_keys:
            try:
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)
                
                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)
                
                # Load data for agent 2
                actions_2 = f[f"{traj_key}/actions/panda-2"][:]  # (T, 8)
                image_2 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent2/rgb"][:]  # (T, H, W, 3)
                
                # Load global image and phase data
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)
                phase_data = f[f"{traj_key}/phase"][:]  # (T,)
                
                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(actions_2), len(image_0), len(image_1), len(image_2), len(global_image), len(phase_data))
                
                for t in range(T):
                    # Process agent 0 data
                    full_state_0 = actions_0[t].astype(np.float32)
                    action_7d_0 = full_state_0[:7]
                    action_1d_0 = full_state_0[7]
                    
                    # Process agent 1 data
                    full_state_1 = actions_1[t].astype(np.float32)
                    action_7d_1 = full_state_1[:7]
                    action_1d_1 = full_state_1[7]
                    
                    # Process agent 2 data
                    full_state_2 = actions_2[t].astype(np.float32)
                    action_7d_2 = full_state_2[:7]
                    action_1d_2 = full_state_2[7]
                    
                    # Combine actions into 24-dimensional vector
                    combined_actions = np.concatenate([full_state_0, full_state_1, full_state_2])
                    
                    # Get current phase and instructions
                    current_phase = int(phase_data[t])
                    if current_phase in phase_instructions:
                        instruction_0 = phase_instructions[current_phase]["agent0"]
                        instruction_1 = phase_instructions[current_phase]["agent1"]
                        instruction_2 = phase_instructions[current_phase]["agent2"]
                        combined_instruction = f"agent0: {instruction_0} agent1: {instruction_1} agent2: {instruction_2}"
                    else:
                        raise ValueError(f"Phase {current_phase} not found in phase_instructions")
                    
                    dataset.add_frame({
                        "image_agent0": image_0[t],
                        "image_agent1": image_1[t],
                        "image_agent2": image_2[t],
                        "global_image": global_image[t],
                        "joint_position_agent0": action_7d_0,
                        "joint_position_agent1": action_7d_1,
                        "joint_position_agent2": action_7d_2,
                        "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                        "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                        "gripper_position_agent2": np.array([action_1d_2], dtype=np.float32),
                        "actions": combined_actions,
                        "phase": np.array([current_phase], dtype=np.int32),
                        "task": combined_instruction
                    })
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["robofactory", "panda", "combined_agents", task_name, "multi-phase", "three_agents"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RoboFactory H5 dataset to LeRobot format (Three Agents - Stack Cubes)")
    parser.add_argument("--h5_path", type=str, required=True, 
                       help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the converted dataset")
    parser.add_argument("--task_name", type=str, required=True,
                       help="Name of the task (e.g., stack_cubes)")
    parser.add_argument("--use_phase", action="store_true", default=False,
                       help="Use phase-based instruction mapping (for merged files with phase field)")
    parser.add_argument("--task_instruction_agent0", type=str, 
                       help="Natural language instruction for agent 0 (required if not using phase)")
    parser.add_argument("--task_instruction_agent1", type=str,
                       help="Natural language instruction for agent 1 (required if not using phase)")
    parser.add_argument("--task_instruction_agent2", type=str,
                       help="Natural language instruction for agent 2 (required if not using phase)")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                       help="Whether to push the dataset to Hugging Face Hub")
    
    args = parser.parse_args()
    
    if args.use_phase:
        # Use phase-based conversion
        convert_robofactory_to_lerobot_combined_with_phase(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            push_to_hub=args.push_to_hub
        )
    else:
        # Use regular conversion
        if not args.task_instruction_agent0 or not args.task_instruction_agent1 or not args.task_instruction_agent2:
            raise ValueError("task_instruction_agent0, task_instruction_agent1, and task_instruction_agent2 are required when not using phase mode")
        
        convert_robofactory_to_lerobot_combined(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            task_instruction_agent2=args.task_instruction_agent2,
            push_to_hub=args.push_to_hub
        )

