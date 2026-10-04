"""
Convert RoboFactory H5 dataset to LeRobot format with agent shuffling.

MODIFICATION NOTE: This script has been modified to only shuffle actions and instructions
while keeping images in their original order. This allows for data augmentation through
action/instruction permutation without changing the visual correspondence.
"""

import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def get_all_agent_permutations():
    """
    Get all possible permutations of 3 agents.
    Returns list of tuples representing different shuffling arrangements.
    """
    return list(itertools.permutations([0, 1, 2]))


def create_shuffle_mapping(permutation):
    """
    Create agent shuffle mapping from permutation.

    Args:
        permutation: tuple like (1, 2, 0) meaning agent0->agent1, agent1->agent2, agent2->agent0

    Returns:
        dict: mapping from original agent index to new agent index
    """
    return {i: permutation[i] for i in range(3)}


def shuffle_phase_instructions(phase_instructions, shuffle_mapping):
    """
    Shuffle phase instructions according to the agent mapping.

    Args:
        phase_instructions: original phase instruction dict
        shuffle_mapping: agent index mapping

    Returns:
        dict: shuffled phase instructions
    """
    shuffled_instructions = {}
    for phase, instructions in phase_instructions.items():
        shuffled_instructions[phase] = {}
        for original_agent, new_agent in shuffle_mapping.items():
            original_key = f"agent{original_agent}"
            new_key = f"agent{new_agent}"
            shuffled_instructions[phase][new_key] = instructions[original_key]
    return shuffled_instructions


def convert_robofactory_to_lerobot_combined(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    task_instruction_agent2: str,
    push_to_hub: bool = False,
    shuffle_mapping: dict = None,
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
        shuffle_mapping: Optional dict to shuffle agent assignments
    """

    # Apply shuffle to instructions if mapping provided
    original_instructions = [task_instruction_agent0, task_instruction_agent1, task_instruction_agent2]
    if shuffle_mapping:
        shuffled_instructions = [None] * 3
        for orig_idx, new_idx in shuffle_mapping.items():
            shuffled_instructions[new_idx] = original_instructions[orig_idx]
        task_instruction_agent0, task_instruction_agent1, task_instruction_agent2 = shuffled_instructions

        # Add shuffle info to repo_id
        shuffle_str = "_".join([str(shuffle_mapping[i]) for i in range(3)])
        repo_id = f"{task_name}_combined_shuffle_{shuffle_str}"
    else:
        repo_id = f"{task_name}_combined"

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        output_path = Path(output_dir) / repo_id

        print(f"\nCreating combined dataset for task '{task_name}'...")
        if shuffle_mapping:
            print(f"Using shuffle mapping: {shuffle_mapping}")

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
                # Load original data for all agents
                original_actions = []
                original_images = []

                for agent_idx in range(3):
                    actions = f[f"{traj_key}/actions/panda-{agent_idx}"][:]  # (T, 8)
                    image = f[f"{traj_key}/obs/sensor_data/head_camera_agent{agent_idx}/rgb"][:]  # (T, H, W, 3)
                    original_actions.append(actions)
                    original_images.append(image)

                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                # Apply shuffle mapping only to actions (not images)
                if shuffle_mapping:
                    shuffled_actions = [None] * 3
                    for orig_idx, new_idx in shuffle_mapping.items():
                        shuffled_actions[new_idx] = original_actions[orig_idx]
                    actions_0, actions_1, actions_2 = shuffled_actions
                    # Images remain in original order
                    image_0, image_1, image_2 = original_images
                else:
                    actions_0, actions_1, actions_2 = original_actions
                    image_0, image_1, image_2 = original_images

                # Get minimum length to ensure alignment
                T = min(
                    len(actions_0),
                    len(actions_1),
                    len(actions_2),
                    len(image_0),
                    len(image_1),
                    len(image_2),
                    len(global_image),
                )

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

                    dataset.add_frame(
                        {
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
                            "task": combined_instruction,
                        }
                    )
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            tags = ["robofactory", "panda", "combined_agents", task_name, "three_agents"]
            if shuffle_mapping:
                tags.append("shuffled")
            dataset.push_to_hub(
                tags=tags,
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_robofactory_to_lerobot_combined_with_phase(
    h5_path: str,
    output_dir: str,
    task_name: str,
    phase_instructions: dict = None,
    push_to_hub: bool = False,
    shuffle_mapping: dict = None,
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
        shuffle_mapping: Optional dict to shuffle agent assignments
    """

    # Default phase instructions for stack cubes task
    if phase_instructions is None:
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait.",
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.",
                "agent2": "Wait for the other robots.",
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target.",
            },
        }

    # Apply shuffle to phase instructions if mapping provided
    if shuffle_mapping:
        phase_instructions = shuffle_phase_instructions(phase_instructions, shuffle_mapping)
        shuffle_str = "_".join([str(shuffle_mapping[i]) for i in range(3)])
        repo_id = f"{task_name}_combined_shuffle_{shuffle_str}"
    else:
        repo_id = f"{task_name}_combined"

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        output_path = Path(output_dir) / repo_id

        print(f"\nCreating combined dataset for task '{task_name}'...")
        if shuffle_mapping:
            print(f"Using shuffle mapping: {shuffle_mapping}")

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
                # Load original data for all agents
                original_actions = []
                original_images = []

                for agent_idx in range(3):
                    actions = f[f"{traj_key}/actions/panda-{agent_idx}"][:]  # (T, 8)
                    image = f[f"{traj_key}/obs/sensor_data/head_camera_agent{agent_idx}/rgb"][:]  # (T, H, W, 3)
                    original_actions.append(actions)
                    original_images.append(image)

                # Load global image and phase data
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)
                phase_data = f[f"{traj_key}/phase"][:]  # (T,)

                # Apply shuffle mapping only to actions (not images)
                if shuffle_mapping:
                    shuffled_actions = [None] * 3
                    for orig_idx, new_idx in shuffle_mapping.items():
                        shuffled_actions[new_idx] = original_actions[orig_idx]
                    actions_0, actions_1, actions_2 = shuffled_actions
                    # Images remain in original order
                    image_0, image_1, image_2 = original_images
                else:
                    actions_0, actions_1, actions_2 = original_actions
                    image_0, image_1, image_2 = original_images

                # Get minimum length to ensure alignment
                T = min(
                    len(actions_0),
                    len(actions_1),
                    len(actions_2),
                    len(image_0),
                    len(image_1),
                    len(image_2),
                    len(global_image),
                    len(phase_data),
                )

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
                        combined_instruction = (
                            f"agent0: {instruction_0} agent1: {instruction_1} agent2: {instruction_2}"
                        )
                    else:
                        raise ValueError(f"Phase {current_phase} not found in phase_instructions")

                    dataset.add_frame(
                        {
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
                            "task": combined_instruction,
                        }
                    )
                dataset.save_episode()

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        if push_to_hub:
            tags = ["robofactory", "panda", "combined_agents", task_name, "multi-phase", "three_agents"]
            if shuffle_mapping:
                tags.append("shuffled")
            dataset.push_to_hub(
                tags=tags,
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_with_all_shuffles_combined(
    h5_path: str,
    output_dir: str,
    task_name: str,
    use_phase: bool = False,
    task_instruction_agent0: str = None,
    task_instruction_agent1: str = None,
    task_instruction_agent2: str = None,
    phase_instructions: dict = None,
    push_to_hub: bool = False,
    include_original: bool = True,
):
    """
    Convert dataset with all possible agent shuffles into a single combined dataset.

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted dataset
        task_name: Name of the task
        use_phase: Whether to use phase-based conversion
        task_instruction_agent0/1/2: Instructions for non-phase mode
        phase_instructions: Phase instruction dict for phase mode
        push_to_hub: Whether to push to hub
        include_original: Whether to include the original (non-shuffled) version
    """

    all_permutations = get_all_agent_permutations()

    # Set default phase instructions if using phase mode and none provided
    if use_phase and phase_instructions is None:
        phase_instructions = {
            1: {
                "agent0": "Lift the object and place it on the target.",
                "agent1": "Lift the object and wait.",
                "agent2": "Lift the object and wait.",
            },
            2: {
                "agent0": "Return and wait for the other robots.",
                "agent1": "Place the object on the target.",
                "agent2": "Wait for the other robots.",
            },
            3: {
                "agent0": "Wait for the other robots.",
                "agent1": "Return and wait for the other robots.",
                "agent2": "Place the object on the target.",
            },
        }

    # Create single combined dataset
    repo_id = f"{task_name}_combined_shuffled"
    output_path = Path(output_dir) / repo_id

    print(f"\nCreating single combined dataset with all shuffles for task '{task_name}'...")

    # Build features dict based on whether using phase mode
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
    }

    # Add phase feature if using phase mode
    if use_phase:
        features["phase"] = {
            "dtype": "int32",
            "shape": (1,),
            "names": ["phase"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=30,
        features=features,
        image_writer_threads=4,
    )

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Determine which shuffles to include
        shuffles_to_process = []
        if include_original:
            shuffles_to_process.append((None, "original"))

        start_idx = 1 if include_original else 0
        for i, permutation in enumerate(all_permutations[start_idx:], start_idx):
            shuffle_mapping = create_shuffle_mapping(permutation)
            shuffles_to_process.append((shuffle_mapping, f"shuffle_{permutation}"))

        print(f"Processing {len(shuffles_to_process)} shuffle variations...")

        for shuffle_idx, (shuffle_mapping, shuffle_name) in enumerate(shuffles_to_process):
            print(f"\n=== Processing {shuffle_name} ({shuffle_idx+1}/{len(shuffles_to_process)}) ===")
            if shuffle_mapping:
                print(f"Shuffle mapping: {shuffle_mapping}")

            # Get appropriate instructions for this shuffle
            if use_phase:
                if phase_instructions is None:
                    raise ValueError("phase_instructions cannot be None when use_phase=True")
                current_phase_instructions = phase_instructions
                if shuffle_mapping:
                    current_phase_instructions = shuffle_phase_instructions(phase_instructions, shuffle_mapping)
            else:
                if any(
                    instr is None
                    for instr in [task_instruction_agent0, task_instruction_agent1, task_instruction_agent2]
                ):
                    raise ValueError(
                        "task_instruction_agent0, task_instruction_agent1, and task_instruction_agent2 cannot be None when use_phase=False"
                    )
                original_instructions = [task_instruction_agent0, task_instruction_agent1, task_instruction_agent2]
                if shuffle_mapping:
                    shuffled_instructions = [None] * 3
                    for orig_idx, new_idx in shuffle_mapping.items():
                        shuffled_instructions[new_idx] = original_instructions[orig_idx]
                    (
                        current_task_instruction_agent0,
                        current_task_instruction_agent1,
                        current_task_instruction_agent2,
                    ) = shuffled_instructions
                else:
                    (
                        current_task_instruction_agent0,
                        current_task_instruction_agent1,
                        current_task_instruction_agent2,
                    ) = original_instructions

            # Process each trajectory with current shuffle
            for traj_key in traj_keys:
                try:
                    # Load original data for all agents
                    original_actions = []
                    original_images = []

                    for agent_idx in range(3):
                        actions = f[f"{traj_key}/actions/panda-{agent_idx}"][:]  # (T, 8)
                        image = f[f"{traj_key}/obs/sensor_data/head_camera_agent{agent_idx}/rgb"][:]  # (T, H, W, 3)
                        original_actions.append(actions)
                        original_images.append(image)

                    # Load global image
                    global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                    # Load phase data if using phase mode
                    if use_phase:
                        phase_data = f[f"{traj_key}/phase"][:]  # (T,)

                    # Apply shuffle mapping only to actions (not images)
                    if shuffle_mapping:
                        shuffled_actions = [None] * 3
                        for orig_idx, new_idx in shuffle_mapping.items():
                            shuffled_actions[new_idx] = original_actions[orig_idx]
                        actions_0, actions_1, actions_2 = shuffled_actions
                        # Images remain in original order
                        image_0, image_1, image_2 = original_images
                    else:
                        actions_0, actions_1, actions_2 = original_actions
                        image_0, image_1, image_2 = original_images

                    # Get minimum length to ensure alignment
                    if use_phase:
                        T = min(
                            len(actions_0),
                            len(actions_1),
                            len(actions_2),
                            len(image_0),
                            len(image_1),
                            len(image_2),
                            len(global_image),
                            len(phase_data),
                        )
                    else:
                        T = min(
                            len(actions_0),
                            len(actions_1),
                            len(actions_2),
                            len(image_0),
                            len(image_1),
                            len(image_2),
                            len(global_image),
                        )

                    for t in range(T):
                        # Process agent data
                        full_state_0 = actions_0[t].astype(np.float32)
                        action_7d_0 = full_state_0[:7]
                        action_1d_0 = full_state_0[7]

                        full_state_1 = actions_1[t].astype(np.float32)
                        action_7d_1 = full_state_1[:7]
                        action_1d_1 = full_state_1[7]

                        full_state_2 = actions_2[t].astype(np.float32)
                        action_7d_2 = full_state_2[:7]
                        action_1d_2 = full_state_2[7]

                        # Combine actions into 24-dimensional vector
                        combined_actions = np.concatenate([full_state_0, full_state_1, full_state_2])

                        # Prepare frame data
                        frame_data = {
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
                        }

                        # Add task instruction
                        if use_phase:
                            current_phase = int(phase_data[t])
                            if current_phase not in current_phase_instructions:
                                print(
                                    f"[Warning] Phase {current_phase} not found in phase_instructions for {traj_key} at time {t}, skipping frame"
                                )
                                continue
                            try:
                                instruction_0 = current_phase_instructions[current_phase]["agent0"]
                                instruction_1 = current_phase_instructions[current_phase]["agent1"]
                                instruction_2 = current_phase_instructions[current_phase]["agent2"]
                                combined_instruction = (
                                    f"agent0: {instruction_0} agent1: {instruction_1} agent2: {instruction_2}"
                                )
                            except KeyError as e:
                                print(
                                    f"[Warning] Missing agent instruction in phase {current_phase}: {e}, skipping frame"
                                )
                                continue
                            frame_data["phase"] = np.array([current_phase], dtype=np.int32)
                        else:
                            combined_instruction = f"{current_task_instruction_agent0} {current_task_instruction_agent1} {current_task_instruction_agent2}"

                        frame_data["task"] = combined_instruction
                        dataset.add_frame(frame_data)

                    dataset.save_episode()

                except Exception as e:
                    print(f"[Warning] Failed to process {traj_key} for {shuffle_name}: {e}")

    if push_to_hub:
        tags = ["robofactory", "panda", "combined_agents", task_name, "shuffled", "augmented", "three_agents"]
        if use_phase:
            tags.append("multi-phase")
        dataset.push_to_hub(
            tags=tags,
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"\n=== Conversion complete! Combined dataset with {len(shuffles_to_process)} shuffle variations created ===")


def convert_with_all_shuffles(
    h5_path: str,
    output_dir: str,
    task_name: str,
    use_phase: bool = False,
    task_instruction_agent0: str = None,
    task_instruction_agent1: str = None,
    task_instruction_agent2: str = None,
    phase_instructions: dict = None,
    push_to_hub: bool = False,
    include_original: bool = True,
):
    """
    Convert dataset with all possible agent shuffles to multiply the dataset size.

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted datasets
        task_name: Name of the task
        use_phase: Whether to use phase-based conversion
        task_instruction_agent0/1/2: Instructions for non-phase mode
        phase_instructions: Phase instruction dict for phase mode
        push_to_hub: Whether to push to hub
        include_original: Whether to include the original (non-shuffled) version
    """

    all_permutations = get_all_agent_permutations()

    # Include original version
    if include_original:
        print("=== Converting original (no shuffle) ===")
        if use_phase:
            convert_robofactory_to_lerobot_combined_with_phase(
                h5_path=h5_path,
                output_dir=output_dir,
                task_name=task_name,
                phase_instructions=phase_instructions,
                push_to_hub=push_to_hub,
                shuffle_mapping=None,
            )
        else:
            convert_robofactory_to_lerobot_combined(
                h5_path=h5_path,
                output_dir=output_dir,
                task_name=task_name,
                task_instruction_agent0=task_instruction_agent0,
                task_instruction_agent1=task_instruction_agent1,
                task_instruction_agent2=task_instruction_agent2,
                push_to_hub=push_to_hub,
                shuffle_mapping=None,
            )

    # Convert with all shuffles (skip identity permutation if original included)
    start_idx = 1 if include_original else 0
    for i, permutation in enumerate(all_permutations[start_idx:], start_idx):
        shuffle_mapping = create_shuffle_mapping(permutation)
        print(f"\n=== Converting shuffle {i+1}/{len(all_permutations)}: {permutation} ===")
        print(f"Shuffle mapping: {shuffle_mapping}")

        if use_phase:
            convert_robofactory_to_lerobot_combined_with_phase(
                h5_path=h5_path,
                output_dir=output_dir,
                task_name=task_name,
                phase_instructions=phase_instructions,
                push_to_hub=push_to_hub,
                shuffle_mapping=shuffle_mapping,
            )
        else:
            convert_robofactory_to_lerobot_combined(
                h5_path=h5_path,
                output_dir=output_dir,
                task_name=task_name,
                task_instruction_agent0=task_instruction_agent0,
                task_instruction_agent1=task_instruction_agent1,
                task_instruction_agent2=task_instruction_agent2,
                push_to_hub=push_to_hub,
                shuffle_mapping=shuffle_mapping,
            )

    print(
        f"\n=== Conversion complete! Generated {len(all_permutations)} versions (including original: {include_original}) ==="
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RoboFactory H5 dataset to LeRobot format (Three Agents - Stack Cubes with Shuffle)"
    )
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted dataset")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (e.g., stack_cubes)")
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
        "--task_instruction_agent2",
        type=str,
        help="Natural language instruction for agent 2 (required if not using phase)",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", default=False, help="Whether to push the dataset to Hugging Face Hub"
    )

    # Shuffle-related arguments
    parser.add_argument(
        "--enable_shuffle", action="store_true", default=False, help="Enable agent shuffling to expand dataset"
    )
    parser.add_argument(
        "--all_shuffles",
        action="store_true",
        default=False,
        help="Generate all possible shuffle combinations (6 total) as separate datasets",
    )
    parser.add_argument(
        "--all_shuffles_combined",
        action="store_true",
        default=False,
        help="Generate all possible shuffle combinations into a single combined dataset",
    )
    parser.add_argument(
        "--specific_shuffle",
        type=str,
        default=None,
        help="Specific shuffle mapping as comma-separated values (e.g., '1,2,0' for agent0->agent1, agent1->agent2, agent2->agent0)",
    )
    parser.add_argument(
        "--include_original",
        action="store_true",
        default=True,
        help="Include original (non-shuffled) version when using --all_shuffles or --all_shuffles_combined",
    )

    args = parser.parse_args()

    # Determine shuffle mapping
    shuffle_mapping = None
    if args.specific_shuffle:
        try:
            permutation = tuple(map(int, args.specific_shuffle.split(",")))
            if len(permutation) != 3 or set(permutation) != {0, 1, 2}:
                raise ValueError("Invalid permutation")
            shuffle_mapping = create_shuffle_mapping(permutation)
            print(f"Using specific shuffle mapping: {shuffle_mapping}")
        except Exception:
            raise ValueError(
                f"Invalid specific_shuffle format. Expected 'x,y,z' where x,y,z are permutation of 0,1,2. Got: {args.specific_shuffle}"
            )

    if args.all_shuffles:
        # Generate all shuffle combinations as separate datasets
        convert_with_all_shuffles(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            use_phase=args.use_phase,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            task_instruction_agent2=args.task_instruction_agent2,
            push_to_hub=args.push_to_hub,
            include_original=args.include_original,
        )
    elif args.all_shuffles_combined:
        # Generate all shuffle combinations into a single combined dataset
        convert_with_all_shuffles_combined(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            use_phase=args.use_phase,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            task_instruction_agent2=args.task_instruction_agent2,
            push_to_hub=args.push_to_hub,
            include_original=args.include_original,
        )
    # Single conversion (with or without shuffle)
    elif args.use_phase:
        # Use phase-based conversion
        convert_robofactory_to_lerobot_combined_with_phase(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            push_to_hub=args.push_to_hub,
            shuffle_mapping=shuffle_mapping,
        )
    else:
        # Use regular conversion
        if not args.task_instruction_agent0 or not args.task_instruction_agent1 or not args.task_instruction_agent2:
            raise ValueError(
                "task_instruction_agent0, task_instruction_agent1, and task_instruction_agent2 are required when not using phase mode"
            )

        convert_robofactory_to_lerobot_combined(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            task_instruction_agent2=args.task_instruction_agent2,
            push_to_hub=args.push_to_hub,
            shuffle_mapping=shuffle_mapping,
        )
