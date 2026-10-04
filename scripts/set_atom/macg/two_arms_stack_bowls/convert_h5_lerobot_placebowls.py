import argparse
from pathlib import Path

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def convert_placebowls_to_lerobot_combined(
    h5_path: str,
    output_dir: str,
    task_name: str,
    task_instruction_agent0: str,
    task_instruction_agent1: str,
    push_to_hub: bool = False,
):
    """
    Convert Place Bowls H5 dataset to LeRobot format combining both agents' data.

    Args:
        h5_path: Path to the input H5 file
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'place_bowls')
        task_instruction_agent0: Natural language instruction for agent 0
        task_instruction_agent1: Natural language instruction for agent 1
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    with h5py.File(h5_path, "r") as f:
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
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)

                # Load global image
                global_image = f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:]  # (T, H, W, 3)

                # Get minimum length to ensure alignment
                T = min(len(actions_0), len(actions_1), len(image_0), len(image_1), len(global_image))

                # Combine task instructions with separator for precise tokenization boundary
                combined_instruction = f"{task_instruction_agent0} | {task_instruction_agent1}"

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
                tags=["placebowls", "panda", "combined_agents", task_name],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def convert_placebowls_to_lerobot_combined_with_phase(
    h5_path: str,
    output_dir: str,
    task_name: str,
    phase_instructions: dict = None,
    add_wait_data: bool = False,
    num_wait_episodes: int = 100,
    frames_per_wait_episode: int = 50,
    wait_phase: int = 0,
    push_to_hub: bool = False,
):
    """
    Convert Place Bowls H5 dataset with phase information to LeRobot format, combining both agents.
    Optionally add "wait" data where both agents stay still.

    Args:
        h5_path: Path to the input H5 file (merged file with phase field)
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'place_bowls')
        phase_instructions: Dictionary mapping phase values to instruction dictionaries
                          Example: {
                              1: {"agent0": "Grasp and lift the bowl", "agent1": "Wait for agent 0"},
                              2: {"agent0": "Place the bowl and return", "agent1": "Wait for agent 0"},
                              3: {"agent0": "Wait for agent 1", "agent1": "Grasp and lift the bowl"},
                              4: {"agent0": "Wait for agent 1", "agent1": "Place the bowl and return"}
                          }
        add_wait_data: Whether to add "wait" data (both agents stay still)
        num_wait_episodes: Number of wait episodes to generate
        frames_per_wait_episode: Number of consecutive frames per wait episode
        wait_phase: Phase value for wait data (default: 0)
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """

    def qpos_to_gripper_action(qpos_finger: float) -> float:
        """Convert qpos finger position to gripper action."""
        action = (qpos_finger - 0.02) / 0.02 * 2 - 1
        return np.clip(action, -1.0, 1.0)

    # Default phase instructions for place bowls task
    if phase_instructions is None:
        phase_instructions = {
            1: {"agent0": "Pick up the bowl.", "agent1": "Wait for the other robots."},
            2: {"agent0": "Place the bowl on the plate on the left.", "agent1": "Wait for the other robots."},
            3: {"agent0": "Recover.", "agent1": "Pick up the bowl."},
            4: {"agent0": "Wait for the other robots.", "agent1": "Place the bowl on the plate on the right."},
            5: {"agent0": "Wait for the other robots.", "agent1": "Recover."},
        }

    with h5py.File(h5_path, "r") as f:
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

        print("\nProcessing original trajectory data (phases 1-5)...")
        original_episode_count = 0

        for traj_key in traj_keys:
            try:
                # Load data for agent 0
                actions_0 = f[f"{traj_key}/actions/panda-0"][:]  # (T, 8)
                image_0 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:]  # (T, H, W, 3)

                # Load data for agent 1
                actions_1 = f[f"{traj_key}/actions/panda-1"][:]  # (T, 8)
                image_1 = f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:]  # (T, H, W, 3)

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
                        # Use | as separator for precise tokenization boundary
                        combined_instruction = f"{instruction_0} | {instruction_1}"
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
                original_episode_count += 1

            except Exception as e:
                print(f"[Warning] Failed to process {traj_key}: {e}")

        print(f"Processed {original_episode_count} original episodes")

        # Add wait data if requested
        if add_wait_data:
            print(f"\nGenerating wait data (phase={wait_phase})...")
            print(f"Will generate {num_wait_episodes} episodes with {frames_per_wait_episode} CONSECUTIVE frames each")

            # Wait instruction for both agents
            wait_instruction = "Wait for the other robots. | Wait for the other robots."

            np.random.seed(42)  # For reproducibility

            # Pre-load all trajectory data for efficient sampling
            all_traj_data = []
            print("Loading trajectory data for wait generation...")
            for traj_key in traj_keys:
                try:
                    traj_data = {
                        "qpos_0": f[f"{traj_key}/obs/agent/panda-0/qpos"][:],
                        "qpos_1": f[f"{traj_key}/obs/agent/panda-1/qpos"][:],
                        "image_0": f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:],
                        "image_1": f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:],
                        "global_image": f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:],
                    }
                    T = min(
                        len(traj_data["qpos_0"]),
                        len(traj_data["qpos_1"]),
                        len(traj_data["image_0"]),
                        len(traj_data["image_1"]),
                        len(traj_data["global_image"]),
                    )
                    traj_data["T"] = T
                    # Only include trajectories long enough for consecutive sampling
                    if T >= frames_per_wait_episode:
                        all_traj_data.append(traj_data)
                except Exception as e:
                    print(f"[Warning] Failed to load {traj_key} for wait data: {e}")

            print(f"Loaded {len(all_traj_data)} trajectories for wait generation (with T >= {frames_per_wait_episode})")

            if len(all_traj_data) == 0:
                print(f"[Warning] No trajectories have enough frames (>= {frames_per_wait_episode}), skipping wait data")
            else:
                wait_episode_count = 0
                for ep_idx in range(num_wait_episodes):
                    try:
                        # Randomly select a trajectory
                        traj_idx = np.random.randint(0, len(all_traj_data))
                        traj_data = all_traj_data[traj_idx]

                        # Randomly select a starting frame (ensure we have enough consecutive frames)
                        max_start = traj_data["T"] - frames_per_wait_episode
                        start_idx = np.random.randint(0, max_start + 1)

                        # For each CONSECUTIVE frame in this segment
                        for t in range(frames_per_wait_episode):
                            frame_idx = start_idx + t

                            # Get current state
                            state_0 = traj_data["qpos_0"][frame_idx].astype(np.float32)
                            state_1 = traj_data["qpos_1"][frame_idx].astype(np.float32)

                            # For "wait" action: action = current state (robot stays still)
                            action_7d_0 = state_0[:7]
                            action_1d_0 = qpos_to_gripper_action(state_0[7])

                            action_7d_1 = state_1[:7]
                            action_1d_1 = qpos_to_gripper_action(state_1[7])

                            # Create 8D actions for each agent
                            full_action_0 = np.concatenate([action_7d_0, np.array([action_1d_0])]).astype(np.float32)
                            full_action_1 = np.concatenate([action_7d_1, np.array([action_1d_1])]).astype(np.float32)

                            # Combine actions into 16-dimensional vector
                            combined_actions = np.concatenate([full_action_0, full_action_1])

                            dataset.add_frame(
                                {
                                    "image_agent0": traj_data["image_0"][frame_idx],
                                    "image_agent1": traj_data["image_1"][frame_idx],
                                    "global_image": traj_data["global_image"][frame_idx],
                                    "joint_position_agent0": action_7d_0,
                                    "joint_position_agent1": action_7d_1,
                                    "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                                    "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                                    "actions": combined_actions,
                                    "phase": np.array([wait_phase], dtype=np.int32),
                                    "task": wait_instruction,
                                }
                            )

                        dataset.save_episode()
                        wait_episode_count += 1

                        if (ep_idx + 1) % 10 == 0:
                            print(f"  Generated {ep_idx + 1}/{num_wait_episodes} wait episodes")

                    except Exception as e:
                        print(f"[Warning] Failed to process wait episode {ep_idx}: {e}")

                print(f"Generated {wait_episode_count} wait episodes")
                print(f"\nTotal: {original_episode_count} original + {wait_episode_count} wait = {original_episode_count + wait_episode_count} episodes")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["placebowls", "panda", "combined_agents", task_name, "multi-phase"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


def generate_wait_data_to_lerobot(
    h5_path: str,
    output_dir: str,
    task_name: str,
    num_episodes: int = 100,
    frames_per_episode: int = 50,
    wait_phase: int = 0,
    push_to_hub: bool = False,
):
    """
    Generate "wait" data where both agents have the instruction "Wait for the other robots."
    The action equals the current state, meaning the robot stays still.
    Compatible with use_phase mode - includes phase field.

    IMPORTANT: Wait data must use CONSECUTIVE frames from the same trajectory!
    - This is critical for models that use action_horizon (action chunk prediction)
    - Each episode samples a contiguous segment from one trajectory
    - Action at each timestep = current state (robot stays still)

    Args:
        h5_path: Path to the input H5 file (source of images and states)
        output_dir: Output directory for the converted dataset
        task_name: Name of the task (e.g., 'wait_data')
        num_episodes: Number of episodes to generate
        frames_per_episode: Number of CONSECUTIVE frames per episode
        wait_phase: Phase value for wait data (default: 0, to distinguish from phases 1-5)
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """

    def qpos_to_gripper_action(qpos_finger: float) -> float:
        """Convert qpos finger position to gripper action.
        qpos ≈ 0.04 (open) -> action = 1.0
        qpos ≈ 0.02 (closed) -> action = -1.0
        """
        # Linear mapping: action = (qpos - 0.02) / 0.02 * 2 - 1
        action = (qpos_finger - 0.02) / 0.02 * 2 - 1
        # Clamp to [-1, 1]
        return np.clip(action, -1.0, 1.0)

    with h5py.File(h5_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]

        # Create a standardized dataset name
        repo_id = f"{task_name}_combined"
        output_path = Path(output_dir) / repo_id

        print(f"\nCreating wait dataset for task '{task_name}'...")
        print(f"Will generate {num_episodes} episodes with {frames_per_episode} CONSECUTIVE frames each")

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

        # Wait instruction for both agents
        wait_instruction = "Wait for the other robots. | Wait for the other robots."

        print("\nGenerating wait data with CONSECUTIVE frame sampling...")

        np.random.seed(42)  # For reproducibility

        # Pre-load all trajectory data for efficient sampling
        all_traj_data = []
        print("Loading trajectory data...")
        for traj_key in traj_keys:
            try:
                traj_data = {
                    "qpos_0": f[f"{traj_key}/obs/agent/panda-0/qpos"][:],
                    "qpos_1": f[f"{traj_key}/obs/agent/panda-1/qpos"][:],
                    "image_0": f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"][:],
                    "image_1": f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"][:],
                    "global_image": f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"][:],
                }
                T = min(
                    len(traj_data["qpos_0"]),
                    len(traj_data["qpos_1"]),
                    len(traj_data["image_0"]),
                    len(traj_data["image_1"]),
                    len(traj_data["global_image"]),
                )
                traj_data["T"] = T
                # Only include trajectories that are long enough
                if T >= frames_per_episode:
                    all_traj_data.append(traj_data)
            except Exception as e:
                print(f"[Warning] Failed to load {traj_key}: {e}")

        print(f"Loaded {len(all_traj_data)} trajectories (with T >= {frames_per_episode})")

        if len(all_traj_data) == 0:
            raise ValueError(f"No trajectories have enough frames (>= {frames_per_episode})")

        for ep_idx in range(num_episodes):
            try:
                # Randomly select a trajectory
                traj_idx = np.random.randint(0, len(all_traj_data))
                traj_data = all_traj_data[traj_idx]

                # Randomly select a starting frame (ensure we have enough consecutive frames)
                max_start = traj_data["T"] - frames_per_episode
                start_idx = np.random.randint(0, max_start + 1)

                # For each CONSECUTIVE frame in this segment
                for t in range(frames_per_episode):
                    frame_idx = start_idx + t

                    # Get current state
                    state_0 = traj_data["qpos_0"][frame_idx].astype(np.float32)
                    state_1 = traj_data["qpos_1"][frame_idx].astype(np.float32)

                    # For "wait" action: action = current state (robot stays still)
                    # This means the target position is the current position
                    action_7d_0 = state_0[:7]
                    action_1d_0 = qpos_to_gripper_action(state_0[7])

                    action_7d_1 = state_1[:7]
                    action_1d_1 = qpos_to_gripper_action(state_1[7])

                    # Create 8D actions for each agent
                    full_action_0 = np.concatenate([action_7d_0, np.array([action_1d_0])]).astype(np.float32)
                    full_action_1 = np.concatenate([action_7d_1, np.array([action_1d_1])]).astype(np.float32)

                    # Combine actions into 16-dimensional vector
                    combined_actions = np.concatenate([full_action_0, full_action_1])

                    dataset.add_frame(
                        {
                            "image_agent0": traj_data["image_0"][frame_idx],
                            "image_agent1": traj_data["image_1"][frame_idx],
                            "global_image": traj_data["global_image"][frame_idx],
                            "joint_position_agent0": action_7d_0,
                            "joint_position_agent1": action_7d_1,
                            "gripper_position_agent0": np.array([action_1d_0], dtype=np.float32),
                            "gripper_position_agent1": np.array([action_1d_1], dtype=np.float32),
                            "actions": combined_actions,
                            "phase": np.array([wait_phase], dtype=np.int32),
                            "task": wait_instruction,
                        }
                    )

                dataset.save_episode()

                if (ep_idx + 1) % 10 == 0:
                    print(f"  Generated {ep_idx + 1}/{num_episodes} episodes")

            except Exception as e:
                print(f"[Warning] Failed to process episode {ep_idx}: {e}")

        print(f"\nGenerated {num_episodes} wait episodes successfully!")
        print(f"Total frames: {num_episodes * frames_per_episode}")

        if push_to_hub:
            dataset.push_to_hub(
                tags=["wait_data", "panda", "combined_agents", task_name, "multi-phase"],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Place Bowls H5 dataset to LeRobot format (Combined Agents)")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the converted dataset")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (e.g., place_bowls)")
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
    parser.add_argument(
        "--add_wait_data",
        action="store_true",
        default=False,
        help="Add wait data (both agents stay still) to the dataset when using --use_phase",
    )
    parser.add_argument(
        "--generate_wait",
        action="store_true",
        default=False,
        help="Generate wait data ONLY (separate dataset) where both agents stay still",
    )
    parser.add_argument(
        "--num_wait_episodes",
        type=int,
        default=100,
        help="Number of wait episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--frames_per_wait_episode",
        type=int,
        default=50,
        help="Number of randomly sampled frames per wait episode (default: 50)",
    )
    parser.add_argument(
        "--wait_phase",
        type=int,
        default=0,
        help="Phase value for wait data (default: 0, to distinguish from phases 1-5)",
    )

    args = parser.parse_args()

    if args.generate_wait:
        # Generate wait data ONLY (separate dataset)
        generate_wait_data_to_lerobot(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            num_episodes=args.num_wait_episodes,
            frames_per_episode=args.frames_per_wait_episode,
            wait_phase=args.wait_phase,
            push_to_hub=args.push_to_hub,
        )
    elif args.use_phase:
        # Use phase-based conversion, optionally with wait data
        convert_placebowls_to_lerobot_combined_with_phase(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            add_wait_data=args.add_wait_data,
            num_wait_episodes=args.num_wait_episodes,
            frames_per_wait_episode=args.frames_per_wait_episode,
            wait_phase=args.wait_phase,
            push_to_hub=args.push_to_hub,
        )
    else:
        # Use regular conversion
        if not args.task_instruction_agent0 or not args.task_instruction_agent1:
            raise ValueError(
                "task_instruction_agent0 and task_instruction_agent1 are required when not using phase mode"
            )

        convert_placebowls_to_lerobot_combined(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            task_name=args.task_name,
            task_instruction_agent0=args.task_instruction_agent0,
            task_instruction_agent1=args.task_instruction_agent1,
            push_to_hub=args.push_to_hub,
        )




