import os

import cv2
import h5py
import numpy as np


def create_phase_video(
    rgb_data, phases, gripper_states, traj_key, camera_name, output_dir="phase_videos_therestackcube"
):
    """
    Create a video showing different phases with gripper state overlay

    Args:
        rgb_data: RGB frames from the camera
        phases: Array of phase labels for each frame
        gripper_states: Tuple of (agent0_gripper, agent1_gripper, agent2_gripper) binary states
        traj_key: Trajectory key for naming
        camera_name: Camera name for naming
        output_dir: Directory to save videos
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = len(rgb_data)
    height, width = rgb_data[0].shape[:2]

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, f"{traj_key}_{camera_name}_phases.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height + 120))  # Extra space for info

    agent0_gripper, agent1_gripper, agent2_gripper = gripper_states

    phase_colors = {
        1: (255, 200, 200),  # Light red - Phase 1
        2: (200, 255, 200),  # Light green - Phase 2
        3: (200, 200, 255),  # Light blue - Phase 3
        4: (255, 255, 200),  # Light yellow - Phase 4
        5: (255, 200, 255),  # Light purple - Phase 5
    }

    phase_names = {
        1: "Phase 1: Agent0 & Agent1 Grasping",
        2: "Phase 2: Agent0 & Agent1 Lifting",
        3: "Phase 3: Agent0 Placing (Agent2 Stop)",
        4: "Phase 4: Agent1 Placing (Agent0 Stop)",
        5: "Phase 5: All Complete (Agent1 Stop)",
    }

    for i in range(n_frames):
        frame = rgb_data[i].copy()

        # Create info panel
        info_panel = np.ones((120, width, 3), dtype=np.uint8) * 255

        # Current phase info
        current_phase = phases[i]
        phase_color = phase_colors.get(current_phase, (200, 200, 200))

        # Add phase background color
        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add text info to info panel
        cv2.putText(info_panel, f"Frame: {i}/{n_frames-1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(
            info_panel,
            f"{phase_names.get(current_phase, f'Phase {current_phase}')}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # Gripper states for all three agents
        agent0_grip_color = (0, 150, 0) if agent0_gripper[i] else (150, 0, 0)
        agent1_grip_color = (0, 150, 0) if agent1_gripper[i] else (150, 0, 0)
        agent2_grip_color = (0, 150, 0) if agent2_gripper[i] else (150, 0, 0)

        cv2.putText(
            info_panel,
            f"Agent0 Gripper: {'CLOSED' if agent0_gripper[i] else 'OPEN'}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent0_grip_color,
            1,
        )
        cv2.putText(
            info_panel,
            f"Agent1 Gripper: {'CLOSED' if agent1_gripper[i] else 'OPEN'}",
            (230, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent1_grip_color,
            1,
        )
        cv2.putText(
            info_panel,
            f"Agent2 Gripper: {'CLOSED' if agent2_gripper[i] else 'OPEN'}",
            (450, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent2_grip_color,
            1,
        )

        # Add phase progress bar
        phase_progress = (current_phase - 1) / 4 * 200  # 200 pixels wide, 5 phases
        cv2.rectangle(info_panel, (10, 100), (210, 115), (200, 200, 200), -1)
        cv2.rectangle(info_panel, (10, 100), (int(10 + phase_progress), 115), (0, 200, 0), -1)
        cv2.putText(
            info_panel, f"Progress: Phase {current_phase}/5", (220, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )

        # Combine frame and info panel
        combined_frame = np.vstack([frame, info_panel])

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")
    return output_path


def visualize_trajectories(file_path, n_trajectories=2, specific_trajs=None):
    """
    Visualize trajectories with phase information

    Args:
        file_path: Path to the H5 file containing trajectories
        n_trajectories: Number of random trajectories to visualize (if specific_trajs is None)
        specific_trajs: List of specific trajectory keys to visualize (e.g., ['traj_0', 'traj_1'])
    """
    with h5py.File(file_path, "r") as f:
        # Get trajectory keys that have phase information
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]

        if specific_trajs:
            selected_trajs = [t for t in specific_trajs if t in traj_keys]
        else:
            selected_trajs = np.random.choice(traj_keys, size=min(n_trajectories, len(traj_keys)), replace=False)

        print(f"Selected trajectories: {selected_trajs}")

        for traj_key in selected_trajs:
            print(f"\n{'='*60}")
            print(f"Processing {traj_key}...")
            print("=" * 60)

            # Load phases
            phases = f[f"{traj_key}/phase"][:]

            # Load gripper states
            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
            agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]

            # Extract gripper positions (last joint position)
            # Using threshold of 0.02 to determine open/closed state
            gripper_threshold = 0.02
            agent0_gripper = (agent0_qpos[:, -1] < gripper_threshold).astype(int)
            agent1_gripper = (agent1_qpos[:, -1] < gripper_threshold).astype(int)
            agent2_gripper = (agent2_qpos[:, -1] < gripper_threshold).astype(int)

            # Print phase statistics
            phase_counts = np.bincount(phases)
            print("\nPhase distribution:")
            for i, count in enumerate(phase_counts):
                if count > 0:
                    print(f"  Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")

            # Print gripper statistics
            print("\nGripper statistics (closed ratio):")
            print(f"  Agent0: {agent0_gripper.mean():.3f}")
            print(f"  Agent1: {agent1_gripper.mean():.3f}")
            print(f"  Agent2: {agent2_gripper.mean():.3f}")

            # Create videos for different camera views
            camera_names = ["head_camera_agent0", "head_camera_agent1", "head_camera_global"]

            for camera_name in camera_names:
                try:
                    rgb_data = f[f"{traj_key}/obs/sensor_data/{camera_name}/rgb"][:]
                    print(f"\nCreating video for {camera_name}...")
                    print(f"  RGB data shape: {rgb_data.shape}")

                    create_phase_video(
                        rgb_data, phases, (agent0_gripper, agent1_gripper, agent2_gripper), traj_key, camera_name
                    )
                except KeyError as e:
                    print(f"  Camera {camera_name} not found in trajectory: {e}")
                except Exception as e:
                    print(f"  Error creating video for {camera_name}: {e}")


def print_h5_structure(file_path, traj_key="traj_0"):
    """
    Print the structure of the H5 file to understand available cameras

    Args:
        file_path: Path to the H5 file
        traj_key: Trajectory key to examine
    """
    print(f"\n{'='*60}")
    print(f"H5 File Structure for {traj_key}")
    print("=" * 60)

    with h5py.File(file_path, "r") as f:
        if traj_key in f:
            print(f"\nAvailable groups in {traj_key}:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"  {name}/")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape} {obj.dtype}")

            f[traj_key].visititems(print_structure)
        else:
            print(f"Trajectory {traj_key} not found!")
            print(f"Available trajectories: {[k for k in f.keys() if k.startswith('traj_')][:5]}...")


if __name__ == "__main__":
    # File path
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/three_arms/three_arms_place_objects_123_213/20260411_095345_with_phases.h5"

    print("=" * 60)
    print("Phase Visualization Tool for Three Stack Cube Task")
    print("=" * 60)

    # First, print the structure to see available cameras
    print_h5_structure(file_path, "traj_0")

    # Visualize specific trajectories
    print("\n\n" + "=" * 60)
    print("Creating Phase Videos")
    print("=" * 60)

    # Option 1: Visualize specific trajectories
    visualize_trajectories(file_path, specific_trajs=["traj_40", "traj_60"])

    # Option 2: Visualize random trajectories
    # visualize_trajectories(file_path, n_trajectories=2)

    print("\n" + "=" * 60)
    print("Done! Check the phase_videos_therestackcube/ directory for output videos.")
    print("=" * 60)
