import os

import cv2
import h5py
import numpy as np


def create_phase_video(rgb_data, phases, gripper_states, traj_key, camera_name, output_dir="phase_videos"):
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
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height + 100))  # Extra space for info

    agent0_gripper, agent1_gripper, agent2_gripper = gripper_states

    phase_colors = {
        1: (255, 200, 200),  # Light red - Phase 1
        2: (200, 255, 200),  # Light green - Phase 2
        3: (200, 200, 255),  # Light blue - Phase 3
        4: (255, 200, 255),  # Light purple - Phase 4
    }

    phase_names = {
        1: "Phase 1: All Robots Gripping & Agent0 Placing",
        2: "Phase 2: Agent1 Placing",
        3: "Phase 3: Agent2 Placing",
        4: "Phase 4: Agent3 Placing",
    }

    for i in range(n_frames):
        frame = rgb_data[i].copy()

        # Create info panel
        info_panel = np.ones((100, width, 3), dtype=np.uint8) * 255

        # Current phase info
        current_phase = phases[i]
        phase_color = phase_colors[current_phase]

        # Add phase background color
        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add text info to info panel
        cv2.putText(info_panel, f"Frame: {i}/{n_frames-1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(info_panel, f"{phase_names[current_phase]}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Gripper states for all three agents
        agent0_grip_color = (0, 150, 0) if agent0_gripper[i] else (150, 0, 0)
        agent1_grip_color = (0, 150, 0) if agent1_gripper[i] else (150, 0, 0)
        agent2_grip_color = (0, 150, 0) if agent2_gripper[i] else (150, 0, 0)

        cv2.putText(
            info_panel,
            f"A0 Grip: {'CLOSED' if agent0_gripper[i] else 'OPEN'}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent0_grip_color,
            1,
        )
        cv2.putText(
            info_panel,
            f"A1 Grip: {'CLOSED' if agent1_gripper[i] else 'OPEN'}",
            (150, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent1_grip_color,
            1,
        )
        cv2.putText(
            info_panel,
            f"A2 Grip: {'CLOSED' if agent2_gripper[i] else 'OPEN'}",
            (290, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            agent2_grip_color,
            1,
        )

        # Add phase progress bar
        phase_progress = (current_phase - 1) / 3 * 200  # 200 pixels wide
        cv2.rectangle(info_panel, (400, 70), (600, 90), (200, 200, 200), -1)
        cv2.rectangle(info_panel, (400, 70), (int(400 + phase_progress), 90), (0, 200, 0), -1)
        cv2.putText(
            info_panel, f"Progress: Phase {current_phase}/3", (610, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
        )

        # Combine frame and info panel
        combined_frame = np.vstack([frame, info_panel])

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")
    return output_path


def visualize_trajectories(file_path, n_trajectories=2):
    """
    Visualize random trajectories with phase information

    Args:
        file_path: Path to the H5 file containing trajectories
        n_trajectories: Number of trajectories to visualize
    """
    with h5py.File(file_path, "r") as f:
        # Get trajectory keys that have phase information
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        selected_trajs = np.random.choice(traj_keys, size=min(n_trajectories, len(traj_keys)), replace=False)

        print(f"Selected trajectories: {selected_trajs}")

        for traj_key in selected_trajs:
            print(f"\nProcessing {traj_key}...")

            # Load phases
            phases = f[f"{traj_key}/phase"][:]

            # Load gripper states
            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
            agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]

            # Extract gripper positions (last joint position)
            agent0_gripper = (agent0_qpos[:, -1] <= np.mean(agent0_qpos[:, -1])).astype(int)
            agent1_gripper = (agent1_qpos[:, -1] <= np.mean(agent1_qpos[:, -1])).astype(int)
            agent2_gripper = (agent2_qpos[:, -1] <= np.mean(agent2_qpos[:, -1])).astype(int)

            # Print phase statistics
            phase_counts = np.bincount(phases)
            print("Phase distribution:")
            for i, count in enumerate(phase_counts):
                if count > 0:
                    print(f"Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")

            # Create videos for different camera views
            camera_names = ["head_camera_agent0", "head_camera_agent1", "head_camera_global"]

            for camera_name in camera_names:
                try:
                    rgb_data = f[f"{traj_key}/obs/sensor_data/{camera_name}/rgb"][:]
                    print(f"Creating video for {camera_name}...")

                    create_phase_video(
                        rgb_data, phases, (agent0_gripper, agent1_gripper, agent2_gripper), traj_key, camera_name
                    )
                except Exception as e:
                    print(f"Error creating video for {camera_name}: {e}")


if __name__ == "__main__":
    # Update this path to your H5 file with phase information
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_order_123_20251211_with_phases.h5"

    print("=== Phase Visualization Tool ===")
    visualize_trajectories(file_path, n_trajectories=2)
    print("\nDone! Check the phase_videos/ directory for output videos.")
