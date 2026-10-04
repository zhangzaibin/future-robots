import os
import random

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.02):
    """
    Analyze gripper state from qpos data.
    Returns binary array: 1 for closed/gripping, 0 for open
    """
    gripper_pos = qpos[:, -1]  # Last joint is usually gripper
    gripper_closed = gripper_pos < gripper_threshold
    return gripper_closed.astype(int)


def detect_phases(agent0_gripper, agent1_gripper):
    """
    Detect phases based on gripper states:
    Phase 1: Before agent0 grips (agent0 open)
    Phase 2: Agent0 gripping to agent0 release
    Phase 3: Agent0 released to agent1 grips
    Phase 4: Agent1 gripping to end
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    # Find grip/release events
    agent0_grip_start = None
    agent0_grip_end = None
    agent1_grip_start = None

    # Detect agent0 grip start (transition from 0 to 1)
    for i in range(1, n_steps):
        if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break

    # Detect agent0 grip end (transition from 1 to 0)
    if agent0_grip_start is not None:
        for i in range(agent0_grip_start + 1, n_steps):
            if agent0_gripper[i - 1] == 1 and agent0_gripper[i] == 0:
                agent0_grip_end = i
                break

    # Detect agent1 grip start (transition from 0 to 1)
    start_search = agent0_grip_end if agent0_grip_end is not None else 0
    for i in range(start_search + 1, n_steps):
        if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
            agent1_grip_start = i
            break

    # Assign phases
    if agent0_grip_start is not None:
        phases[:agent0_grip_start] = 1

        if agent0_grip_end is not None:
            phases[agent0_grip_start:agent0_grip_end] = 2

            if agent1_grip_start is not None:
                phases[agent0_grip_end:agent1_grip_start] = 3
                phases[agent1_grip_start:] = 4
            else:
                phases[agent0_grip_end:] = 3
        else:
            phases[agent0_grip_start:] = 2
    elif agent1_grip_start is not None:
        phases[:agent1_grip_start] = 1
        phases[agent1_grip_start:] = 4
    else:
        phases[:] = 1

    return phases, (agent0_grip_start, agent0_grip_end, agent1_grip_start)


def create_phase_video(
    rgb_data, phases, gripper_states, phase_events, traj_key, camera_name, output_dir="phase_videos"
):
    """
    Create a video showing different phases with gripper state overlay
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = len(rgb_data)
    height, width = rgb_data[0].shape[:2]

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, f"{traj_key}_{camera_name}_phases.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height + 100))  # Extra space for info

    agent0_gripper, agent1_gripper = gripper_states
    agent0_grip_start, agent0_grip_end, agent1_grip_start = phase_events

    phase_colors = {
        1: (255, 100, 100),  # Light red
        2: (100, 255, 100),  # Light green
        3: (100, 100, 255),  # Light blue
        4: (255, 255, 100),  # Light yellow
    }

    phase_names = {
        1: "Phase 1: Before Agent0 Grips",
        2: "Phase 2: Agent0 Gripping",
        3: "Phase 3: Agent0 Released, Before Agent1",
        4: "Phase 4: Agent1 Gripping",
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
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Add text info to info panel
        cv2.putText(info_panel, f"Frame: {i}/{n_frames-1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(info_panel, f"{phase_names[current_phase]}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Gripper states
        agent0_color = (0, 255, 0) if agent0_gripper[i] else (0, 0, 255)
        agent1_color = (0, 255, 0) if agent1_gripper[i] else (0, 0, 255)

        cv2.putText(
            info_panel,
            f"Agent0 Gripper: {'CLOSED' if agent0_gripper[i] else 'OPEN'}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            agent0_color,
            2,
        )
        cv2.putText(
            info_panel,
            f"Agent1 Gripper: {'CLOSED' if agent1_gripper[i] else 'OPEN'}",
            (250, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            agent1_color,
            2,
        )

        # Mark key events
        if i == agent0_grip_start:
            cv2.putText(info_panel, ">>> AGENT0 GRIPS <<<", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif i == agent0_grip_end:
            cv2.putText(info_panel, ">>> AGENT0 RELEASES <<<", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif i == agent1_grip_start:
            cv2.putText(info_panel, ">>> AGENT1 GRIPS <<<", (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Combine frame and info panel
        combined_frame = np.vstack([frame, info_panel])

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")
    return output_path


def create_phase_comparison_plot(phases_data, output_dir="phase_plots"):
    """
    Create comparison plots for multiple trajectories
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(len(phases_data), 1, figsize=(15, 3 * len(phases_data)))
    if len(phases_data) == 1:
        axes = [axes]

    phase_colors = ["red", "green", "blue", "orange"]

    for idx, (traj_key, data) in enumerate(phases_data.items()):
        phases = data["phases"]
        agent0_gripper = data["agent0_gripper"]
        agent1_gripper = data["agent1_gripper"]
        events = data["events"]

        ax = axes[idx]
        time_steps = np.arange(len(phases))

        # Plot phases as background
        for phase in [1, 2, 3, 4]:
            phase_mask = phases == phase
            if np.any(phase_mask):
                ax.fill_between(
                    time_steps, 0, 3, where=phase_mask, alpha=0.3, color=phase_colors[phase - 1], label=f"Phase {phase}"
                )

        # Plot gripper states
        ax.plot(time_steps, agent0_gripper + 0.1, "b-", linewidth=2, label="Agent0 Gripper")
        ax.plot(time_steps, agent1_gripper + 1.1, "r-", linewidth=2, label="Agent1 Gripper")

        # Mark key events
        agent0_grip_start, agent0_grip_end, agent1_grip_start = events
        if agent0_grip_start is not None:
            ax.axvline(agent0_grip_start, color="green", linestyle="--", alpha=0.7, label="Agent0 Grip Start")
        if agent0_grip_end is not None:
            ax.axvline(agent0_grip_end, color="orange", linestyle="--", alpha=0.7, label="Agent0 Grip End")
        if agent1_grip_start is not None:
            ax.axvline(agent1_grip_start, color="purple", linestyle="--", alpha=0.7, label="Agent1 Grip Start")

        ax.set_title(f"{traj_key} - Phase Analysis")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State")
        ax.set_ylim(-0.5, 2.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "phase_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved: {output_path}")


def visualize_random_trajectories(file_path, n_trajectories=2, gripper_threshold=0.02):
    """
    Visualize random trajectories with phase detection
    """
    with h5py.File(file_path, "r") as f:
        # Get all trajectory keys
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]

        # Randomly select trajectories
        selected_trajs = random.sample(traj_keys, min(n_trajectories, len(traj_keys)))
        print(f"Selected trajectories: {selected_trajs}")

        phases_data = {}

        for traj_key in selected_trajs:
            print(f"\n=== Processing {traj_key} ===")

            # Load data
            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

            # Analyze gripper states
            agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
            agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

            # Detect phases
            phases, events = detect_phases(agent0_gripper, agent1_gripper)

            # Store data for comparison
            phases_data[traj_key] = {
                "phases": phases,
                "agent0_gripper": agent0_gripper,
                "agent1_gripper": agent1_gripper,
                "events": events,
            }

            # Print analysis
            print(f"Trajectory length: {len(phases)} steps")
            phase_counts = np.bincount(phases)
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
                        rgb_data, phases, (agent0_gripper, agent1_gripper), events, traj_key, camera_name
                    )
                except Exception as e:
                    print(f"Error creating video for {camera_name}: {e}")

        # Create comparison plot
        print("\nCreating comparison plot...")
        create_phase_comparison_plot(phases_data)

        return phases_data


def analyze_phase_statistics(file_path, sample_size=10):
    """
    Analyze phase statistics across multiple trajectories
    """
    with h5py.File(file_path, "r") as f:
        traj_keys = [key for key in f.keys() if key.startswith("traj_")]
        selected_trajs = random.sample(traj_keys, min(sample_size, len(traj_keys)))

        all_phase_distributions = []

        for traj_key in selected_trajs:
            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

            agent0_gripper = analyze_gripper_state(agent0_qpos)
            agent1_gripper = analyze_gripper_state(agent1_qpos)

            phases, _ = detect_phases(agent0_gripper, agent1_gripper)
            phase_counts = np.bincount(phases, minlength=5)[1:5]  # phases 1-4
            phase_ratios = phase_counts / len(phases)

            all_phase_distributions.append(phase_ratios)

        # Statistics
        all_phase_distributions = np.array(all_phase_distributions)
        mean_ratios = np.mean(all_phase_distributions, axis=0)
        std_ratios = np.std(all_phase_distributions, axis=0)

        print(f"\n=== Phase Statistics (n={len(selected_trajs)}) ===")
        for i in range(4):
            print(f"Phase {i+1}: {mean_ratios[i]:.3f} ± {std_ratios[i]:.3f}")


if __name__ == "__main__":
    # File path
    file_path = "/18141169908/zhangzaibin/multivla-pi0/checkpoints/pass_shoe--only-count-success.h5"

    print("=== Phase Visualization Tool ===")

    # Analyze phase statistics
    print("Analyzing phase statistics...")
    analyze_phase_statistics(file_path, sample_size=10)

    # Visualize random trajectories
    print("\nVisualizing random trajectories...")
    phases_data = visualize_random_trajectories(file_path, n_trajectories=2)

    print("\n=== Done! ===")
    print("Check the following directories:")
    print("- phase_videos/: Individual trajectory videos")
    print("- phase_plots/: Comparison plots")
