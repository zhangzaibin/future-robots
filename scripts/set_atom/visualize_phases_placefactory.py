#!/usr/bin/env python3
"""
Phase visualization tool for placefactory sequential grasping task
Creates videos showing phase transitions and gripper states
"""

import argparse
import os

import cv2
import h5py
import numpy as np

# Global output directory (can be set via command line)
OUTPUT_DIR = "phase_videos_placefactory"


def create_phase_video(rgb_data, phases, gripper_states, traj_key, camera_name, output_dir=None):
    """
    Create a video showing different phases with gripper state overlay for placefactory task

    Args:
        rgb_data: RGB frames from the camera
        phases: Array of phase labels for each frame
        gripper_states: Tuple of (agent0_gripper, agent1_gripper, agent2_gripper) binary states
        traj_key: Trajectory key for naming
        camera_name: Camera name for naming
        output_dir: Directory to save videos (uses global OUTPUT_DIR if None)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = len(rgb_data)
    height, width = rgb_data[0].shape[:2]

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, f"{traj_key}_{camera_name}_phases.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height + 120))  # Extra space for info

    agent0_gripper, agent1_gripper, agent2_gripper = gripper_states

    # Phase colors for placefactory (3 phases only)
    phase_colors = {
        1: (135, 206, 235),  # Sky blue - Agent0's turn
        2: (144, 238, 144),  # Light green - Agent1's turn
        3: (255, 182, 193),  # Light pink - Agent2's turn
    }

    # Phase names for sequential grasping
    phase_names = {
        1: "Phase 1: Agent0 Grasping & Placing",
        2: "Phase 2: Agent1 Grasping & Placing",
        3: "Phase 3: Agent2 Grasping & Placing",
    }

    for i in range(n_frames):
        frame = rgb_data[i].copy()

        # Create info panel
        info_panel = np.ones((120, width, 3), dtype=np.uint8) * 255

        # Current phase info
        current_phase = phases[i]
        phase_color = phase_colors.get(current_phase, (200, 200, 200))

        # Add phase background color (subtle overlay)
        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        # Add border to indicate current phase
        border_thickness = 8
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), phase_color, border_thickness)

        # === Info Panel ===

        # Line 1: Frame counter and trajectory
        cv2.putText(
            info_panel,
            f"{traj_key} | Frame: {i+1}/{n_frames} | {camera_name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # Line 2: Current phase with colored background
        phase_text = phase_names.get(current_phase, f"Phase {current_phase}")
        text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        # Draw colored background for phase text
        cv2.rectangle(info_panel, (8, 35), (12 + text_size[0], 60), phase_color, -1)
        cv2.putText(info_panel, phase_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Line 3: Gripper states for all three agents
        y_pos = 85
        x_start = 10
        x_gap = 180

        agents_info = [
            ("Agent0 (panda-0)", agent0_gripper[i], 1),
            ("Agent1 (panda-1)", agent1_gripper[i], 2),
            ("Agent2 (panda-2)", agent2_gripper[i], 3),
        ]

        for idx, (agent_name, gripper_state, agent_phase) in enumerate(agents_info):
            x_pos = x_start + idx * x_gap

            # Determine color: green if closed, red if open
            grip_color = (0, 180, 0) if gripper_state else (200, 0, 0)
            grip_text = "CLOSED" if gripper_state else "OPEN"

            # Highlight if this agent's phase
            if current_phase == agent_phase:
                # Draw highlight box
                cv2.rectangle(info_panel, (x_pos - 5, y_pos - 20), (x_pos + 170, y_pos + 10), phase_color, 2)

            # Draw gripper state
            cv2.putText(
                info_panel, f"{agent_name}: {grip_text}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grip_color, 2
            )

        # Line 4: Phase progress bar
        progress_x = width - 230
        progress_y = 85
        bar_width = 200
        bar_height = 20

        # Background bar
        cv2.rectangle(
            info_panel, (progress_x, progress_y - 15), (progress_x + bar_width, progress_y + 5), (220, 220, 220), -1
        )

        # Progress fill (based on phase)
        phase_progress = (current_phase - 1) / 2 * bar_width  # 3 phases: 0%, 50%, 100%
        cv2.rectangle(
            info_panel,
            (progress_x, progress_y - 15),
            (int(progress_x + phase_progress), progress_y + 5),
            (50, 200, 50),
            -1,
        )

        # Progress text
        cv2.putText(
            info_panel,
            f"{current_phase}/3",
            (progress_x + bar_width + 10, progress_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # Combine frame and info panel
        combined_frame = np.vstack([frame, info_panel])

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

    out.release()
    print(f"  ✓ Video saved: {output_path}")
    return output_path


def apply_hysteresis(gripper_pos, close_threshold, open_threshold):
    """Apply hysteresis to gripper positions to get binary state"""
    n_steps = len(gripper_pos)
    state = np.zeros(n_steps, dtype=int)

    state[0] = 1 if gripper_pos[0] <= close_threshold else 0

    for i in range(1, n_steps):
        if state[i - 1] == 0:  # Was open
            state[i] = 1 if gripper_pos[i] <= close_threshold else 0
        else:  # Was closed
            state[i] = 0 if gripper_pos[i] >= open_threshold else 1

    return state


def visualize_trajectories(file_path, traj_indices=None, n_trajectories=2, camera_views=None):
    """
    Visualize trajectories with phase information for placefactory task

    Args:
        file_path: Path to the H5 file containing trajectories with phases
        traj_indices: Specific trajectory indices to visualize (e.g., [0, 1, 5])
        n_trajectories: Number of random trajectories to visualize (if traj_indices is None)
        camera_views: List of camera names to visualize (default: all available)
    """
    with h5py.File(file_path, "r") as f:
        # Get trajectory keys
        traj_keys = sorted([key for key in f.keys() if key.startswith("traj_")])

        # Select trajectories
        if traj_indices is not None:
            selected_trajs = [f"traj_{idx}" for idx in traj_indices if f"traj_{idx}" in traj_keys]
        else:
            selected_trajs = np.random.choice(traj_keys, size=min(n_trajectories, len(traj_keys)), replace=False)

        print(f"\n{'='*60}")
        print(f"Visualizing {len(selected_trajs)} trajectories from: {file_path}")
        print(f"Selected trajectories: {selected_trajs}")
        print(f"{'='*60}\n")

        # Default camera views
        if camera_views is None:
            camera_views = ["head_camera_global", "head_camera_agent0", "head_camera_agent1", "head_camera_agent2"]

        for traj_idx, traj_key in enumerate(selected_trajs):
            print(f"\n[{traj_idx+1}/{len(selected_trajs)}] Processing {traj_key}...")

            # Load phases
            if f"{traj_key}/phase" not in f:
                print(f"  ✗ No phase data found for {traj_key}, skipping...")
                continue

            phases = f[f"{traj_key}/phase"][:]

            # Try to load execution order from attributes
            execution_order = "Unknown"
            try:
                if "execution_order" in f[f"{traj_key}/phase"].attrs:
                    execution_order = f[f"{traj_key}/phase"].attrs["execution_order"]
                    print(f"  Execution order: {execution_order}")
            except:
                pass

            # Load gripper positions for all agents
            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
            agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]

            # Calculate unified thresholds (same as in phase detection)
            all_gripper_pos = np.concatenate([agent0_qpos[:, -1], agent1_qpos[:, -1], agent2_qpos[:, -1]])
            gripper_min = all_gripper_pos.min()
            gripper_max = all_gripper_pos.max()
            gripper_range = gripper_max - gripper_min
            close_ratio = 0.3
            unified_close_threshold = gripper_min + gripper_range * close_ratio
            unified_open_threshold = gripper_min + gripper_range * (close_ratio + 0.2)

            # Apply hysteresis to get binary gripper states
            agent0_gripper = apply_hysteresis(agent0_qpos[:, -1], unified_close_threshold, unified_open_threshold)
            agent1_gripper = apply_hysteresis(agent1_qpos[:, -1], unified_close_threshold, unified_open_threshold)
            agent2_gripper = apply_hysteresis(agent2_qpos[:, -1], unified_close_threshold, unified_open_threshold)

            # Print phase statistics
            phase_counts = np.bincount(phases)
            print("  Phase distribution:")
            for i, count in enumerate(phase_counts):
                if count > 0:
                    print(f"    Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")

            # Print gripper statistics
            print("  Gripper closed ratio:")
            print(f"    Agent0: {agent0_gripper.mean():.3f}")
            print(f"    Agent1: {agent1_gripper.mean():.3f}")
            print(f"    Agent2: {agent2_gripper.mean():.3f}")

            # Create videos for different camera views
            for camera_name in camera_views:
                camera_path = f"{traj_key}/obs/sensor_data/{camera_name}/rgb"
                if camera_path not in f:
                    print(f"  ⚠ Camera {camera_name} not found, skipping...")
                    continue

                try:
                    rgb_data = f[camera_path][:]
                    print(f"  Creating video for {camera_name} ({rgb_data.shape})...")

                    create_phase_video(
                        rgb_data, phases, (agent0_gripper, agent1_gripper, agent2_gripper), traj_key, camera_name
                    )
                except Exception as e:
                    print(f"  ✗ Error creating video for {camera_name}: {e}")
                    import traceback

                    traceback.print_exc()

        print(f"\n{'='*60}")
        print("✓ All done! Videos saved to: phase_videos_placefactory/")
        print(f"{'='*60}")


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description="Visualize phase transitions for sequential grasping tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize specific trajectories
  python %(prog)s /path/to/dataset_with_phases.h5 --trajs 0 1 5
  
  # Visualize random trajectories
  python %(prog)s /path/to/dataset_with_phases.h5 --random 3
  
  # Select specific camera views
  python %(prog)s /path/to/dataset_with_phases.h5 --cameras head_camera_global head_camera_agent0
  
  # Specify output directory
  python %(prog)s /path/to/dataset_with_phases.h5 --output-dir my_videos
        """,
    )

    parser.add_argument("input_file", type=str, help="Path to H5 file with phase information")
    parser.add_argument(
        "--trajs",
        "-t",
        type=int,
        nargs="+",
        default=None,
        help="Specific trajectory indices to visualize (e.g., 0 1 5)",
    )
    parser.add_argument("--random", "-r", type=int, default=None, help="Number of random trajectories to visualize")
    parser.add_argument(
        "--cameras",
        "-c",
        nargs="+",
        default=["head_camera_global", "head_camera_agent0"],
        help="Camera views to include (default: head_camera_global head_camera_agent0)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="phase_videos_placefactory",
        help="Output directory for videos (default: phase_videos_placefactory)",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"✗ Error: Input file not found: {args.input_file}")
        return 1

    print("\n" + "=" * 70)
    print("Phase Visualization Tool for PlaceFactory Sequential Grasping")
    print("=" * 70)
    print(f"Input file:     {args.input_file}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Camera views:   {args.cameras}")
    if args.trajs:
        print(f"Trajectories:   {args.trajs}")
    elif args.random:
        print(f"Random trajs:   {args.random}")
    print("=" * 70)

    # Set output directory globally (for create_phase_video function)
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    # Visualize trajectories
    try:
        if args.trajs:
            visualize_trajectories(args.input_file, traj_indices=args.trajs, camera_views=args.cameras)
        else:
            n_random = args.random if args.random else 2
            visualize_trajectories(args.input_file, n_trajectories=n_random, camera_views=args.cameras)

        print(f"\n✓ Videos saved to: {args.output_dir}/")
        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
