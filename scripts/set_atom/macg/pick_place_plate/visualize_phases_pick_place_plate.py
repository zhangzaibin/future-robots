"""
Visualize phase videos for the pick-and-place-to-plate task.
每个机械臂分别抓取物体放到盘子里。

Generates mp4 videos with phase overlay, gripper state indicators,
and a phase progress bar for each camera view.

Cameras:
  - head_camera_global   (全局视角)
  - wrist_camera_agent0  (Agent0 腕部)
  - wrist_camera_agent1  (Agent1 腕部)
  - wrist_camera_agent2  (Agent2 腕部)
"""

import os

import cv2
import h5py
import numpy as np


# ========== Phase Meta ==========

PHASE_COLORS = {
    1: (255, 180, 180),  # Light red    - Agent0 pick up
    2: (180, 255, 180),  # Light green  - Agent0 place
    3: (180, 180, 255),  # Light blue   - Agent1 pick up
    4: (255, 255, 180),  # Light yellow - Agent1 place
    5: (255, 180, 255),  # Light purple - Agent2 pick up
    6: (180, 255, 255),  # Light cyan   - Agent2 place
    7: (220, 220, 220),  # Light grey   - Wait / Done
}

PHASE_NAMES = {
    1: "Phase 1: Agent0 Pick Up",
    2: "Phase 2: Agent0 Place",
    3: "Phase 3: Agent1 Pick Up",
    4: "Phase 4: Agent1 Place",
    5: "Phase 5: Agent2 Pick Up",
    6: "Phase 6: Agent2 Place",
    7: "Phase 7: Wait / Done",
}

TOTAL_PHASES = 7


# ========== Video Creation ==========

def create_phase_video(
    rgb_data,
    phases,
    gripper_states,
    traj_key,
    camera_name,
    output_dir="phase_videos_pick_place_plate",
    fps=10.0,
):
    """
    Create a video showing different phases with gripper state overlay.

    Args:
        rgb_data:        RGB frames (T, H, W, 3)
        phases:          Phase label per frame (T,)
        gripper_states:  Tuple of binary arrays (agent0, agent1, agent2)
        traj_key:        Trajectory key for naming
        camera_name:     Camera name for naming
        output_dir:      Output directory
        fps:             Video frame rate
    """
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(rgb_data)
    height, width = rgb_data[0].shape[:2]
    info_panel_height = 130

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, f"{traj_key}_{camera_name}_phases.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + info_panel_height))

    agent0_gripper, agent1_gripper, agent2_gripper = gripper_states

    for i in range(n_frames):
        frame = rgb_data[i].copy()
        current_phase = phases[i]
        phase_color = PHASE_COLORS.get(current_phase, (200, 200, 200))

        # --- Phase color overlay on the video frame ---
        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        # --- Info panel (white background) ---
        info_panel = np.ones((info_panel_height, width, 3), dtype=np.uint8) * 255

        # Row 1: Frame counter
        cv2.putText(
            info_panel,
            f"Frame: {i}/{n_frames - 1}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

        # Row 2: Phase name
        phase_name = PHASE_NAMES.get(current_phase, f"Phase {current_phase}")
        cv2.putText(
            info_panel,
            phase_name,
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

        # Row 3: Gripper states for all three agents
        for agent_idx, (gripper, label_x) in enumerate(
            zip(
                [agent0_gripper, agent1_gripper, agent2_gripper],
                [10, 180, 350],
            )
        ):
            grip_val = gripper[i] if i < len(gripper) else 0
            color = (0, 150, 0) if grip_val else (150, 0, 0)
            text = f"Agent{agent_idx}: {'CLOSED' if grip_val else 'OPEN'}"
            cv2.putText(
                info_panel,
                text,
                (label_x, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )

        # Row 4: Phase progress bar
        bar_x0, bar_y0 = 10, 100
        bar_width, bar_height = 300, 15
        # Background
        cv2.rectangle(
            info_panel,
            (bar_x0, bar_y0),
            (bar_x0 + bar_width, bar_y0 + bar_height),
            (200, 200, 200),
            -1,
        )
        # Fill segments for each phase by color
        for p in range(1, TOTAL_PHASES + 1):
            seg_x0 = bar_x0 + int((p - 1) / TOTAL_PHASES * bar_width)
            seg_x1 = bar_x0 + int(p / TOTAL_PHASES * bar_width)
            if p <= current_phase:
                pc = PHASE_COLORS.get(p, (200, 200, 200))
                # Convert to darker version for the bar
                dark = tuple(max(0, c - 40) for c in pc)
                cv2.rectangle(info_panel, (seg_x0, bar_y0), (seg_x1, bar_y0 + bar_height), dark, -1)
        # Outline
        cv2.rectangle(
            info_panel,
            (bar_x0, bar_y0),
            (bar_x0 + bar_width, bar_y0 + bar_height),
            (100, 100, 100),
            1,
        )
        cv2.putText(
            info_panel,
            f"Phase {current_phase}/{TOTAL_PHASES}",
            (bar_x0 + bar_width + 10, bar_y0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
        )

        # --- Combine frame + info panel ---
        combined = np.vstack([frame, info_panel])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

    out.release()
    print(f"  Video saved: {output_path}")
    return output_path


# ========== Main Visualization Logic ==========

def visualize_trajectories(
    file_path,
    output_dir="phase_videos_pick_place_plate",
    n_trajectories=2,
    specific_trajs=None,
    camera_names=None,
    agent_names=None,
    gripper_threshold=0.02,
    fps=10.0,
):
    """
    Create phase-overlay videos for selected trajectories.

    Args:
        file_path:          Path to the H5 file (with phases)
        output_dir:         Output directory for videos
        n_trajectories:     Number of random trajectories (if specific_trajs is None)
        specific_trajs:     List of specific trajectory keys, e.g. ["traj_0", "traj_3"]
        camera_names:       List of camera sub-paths under obs/sensor_data/
        agent_names:        List of agent sub-paths under obs/agent/
        gripper_threshold:  Threshold for binary gripper display
        fps:                Video frame rate
    """
    if camera_names is None:
        camera_names = [
            "head_camera_global",
            "wrist_camera_agent0",
            "wrist_camera_agent1",
            "wrist_camera_agent2",
        ]
    if agent_names is None:
        agent_names = ["panda-0", "panda-1", "panda-2"]

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda k: int(k.split("_")[1]),
        )

        if specific_trajs:
            selected = [t for t in specific_trajs if t in traj_keys]
        else:
            selected = list(
                np.random.choice(
                    traj_keys,
                    size=min(n_trajectories, len(traj_keys)),
                    replace=False,
                )
            )

        print(f"Selected trajectories: {selected}")

        for traj_key in selected:
            print(f"\n{'=' * 60}")
            print(f"Processing {traj_key}...")
            print("=" * 60)

            # --- Load phase data ---
            if f"{traj_key}/phase" not in f:
                print(f"  WARNING: {traj_key}/phase not found, skipping.")
                continue
            phases = f[f"{traj_key}/phase"][:]

            # --- Load & binarize gripper states ---
            gripper_arrays = []
            for name in agent_names:
                qpos = f[f"{traj_key}/obs/agent/{name}/qpos"][:]
                gripper_bin = (qpos[:, -1] < gripper_threshold).astype(int)
                gripper_arrays.append(gripper_bin)

            gripper_states = tuple(gripper_arrays)

            # --- Phase statistics ---
            phase_counts = np.bincount(phases)
            print("\n  Phase distribution:")
            for p, count in enumerate(phase_counts):
                if count > 0:
                    pname = PHASE_NAMES.get(p, f"Phase {p}")
                    print(f"    {pname}: {count} steps ({count / len(phases) * 100:.1f}%)")

            print(f"\n  Gripper closed ratio:")
            for idx, name in enumerate(agent_names):
                print(f"    {name}: {gripper_arrays[idx].mean():.3f}")

            # --- Generate videos per camera ---
            for cam in camera_names:
                cam_path = f"{traj_key}/obs/sensor_data/{cam}/rgb"
                if cam_path not in f:
                    print(f"  Camera '{cam}' not found, skipping.")
                    continue

                rgb_data = f[cam_path][:]
                print(f"\n  Creating video for {cam}  (shape: {rgb_data.shape})")

                try:
                    create_phase_video(
                        rgb_data,
                        phases,
                        gripper_states,
                        traj_key,
                        cam,
                        output_dir=output_dir,
                        fps=fps,
                    )
                except Exception as e:
                    print(f"  Error creating video for {cam}: {e}")
                    import traceback
                    traceback.print_exc()


# ========== H5 Structure Helper ==========

def print_h5_structure(file_path, traj_key="traj_0"):
    """Print the structure of the H5 file."""
    print(f"\n{'=' * 60}")
    print(f"H5 File Structure for {traj_key}")
    print("=" * 60)

    with h5py.File(file_path, "r") as f:
        if traj_key in f:

            def print_item(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"  {name}/")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape} {obj.dtype}")

            f[traj_key].visititems(print_item)
        else:
            print(f"  Trajectory {traj_key} not found!")
            available = [k for k in f.keys() if k.startswith("traj_")][:5]
            print(f"  Available: {available}...")


# ========== Main ==========

if __name__ == "__main__":
    # ---- Configuration ----
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_123_with_phases.h5"
    output_dir = "/18141169908/zhangzaibin/multivla-pi0/phase_videos_pick_place_plate"

    # ---- Print h5 structure ----
    print("=" * 60)
    print("Phase Video Visualization — Pick & Place to Plate")
    print("=" * 60)
    print_h5_structure(file_path, "traj_0")

    # ---- Create phase videos ----
    print("\n\n" + "=" * 60)
    print("Creating Phase Videos")
    print("=" * 60)

    visualize_trajectories(
        file_path,
        output_dir=output_dir,
        specific_trajs=["traj_0", "traj_3", "traj_10"],
        camera_names=[
            "head_camera_global",
            "wrist_camera_agent0",
            "wrist_camera_agent1",
            "wrist_camera_agent2",
        ],
        agent_names=["panda-0", "panda-1", "panda-2"],
        gripper_threshold=0.02,
        fps=10.0,
    )

    print("\n" + "=" * 60)
    print(f"Done! Check the output directory: {output_dir}")
    print("=" * 60)




















