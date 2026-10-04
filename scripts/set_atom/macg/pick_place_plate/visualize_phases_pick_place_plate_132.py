"""
Visualize phase videos for the pick-and-place-to-plate task — placefactory_132 版本.
Phase 编号 11-17，与 123 版本 (1-7) 区分。

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


# ========== Phase Meta (132 version: phases 11-17) ==========

PHASE_COLORS = {
    11: (255, 180, 180),  # Light red    - Agent0 pick up
    12: (180, 255, 180),  # Light green  - Agent0 place
    13: (180, 180, 255),  # Light blue   - Agent1 pick up
    14: (255, 255, 180),  # Light yellow - Agent1 place
    15: (255, 180, 255),  # Light purple - Agent2 pick up
    16: (180, 255, 255),  # Light cyan   - Agent2 place
    17: (220, 220, 220),  # Light grey   - Wait / Done
}

PHASE_NAMES = {
    11: "Phase 11: Agent0 Pick Up",
    12: "Phase 12: Agent0 Place",
    13: "Phase 13: Agent1 Pick Up",
    14: "Phase 14: Agent1 Place",
    15: "Phase 15: Agent2 Pick Up",
    16: "Phase 16: Agent2 Place",
    17: "Phase 17: Wait / Done",
}

PHASE_MIN = 11
PHASE_MAX = 17
TOTAL_PHASES = PHASE_MAX - PHASE_MIN + 1  # 7


# ========== Video Creation ==========

def create_phase_video(
    rgb_data,
    phases,
    gripper_states,
    traj_key,
    camera_name,
    output_dir="phase_videos_pick_place_plate_132",
    fps=10.0,
):
    """
    Create a video showing different phases with gripper state overlay.
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

        # --- Phase color overlay ---
        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        # --- Info panel ---
        info_panel = np.ones((info_panel_height, width, 3), dtype=np.uint8) * 255

        # Row 1: Frame counter
        cv2.putText(
            info_panel, f"Frame: {i}/{n_frames - 1}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
        )

        # Row 2: Phase name
        phase_name = PHASE_NAMES.get(current_phase, f"Phase {current_phase}")
        cv2.putText(
            info_panel, phase_name,
            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
        )

        # Row 3: Gripper states
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
                info_panel, text,
                (label_x, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
            )

        # Row 4: Phase progress bar
        bar_x0, bar_y0 = 10, 100
        bar_width, bar_height = 300, 15
        cv2.rectangle(
            info_panel, (bar_x0, bar_y0),
            (bar_x0 + bar_width, bar_y0 + bar_height), (200, 200, 200), -1,
        )
        for p in range(PHASE_MIN, PHASE_MAX + 1):
            seg_x0 = bar_x0 + int((p - PHASE_MIN) / TOTAL_PHASES * bar_width)
            seg_x1 = bar_x0 + int((p - PHASE_MIN + 1) / TOTAL_PHASES * bar_width)
            if p <= current_phase:
                pc = PHASE_COLORS.get(p, (200, 200, 200))
                dark = tuple(max(0, c - 40) for c in pc)
                cv2.rectangle(info_panel, (seg_x0, bar_y0), (seg_x1, bar_y0 + bar_height), dark, -1)
        cv2.rectangle(
            info_panel, (bar_x0, bar_y0),
            (bar_x0 + bar_width, bar_y0 + bar_height), (100, 100, 100), 1,
        )
        cv2.putText(
            info_panel,
            f"Phase {current_phase}/{PHASE_MAX}",
            (bar_x0 + bar_width + 10, bar_y0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
        )

        # --- Combine ---
        combined = np.vstack([frame, info_panel])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

    out.release()
    print(f"  Video saved: {output_path}")
    return output_path


# ========== Main Visualization Logic ==========

def visualize_trajectories(
    file_path,
    output_dir="phase_videos_pick_place_plate_132",
    n_trajectories=2,
    specific_trajs=None,
    camera_names=None,
    agent_names=None,
    gripper_threshold=0.02,
    fps=10.0,
):
    """
    Create phase-overlay videos for selected trajectories (132 version).
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

            if f"{traj_key}/phase" not in f:
                print(f"  WARNING: {traj_key}/phase not found, skipping.")
                continue
            phases = f[f"{traj_key}/phase"][:]

            # Gripper states
            gripper_arrays = []
            for name in agent_names:
                qpos = f[f"{traj_key}/obs/agent/{name}/qpos"][:]
                gripper_bin = (qpos[:, -1] < gripper_threshold).astype(int)
                gripper_arrays.append(gripper_bin)

            gripper_states = tuple(gripper_arrays)

            # Phase statistics
            unique, counts = np.unique(phases, return_counts=True)
            print("\n  Phase distribution:")
            for u, c in zip(unique, counts):
                pname = PHASE_NAMES.get(int(u), f"Phase {int(u)}")
                print(f"    {pname}: {c} steps ({c / len(phases) * 100:.1f}%)")

            print(f"\n  Gripper closed ratio:")
            for idx, name in enumerate(agent_names):
                print(f"    {name}: {gripper_arrays[idx].mean():.3f}")

            # Generate videos
            for cam in camera_names:
                cam_path = f"{traj_key}/obs/sensor_data/{cam}/rgb"
                if cam_path not in f:
                    print(f"  Camera '{cam}' not found, skipping.")
                    continue

                rgb_data = f[cam_path][:]
                print(f"\n  Creating video for {cam}  (shape: {rgb_data.shape})")

                try:
                    create_phase_video(
                        rgb_data, phases, gripper_states,
                        traj_key, cam,
                        output_dir=output_dir, fps=fps,
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
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_132_with_phases.h5"
    output_dir = "/18141169908/zhangzaibin/multivla-pi0/phase_videos_pick_place_plate_132"

    print("=" * 60)
    print("Phase Video Visualization — Pick & Place to Plate (132, phases 11-17)")
    print("=" * 60)
    print_h5_structure(file_path, "traj_0")

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



















