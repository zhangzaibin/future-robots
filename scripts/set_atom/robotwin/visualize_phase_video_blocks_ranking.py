"""
Phase visualization VIDEO generator for RobotWin blocks_ranking_rgb dataset (LeRobot format).

Creates MP4 videos with:
  - Camera views (cam_high, cam_left_wrist, cam_right_wrist) side by side
  - Phase color overlay on the image
  - Info panel showing: arm-wise instruction, gripper states, active arm, progress bar, frame counter

Arm-wise instruction format:
  "{arm0(left) instruction} | {arm1(right) instruction}"
  Active arm gets action instruction, inactive arm gets "Wait for the other robots."

Usage:
  python scripts/set_atom/robotwin/visualize_phase_video_blocks_ranking.py \
    --dataset_dir dataset/robotwin_lerobot/blocks_ranking_rgb_phase \
    --output_dir visualizations/blocks_ranking_phase_videos \
    --episodes 0 1 2 3 4 \
    --fps 25 \
    --camera all
"""

import argparse
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# Phase definitions (action instruction for the active arm)
PHASE_INFO = {
    1: {"name": "Pick Red",    "action": "Pick up the red block.",                          "color_bgr": (120, 120, 255)},
    2: {"name": "Place Red",   "action": "Place the red block on the left position.",       "color_bgr": (0, 0, 200)},
    3: {"name": "Pick Green",  "action": "Pick up the green block.",                        "color_bgr": (120, 255, 120)},
    4: {"name": "Place Green", "action": "Place the green block in the middle position.",   "color_bgr": (0, 160, 0)},
    5: {"name": "Pick Blue",   "action": "Pick up the blue block.",                         "color_bgr": (255, 180, 120)},
    6: {"name": "Place Blue",  "action": "Place the blue block on the right position.",     "color_bgr": (200, 80, 0)},
}

WAIT_INSTRUCTION = "Wait for the other robots."

# Camera names in the dataset
CAMERA_NAMES = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


def decode_image(img_dict):
    """Decode image from LeRobot parquet dict format {'bytes': ..., 'path': ...}."""
    img = Image.open(io.BytesIO(img_dict['bytes']))
    return np.array(img)  # RGB, (H, W, 3)


def detect_phase_active_arms(left_gripper, right_gripper, threshold=0.5):
    """
    Detect which arm is active for each phase from gripper transitions.

    Returns:
        phase_active_arms: dict mapping phase_id (1-6) to 'left' or 'right'
    """
    left_binary = (left_gripper > threshold).astype(int)
    right_binary = (right_gripper > threshold).astype(int)

    events = []
    for f in np.where(np.diff(left_binary) == -1)[0]:
        events.append((int(f), 'close', 'left'))
    for f in np.where(np.diff(left_binary) == 1)[0]:
        events.append((int(f), 'open', 'left'))
    for f in np.where(np.diff(right_binary) == -1)[0]:
        events.append((int(f), 'close', 'right'))
    for f in np.where(np.diff(right_binary) == 1)[0]:
        events.append((int(f), 'open', 'right'))
    events.sort(key=lambda x: x[0])

    if len(events) != 6:
        return {p: "right" for p in range(1, 7)}  # fallback

    block1_arm = events[0][2]
    block2_arm = events[2][2]
    block3_arm = events[4][2]

    return {
        1: block1_arm, 2: block1_arm,
        3: block2_arm, 4: block2_arm,
        5: block3_arm, 6: block3_arm,
    }


def get_arm_wise_instruction(phase_id, active_arm):
    """Build arm-wise instruction string."""
    action = PHASE_INFO[phase_id]["action"]
    if active_arm == "left":
        return f"{action} | {WAIT_INSTRUCTION}"
    else:
        return f"{WAIT_INSTRUCTION} | {action}"


def draw_info_panel(width, frame_idx, n_frames, phase_id, left_gripper_val, right_gripper_val,
                    active_arm, phase_active_arms):
    """
    Draw an info panel (below the image) with arm-wise instruction, gripper states, and progress bar.

    Returns:
        info_panel: numpy array (H_panel, width, 3), BGR format
    """
    panel_height = 160
    panel = np.ones((panel_height, width, 3), dtype=np.uint8) * 40  # dark gray background

    phase = PHASE_INFO.get(phase_id, {"name": "Unknown", "action": "Unknown", "color_bgr": (128, 128, 128)})
    color = phase["color_bgr"]

    # --- Row 1: Phase name + active arm + frame counter ---
    y = 25
    arm_label = "arm0(L)" if active_arm == "left" else "arm1(R)"
    cv2.putText(panel, f"Phase {phase_id}: {phase['name']}  [Active: {arm_label}]", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"Frame {frame_idx}/{n_frames - 1}", (width - 220, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Row 2: Arm-wise instruction ---
    y = 52
    arm_wise_task = get_arm_wise_instruction(phase_id, active_arm)
    # Split at | to color each arm's instruction differently
    parts = arm_wise_task.split(" | ")
    arm0_instr = parts[0]
    arm1_instr = parts[1] if len(parts) > 1 else ""

    # arm0 (left) instruction
    arm0_color = color if active_arm == "left" else (140, 140, 140)
    cv2.putText(panel, f"arm0(L): {arm0_instr}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, arm0_color, 1, cv2.LINE_AA)
    # arm1 (right) instruction
    y = 75
    arm1_color = color if active_arm == "right" else (140, 140, 140)
    cv2.putText(panel, f"arm1(R): {arm1_instr}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, arm1_color, 1, cv2.LINE_AA)

    # --- Row 3: Gripper states ---
    y = 105
    # Left gripper (arm0)
    left_open = left_gripper_val > 0.5
    left_color = (0, 200, 0) if left_open else (0, 0, 220)
    left_status = "OPEN" if left_open else "CLOSED"
    # Highlight active arm with a star
    left_marker = " *" if active_arm == "left" else ""
    left_text = f"arm0(L) Gripper: {left_status} ({left_gripper_val:.2f}){left_marker}"
    cv2.putText(panel, left_text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1, cv2.LINE_AA)
    cv2.circle(panel, (370, y - 5), 7, left_color, -1)

    # Right gripper (arm1)
    right_open = right_gripper_val > 0.5
    right_color = (0, 200, 0) if right_open else (0, 0, 220)
    right_status = "OPEN" if right_open else "CLOSED"
    right_marker = " *" if active_arm == "right" else ""
    right_text = f"arm1(R) Gripper: {right_status} ({right_gripper_val:.2f}){right_marker}"
    x_offset = max(width // 2, 400)
    cv2.putText(panel, right_text, (x_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1, cv2.LINE_AA)
    cv2.circle(panel, (x_offset + 370, y - 5), 7, right_color, -1)

    # --- Row 4: Phase progress bar ---
    y = 133
    bar_x = 15
    bar_w = width - 30
    bar_h = 18

    # Background
    cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (80, 80, 80), -1)

    # Draw 6 phase segments
    segment_w = bar_w / 6
    for pid in range(1, 7):
        x1 = int(bar_x + (pid - 1) * segment_w)
        x2 = int(bar_x + pid * segment_w)
        p_color = PHASE_INFO[pid]["color_bgr"]
        p_arm = phase_active_arms.get(pid, "?")
        p_arm_label = "L" if p_arm == "left" else "R"

        if pid <= phase_id:
            cv2.rectangle(panel, (x1, y), (x2, y + bar_h), p_color, -1)
        else:
            dim_color = tuple(c // 3 for c in p_color)
            cv2.rectangle(panel, (x1, y), (x2, y + bar_h), dim_color, -1)

        if pid == phase_id:
            cv2.rectangle(panel, (x1, y), (x2, y + bar_h), (255, 255, 255), 2)

        # Phase label with arm
        label = f"P{pid}({p_arm_label})"
        text_x = int(x1 + segment_w / 2 - 22)
        cv2.putText(panel, label, (text_x, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return panel


def create_phase_video_single_camera(
    df, camera_col, phases, left_gripper, right_gripper,
    phase_active_arms, ep_idx, camera_name, output_dir, fps=25
):
    """
    Create a phase visualization video for a single camera view with arm-wise instructions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(df)

    # Decode first frame to get dimensions
    first_img = decode_image(df[camera_col].iloc[0])
    h, w = first_img.shape[:2]

    panel_h = 160
    total_h = h + panel_h

    output_path = output_dir / f"episode_{ep_idx:03d}_{camera_name}_phases.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, total_h))

    for i in range(n_frames):
        # Decode image
        img_rgb = decode_image(df[camera_col].iloc[i])
        phase_id = int(phases[i])
        active_arm = phase_active_arms[phase_id]
        phase_color_bgr = PHASE_INFO.get(phase_id, {}).get("color_bgr", (128, 128, 128))

        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Add phase color overlay (semi-transparent border/tint)
        overlay = img_bgr.copy()
        overlay[:] = phase_color_bgr
        img_bgr = cv2.addWeighted(img_bgr, 0.8, overlay, 0.2, 0)

        # Add phase label + active arm on the image
        arm_label = "L" if active_arm == "left" else "R"
        text = f"P{phase_id}: {PHASE_INFO[phase_id]['name']} [{arm_label}]"
        cv2.putText(img_bgr, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color_bgr, 1, cv2.LINE_AA)

        # Draw info panel
        info_panel = draw_info_panel(
            w, i, n_frames, phase_id,
            left_gripper[i], right_gripper[i],
            active_arm, phase_active_arms,
        )

        # Combine
        combined = np.vstack([img_bgr, info_panel])
        out.write(combined)

    out.release()
    print(f"  Video saved: {output_path}")
    return str(output_path)


def create_phase_video_multi_camera(
    df, phases, left_gripper, right_gripper,
    phase_active_arms, ep_idx, output_dir, fps=25
):
    """
    Create a phase visualization video with all 3 camera views side by side.
    Layout: [cam_high | cam_left_wrist | cam_right_wrist] + info panel below.
    Shows arm-wise instructions in the info panel.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(df)

    # Decode first frames to get dimensions
    first_imgs = {}
    for cam_name, col_name in CAMERA_NAMES.items():
        first_imgs[cam_name] = decode_image(df[col_name].iloc[0])

    # Use cam_high as reference size
    ref_h, ref_w = first_imgs["cam_high"].shape[:2]
    cam_h = ref_h
    cam_w = ref_w

    total_w = cam_w * 3  # 3 cameras side by side
    panel_h = 160
    total_h = cam_h + panel_h

    output_path = output_dir / f"episode_{ep_idx:03d}_all_cameras_phases.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (total_w, total_h))

    cam_labels = ["Cam High", "Cam Left Wrist (arm0)", "Cam Right Wrist (arm1)"]
    cam_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    for i in range(n_frames):
        phase_id = int(phases[i])
        active_arm = phase_active_arms[phase_id]
        phase_color_bgr = PHASE_INFO.get(phase_id, {}).get("color_bgr", (128, 128, 128))

        frames_row = []
        for ci, cam_name in enumerate(cam_keys):
            col_name = CAMERA_NAMES[cam_name]
            img_rgb = decode_image(df[col_name].iloc[i])
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Resize if needed
            if img_bgr.shape[0] != cam_h or img_bgr.shape[1] != cam_w:
                img_bgr = cv2.resize(img_bgr, (cam_w, cam_h))

            # Add phase color overlay
            overlay = img_bgr.copy()
            overlay[:] = phase_color_bgr
            img_bgr = cv2.addWeighted(img_bgr, 0.8, overlay, 0.2, 0)

            # Highlight the active arm's camera with a thicker border
            is_active_cam = (cam_name == "cam_left_wrist" and active_arm == "left") or \
                            (cam_name == "cam_right_wrist" and active_arm == "right")
            if is_active_cam:
                cv2.rectangle(img_bgr, (2, 2), (cam_w - 3, cam_h - 3), (0, 255, 255), 3)

            # Camera label
            label = cam_labels[ci]
            if is_active_cam:
                label += " *ACTIVE*"
            cv2.putText(img_bgr, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            frames_row.append(img_bgr)

        # Concatenate cameras horizontally
        cam_frame = np.hstack(frames_row)

        # Info panel
        info_panel = draw_info_panel(
            total_w, i, n_frames, phase_id,
            left_gripper[i], right_gripper[i],
            active_arm, phase_active_arms,
        )

        # Combine
        combined = np.vstack([cam_frame, info_panel])
        out.write(combined)

    out.release()
    print(f"  Multi-camera video saved: {output_path}")
    return str(output_path)


def process_episode(dataset_dir, ep_idx, output_dir, camera="all", fps=25):
    """Process one episode and generate visualization video(s) with arm-wise info."""
    parquet_path = Path(dataset_dir) / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    if not parquet_path.exists():
        print(f"  ERROR: {parquet_path} not found")
        return

    print(f"Processing episode {ep_idx}...")
    df = pd.read_parquet(str(parquet_path))

    states = np.array(df['observation.state'].tolist())
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]
    phases = df['phase'].values

    # Detect which arm is active for each phase
    phase_active_arms = detect_phase_active_arms(left_gripper, right_gripper)
    arm_pattern = "/".join(phase_active_arms[p][0].upper() for p in [1, 3, 5])

    # Print phase summary with arm info
    print(f"  Arm pattern: {arm_pattern} (Red/Green/Blue)")
    for pid in range(1, 7):
        mask = phases == pid
        if mask.any():
            start = np.where(mask)[0][0]
            end = np.where(mask)[0][-1]
            arm = phase_active_arms[pid]
            arm_label = "arm0(L)" if arm == "left" else "arm1(R)"
            instr = get_arm_wise_instruction(pid, arm)
            print(f"  Phase {pid} ({PHASE_INFO[pid]['name']:12s}) [{arm_label}]: "
                  f"frames {start:4d}-{end:4d} ({mask.sum():3d} frames)")
            print(f"         Task: {instr}")

    if camera == "all":
        create_phase_video_multi_camera(
            df, phases, left_gripper, right_gripper,
            phase_active_arms, ep_idx, output_dir, fps
        )
    elif camera == "each":
        for cam_name, col_name in CAMERA_NAMES.items():
            create_phase_video_single_camera(
                df, col_name, phases, left_gripper, right_gripper,
                phase_active_arms, ep_idx, cam_name, output_dir, fps
            )
    else:
        if camera in CAMERA_NAMES:
            col_name = CAMERA_NAMES[camera]
            create_phase_video_single_camera(
                df, col_name, phases, left_gripper, right_gripper,
                phase_active_arms, ep_idx, camera, output_dir, fps
            )
        else:
            print(f"  ERROR: Unknown camera '{camera}'. Available: {list(CAMERA_NAMES.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate phase visualization videos for blocks_ranking_rgb dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/robotwin_lerobot/blocks_ranking_rgb_phase",
        help="Path to phase-labeled LeRobot dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations/blocks_ranking_phase_videos",
        help="Directory for output videos",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Episode indices to visualize (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Video frames per second (default: 25)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="all",
        choices=["all", "each", "cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="Camera mode: 'all' = 3 cameras side by side, 'each' = separate video per camera, "
             "or specific camera name (default: all)",
    )

    args = parser.parse_args()

    print(f"Dataset: {args.dataset_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"FPS: {args.fps}")
    print(f"Camera: {args.camera}")
    print()

    for ep_idx in args.episodes:
        process_episode(
            args.dataset_dir, ep_idx, args.output_dir,
            camera=args.camera, fps=args.fps
        )

    print(f"\nDone! Videos saved to {args.output_dir}/")

