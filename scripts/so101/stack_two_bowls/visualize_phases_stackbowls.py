"""
Visualize stackbowls_v2 episodes as videos with phase overlay.

Each frame shows the 3 camera views (global, left wrist, right wrist)
side by side, with a colored border and text indicating the current phase
and task instruction.

Usage:
    # Visualize specific episodes
    python scripts/so101/visualize_phases_stackbowls.py --episodes 0,1,5,10

    # Visualize all episodes
    python scripts/so101/visualize_phases_stackbowls.py --all

    # Faster output (skip frames)
    python scripts/so101/visualize_phases_stackbowls.py --episodes 0 --step 3
"""

import argparse
import io
import json
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DATASET_DIR = "/18141169908/zhangzaibin/multivla-pi0/dataset/realworld/stackbowls_v2"
DATA_DIR = os.path.join(DATASET_DIR, "data", "chunk-000")
OUTPUT_DIR = os.path.join(DATASET_DIR, "phase_videos")

PHASE_COLORS = {
    1: (0, 140, 255),   # orange (BGR) - arm2 active
    2: (255, 140, 0),   # blue (BGR)  - arm1 active
}

PHASE_LABELS = {
    1: "Phase 1: arm2 active",
    2: "Phase 2: arm1 active",
}


def load_tasks():
    tasks = {}
    with open(os.path.join(DATASET_DIR, "meta", "tasks.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                tasks[obj["task_index"]] = obj["task"]
    return tasks


def decode_image(img_struct):
    raw = img_struct["bytes"]
    pil_img = Image.open(io.BytesIO(raw))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_text_with_bg(frame, text, pos, font_scale=0.55, color=(255, 255, 255),
                     bg_color=(0, 0, 0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def visualize_episode(ep_id, tasks, step=1):
    path = os.path.join(DATA_DIR, "episode_{:06d}.parquet".format(ep_id))
    table = pq.read_table(path)
    n = table.num_rows

    img_col = table.column("image")
    wrist_col = table.column("wrist_image")
    agent1_col = table.column("image_agent1")
    phase_col = table.column("phase")
    task_idx_col = table.column("task_index")
    state_col = table.column("state")

    # Decode first frame to get dimensions
    sample = decode_image(img_col[0].as_py())
    h, w = sample.shape[:2]

    # Canvas: 3 images side by side + info bar at bottom
    canvas_w = w * 3
    info_h = 60
    canvas_h = h + info_h

    out_path = os.path.join(OUTPUT_DIR, "phase_ep{:03d}.mp4".format(ep_id))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30 // step
    writer = cv2.VideoWriter(out_path, fourcc, max(fps, 1), (canvas_w, canvas_h))

    for i in range(0, n, step):
        phase_val = int(phase_col[i].as_py()[0])
        task_idx = int(task_idx_col[i].as_py())
        task_str = tasks.get(task_idx, "?")

        # Decode 3 camera views
        img_global = decode_image(img_col[i].as_py())
        img_wrist_l = decode_image(wrist_col[i].as_py())
        img_wrist_r = decode_image(agent1_col[i].as_py())

        # Resize to same height if needed
        img_global = cv2.resize(img_global, (w, h))
        img_wrist_l = cv2.resize(img_wrist_l, (w, h))
        img_wrist_r = cv2.resize(img_wrist_r, (w, h))

        # Concatenate horizontally
        row = np.concatenate([img_global, img_wrist_l, img_wrist_r], axis=1)

        # Add colored border based on phase
        border_color = PHASE_COLORS.get(phase_val, (128, 128, 128))
        cv2.rectangle(row, (0, 0), (canvas_w - 1, h - 1), border_color, 3)

        # Create info bar
        info_bar = np.zeros((info_h, canvas_w, 3), dtype=np.uint8)
        info_bar[:] = border_color

        # Gripper values
        state = np.array(state_col[i].as_py())
        arm1_g = state[5]
        arm2_g = state[11]

        # Text overlay
        phase_label = PHASE_LABELS.get(phase_val, "Phase ?")
        put_text_with_bg(info_bar,
                         "EP{:02d} | Step {}/{} | {} | arm1_grip={:.1f} arm2_grip={:.1f}".format(
                             ep_id, i, n, phase_label, arm1_g, arm2_g),
                         (8, 18), font_scale=0.5, color=(255, 255, 255), bg_color=border_color)
        put_text_with_bg(info_bar,
                         "Task: {}".format(task_str),
                         (8, 44), font_scale=0.5, color=(255, 255, 255), bg_color=border_color)

        # Camera labels
        put_text_with_bg(row, "Global", (8, 20), font_scale=0.45)
        put_text_with_bg(row, "Left Wrist (arm1)", (w + 8, 20), font_scale=0.45)
        put_text_with_bg(row, "Right Wrist (arm2)", (w * 2 + 8, 20), font_scale=0.45)

        canvas = np.concatenate([row, info_bar], axis=0)
        writer.write(canvas)

    writer.release()
    print("ep {:3d} | {} steps | saved: {}".format(ep_id, n, out_path))


def main():
    parser = argparse.ArgumentParser(description="Visualize phase-labeled stackbowls_v2 episodes")
    parser.add_argument("--episodes", type=str, default="0,1,5,10",
                        help="Comma-separated episode IDs (default: 0,1,5,10)")
    parser.add_argument("--all", action="store_true", help="Process all episodes")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame step (1=every frame, 3=every 3rd frame)")
    args = parser.parse_args()

    tasks = load_tasks()
    print("Tasks loaded:")
    for idx, task in sorted(tasks.items()):
        print("  {}: '{}'".format(idx, task))
    print()

    if args.all:
        parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
        ep_ids = [int(f.split("_")[1].split(".")[0]) for f in parquet_files]
    else:
        ep_ids = [int(x) for x in args.episodes.split(",")]

    for ep_id in ep_ids:
        visualize_episode(ep_id, tasks, step=args.step)

    print("\nDone! Videos saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
