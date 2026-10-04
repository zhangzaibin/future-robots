"""
Phase visualization VIDEO generator for two_arms_place_can_basket dataset.

4-phase scheme based on gripper state transitions.
Per-arm atomic instructions:
  Arm0 (panda-0): Stay -> Approach & Grasp can -> Carry & Place can -> Release & Return
  Arm1 (panda-1): Approach & Grasp basket -> Hold basket -> Hold basket -> Hold basket

Usage:
  python scripts/set_atom/macg/two_arms_place_can_basket/visualize_phases_place_can_basket.py
"""

import argparse
from pathlib import Path

import cv2
import h5py
import imageio.v3 as iio
import numpy as np

PHASE_INFO = {
    1: {"name": "Grasp Basket",  "agent0": "Stay still.",                   "agent1": "Approach and grasp the basket.", "color_bgr": (100, 100, 255)},
    2: {"name": "Grasp Can",     "agent0": "Approach and grasp the can.",    "agent1": "Hold the basket.",               "color_bgr": (50, 180, 255)},
    3: {"name": "Place Can",     "agent0": "Place the can into the basket.", "agent1": "Hold the basket.",               "color_bgr": (50, 220, 100)},
    4: {"name": "Return",        "agent0": "Release and return to rest.",    "agent1": "Hold the basket.",               "color_bgr": (200, 150, 50)},
}
NUM_PHASES = 4


def draw_info_panel(width, frame_idx, n_frames, phase_id, a0_grip, a1_grip):
    h = 170
    panel = np.ones((h, width, 3), dtype=np.uint8) * 40
    ph = PHASE_INFO.get(phase_id, {"name": "?", "agent0": "?", "agent1": "?", "color_bgr": (128, 128, 128)})
    c = ph["color_bgr"]

    y = 25
    cv2.putText(panel, f"Phase {phase_id}: {ph['name']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    cv2.putText(panel, f"Frame {frame_idx}/{n_frames-1}", (width - 250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    y = 55
    c0 = c if ph["agent0"] != "Stay still." else (140, 140, 140)
    cv2.putText(panel, f"Agent0 (panda-0): {ph['agent0']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c0, 1, cv2.LINE_AA)
    y = 80
    c1 = c if ph["agent1"] != "Hold the basket." else (140, 140, 140)
    cv2.putText(panel, f"Agent1 (panda-1): {ph['agent1']}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c1, 1, cv2.LINE_AA)

    y = 110
    a0o = a0_grip > 0.03
    gc0 = (0, 200, 0) if a0o else (0, 0, 220)
    cv2.putText(panel, f"A0 Gripper: {'OPEN' if a0o else 'CLOSED'} ({a0_grip:.4f})", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gc0, 1, cv2.LINE_AA)
    cv2.circle(panel, (310, y - 5), 7, gc0, -1)

    a1o = a1_grip > 0.03
    gc1 = (0, 200, 0) if a1o else (0, 0, 220)
    xo = max(width // 2, 400)
    cv2.putText(panel, f"A1 Gripper: {'OPEN' if a1o else 'CLOSED'} ({a1_grip:.4f})", (xo, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gc1, 1, cv2.LINE_AA)
    cv2.circle(panel, (xo + 295, y - 5), 7, gc1, -1)

    y = 135
    bx, bw, bh = 15, width - 30, 22
    cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), (80, 80, 80), -1)
    sw = bw / NUM_PHASES
    for pid in range(1, NUM_PHASES + 1):
        x1, x2 = int(bx + (pid - 1) * sw), int(bx + pid * sw)
        pc = PHASE_INFO[pid]["color_bgr"]
        if pid < phase_id:
            cv2.rectangle(panel, (x1, y), (x2, y + bh), pc, -1)
        elif pid == phase_id:
            cv2.rectangle(panel, (x1, y), (x2, y + bh), pc, -1)
            cv2.rectangle(panel, (x1, y), (x2, y + bh), (255, 255, 255), 2)
        else:
            cv2.rectangle(panel, (x1, y), (x2, y + bh), tuple(v // 3 for v in pc), -1)
        cv2.putText(panel, f"P{pid}", (int(x1 + sw / 2 - 10), y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if n_frames > 1:
        px = int(bx + (frame_idx / (n_frames - 1)) * bw)
        cv2.line(panel, (px, y - 3), (px, y + bh + 3), (255, 255, 255), 2)
    return panel


def create_phase_video(h5f, traj_key, phases, output_dir, fps=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a0q = h5f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
    a1q = h5f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
    hcam = h5f[f"{traj_key}/obs/sensor_data/head_camera_global/rgb"]
    w0 = h5f[f"{traj_key}/obs/sensor_data/wrist_camera_agent0/rgb"]
    w1 = h5f[f"{traj_key}/obs/sensor_data/wrist_camera_agent1/rgb"]

    n = min(len(phases), len(hcam), len(w0), len(w1))
    ch, cw = hcam[0].shape[:2]
    tw = cw * 3

    opath = output_dir / f"{traj_key}_phases.mp4"

    labels = ["Head Camera (Global)", "Wrist Camera Agent0", "Wrist Camera Agent1"]
    cams = [hcam, w0, w1]
    all_frames = []

    for i in range(n):
        pid = int(phases[i])
        pc = PHASE_INFO.get(pid, {}).get("color_bgr", (128, 128, 128))
        row = []
        for ci, cam in enumerate(cams):
            img = cv2.cvtColor(cam[i], cv2.COLOR_RGB2BGR)
            if img.shape[:2] != (ch, cw):
                img = cv2.resize(img, (cw, ch))
            ov = img.copy()
            ov[:] = pc
            img = cv2.addWeighted(img, 0.82, ov, 0.18, 0)
            cv2.putText(img, labels[ci], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, labels[ci], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            pt = f"P{pid}: {PHASE_INFO[pid]['name']}"
            cv2.putText(img, pt, (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, pt, (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pc, 1, cv2.LINE_AA)
            row.append(img)
        frame = np.hstack(row)
        panel = draw_info_panel(tw, i, n, pid, a0q[i, -1], a1q[i, -1])
        combined = cv2.cvtColor(np.vstack([frame, panel]), cv2.COLOR_BGR2RGB)
        all_frames.append(combined)
        if (i + 1) % 200 == 0:
            print(f"    Frame {i+1}/{n}")

    print(f"  Writing H.264 video ({len(all_frames)} frames)...")
    writer = iio.imopen(str(opath), "w", plugin="pyav")
    writer.write(all_frames, codec="h264", fps=fps)
    writer.close()
    print(f"  Video saved: {opath}")


def process_trajectory(h5_path, traj_idx, output_dir, fps=30):
    tk = f"traj_{traj_idx}"
    with h5py.File(h5_path, "r") as f:
        if tk not in f:
            print(f"  ERROR: {tk} not found")
            return
        if f"{tk}/phase" not in f:
            print(f"  ERROR: {tk}/phase not found")
            return
        phases = f[f"{tk}/phase"][:]
        print(f"\nProcessing {tk} ({len(phases)} steps)...")
        for pid in range(1, NUM_PHASES + 1):
            m = phases == pid
            if m.any():
                s, e = int(np.where(m)[0][0]), int(np.where(m)[0][-1])
                info = PHASE_INFO[pid]
                print(f"  Phase {pid} ({info['name']:>14}): steps {s:4d}-{e:4d} ({m.sum():3d} steps) | A0: {info['agent0']:<35s} | A1: {info['agent1']}")
        create_phase_video(f, tk, phases, output_dir, fps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path", type=str, default="/18141169908/zhangzaibin/multivla-pi0/dataset/two_arms_place_can_basket_with_phases.h5")
    p.add_argument("--output_dir", type=str, default="visualizations/place_can_basket_phase_videos")
    p.add_argument("--trajs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    print(f"H5: {a.h5_path}\nOutput: {a.output_dir}\nTrajs: {a.trajs}\nFPS: {a.fps}")
    for idx in a.trajs:
        process_trajectory(a.h5_path, idx, a.output_dir, a.fps)
    print(f"\nDone! Videos saved to {a.output_dir}/")
