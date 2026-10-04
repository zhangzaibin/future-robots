"""
Detect and write 2-phase labels for the place_two_cubes_v2 dataset.

Phase 1: arm2 (right) is active, arm1 (left) waits
Phase 2: arm1 (left) is active, arm2 (right) waits

Each arm performs 2 gripper open-close cycles (placing 2 cubes sequentially).
The two arms act in strict sequence with a clear idle gap in between.

Detection strategy:
  - Compute smoothed joint velocity for both arms.
  - Find arm2's last active step (vel > threshold) and arm1's first active step.
  - The boundary is the midpoint of the idle gap between them.
  - Fallback: if arm1 starts before arm2 ends (overlap), use arm1's first
    gripper open transition and look back for movement start (stackbowls method).

Usage:
    python scripts/so101/place_two_cubes/set_phases_placecubes.py --dry-run --visualize
    python scripts/so101/place_two_cubes/set_phases_placecubes.py
    python scripts/so101/place_two_cubes/set_phases_placecubes.py --episodes 0,1,5
"""

import argparse
import json
import os

import numpy as np

DATASET_DIR = "/18141169908/zhangzaibin/multivla-pi0/dataset/realworld/place_two_cubes_v2"
DATA_DIR = os.path.join(DATASET_DIR, "data", "chunk-000")

ARM1_GRIPPER_IDX = 5
ARM2_GRIPPER_IDX = 11

ARM1_JOINT_SLICE = slice(0, 5)
ARM2_JOINT_SLICE = slice(6, 11)

JOINT_VEL_THRESH = 1.0
SMOOTH_WINDOW = 15

ARM1_OPEN_THRESH = 10.0
ARM1_CLOSED_THRESH = 5.0


def load_episode(ep_id: int):
    import pyarrow.parquet as pq

    path = os.path.join(DATA_DIR, "episode_{:06d}.parquet".format(ep_id))
    return pq.read_table(path)


def extract_data(table):
    state_col = table.column("state")
    n = table.num_rows
    states = np.array([np.array(state_col[i].as_py()) for i in range(n)])
    arm1_g = states[:, ARM1_GRIPPER_IDX]
    arm2_g = states[:, ARM2_GRIPPER_IDX]
    arm1_joints = states[:, ARM1_JOINT_SLICE]
    arm2_joints = states[:, ARM2_JOINT_SLICE]
    return arm1_g, arm2_g, arm1_joints, arm2_joints


def smooth(x, w=SMOOTH_WINDOW):
    return np.convolve(x, np.ones(w) / w, mode="same")


def compute_joint_vel(joints, w=SMOOTH_WINDOW):
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    return smooth(vel, w)


def find_arm1_first_open_transition(arm1_g):
    """Fallback: find arm1's first close->open gripper transition."""
    n = len(arm1_g)
    was_closed = arm1_g[0] < ARM1_CLOSED_THRESH
    for i in range(1, n):
        if arm1_g[i] < ARM1_CLOSED_THRESH:
            was_closed = True
        if was_closed and arm1_g[i] > ARM1_OPEN_THRESH:
            return i
    return None


def find_movement_start(joint_vel, anchor):
    """Look backwards from anchor to find when velocity first exceeded threshold."""
    for i in range(min(anchor, len(joint_vel) - 1), -1, -1):
        if joint_vel[i] < JOINT_VEL_THRESH:
            return min(i + 1, anchor)
    return 0


def detect_phases(arm1_g, arm2_g, arm1_joints, arm2_joints):
    n = len(arm1_g)
    debug = {}

    arm1_vel = compute_joint_vel(arm1_joints)
    arm2_vel = compute_joint_vel(arm2_joints)

    arm2_active = np.where(arm2_vel > JOINT_VEL_THRESH)[0]
    arm1_active = np.where(arm1_vel > JOINT_VEL_THRESH)[0]

    arm2_end = int(arm2_active[-1]) if len(arm2_active) > 0 else 0
    arm1_start = int(arm1_active[0]) if len(arm1_active) > 0 else n

    debug["arm2_end"] = arm2_end
    debug["arm1_start"] = arm1_start
    debug["gap"] = arm1_start - arm2_end

    if arm1_start > arm2_end:
        boundary = (arm2_end + arm1_start) // 2
        debug["method"] = "gap_midpoint"
    else:
        # Fallback: use arm1 gripper transition + lookback (stackbowls method)
        arm1_open = find_arm1_first_open_transition(arm1_g)
        debug["arm1_open_transition"] = arm1_open
        if arm1_open is not None:
            boundary = find_movement_start(arm1_vel, arm1_open)
            debug["method"] = "gripper_lookback"
        else:
            debug["warning"] = "arm1 never opens; assigning all to Phase 1"
            debug["method"] = "fallback_all_phase1"
            boundary = n - 1

    boundary = max(1, min(boundary, n - 1))
    debug["boundary"] = boundary

    phases = np.ones(n, dtype=np.int32)
    phases[boundary:] = 2

    return phases, boundary, debug


def visualize_episode(ep_id, arm1_g, arm2_g, arm1_joints, arm2_joints, phases, boundary, debug, save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(arm1_g)
    t = np.arange(n)
    arm1_vel = compute_joint_vel(arm1_joints)
    arm2_vel = compute_joint_vel(arm2_joints)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, arm2_g, "r-", linewidth=1.5, label="arm2 (right) gripper")
    axes[0].axvline(x=boundary, color="k", linestyle="--", linewidth=2, label="boundary")
    axes[0].set_ylabel("arm2 gripper")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Episode {} | boundary @ step {} ({})".format(
        ep_id, boundary, debug.get("method", "")))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, arm1_g, "b-", linewidth=1.5, label="arm1 (left) gripper")
    axes[1].axhline(y=ARM1_OPEN_THRESH, color="b", linestyle="--", alpha=0.5, label="open threshold")
    axes[1].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    axes[1].set_ylabel("arm1 gripper")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.arange(len(arm2_vel)), arm2_vel, "red", linewidth=1, label="arm2 joint vel")
    axes[2].axhline(y=JOINT_VEL_THRESH, color="red", linestyle="--", alpha=0.5)
    axes[2].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    if debug.get("arm2_end") is not None:
        axes[2].axvline(x=debug["arm2_end"], color="red", linestyle=":", linewidth=1.5,
                        label="arm2_end @{}".format(debug["arm2_end"]))
    axes[2].set_ylabel("arm2 joint vel")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(np.arange(len(arm1_vel)), arm1_vel, "purple", linewidth=1, label="arm1 joint vel")
    axes[3].axhline(y=JOINT_VEL_THRESH, color="purple", linestyle="--", alpha=0.5)
    axes[3].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    if debug.get("arm1_start") is not None:
        axes[3].axvline(x=debug["arm1_start"], color="purple", linestyle=":", linewidth=1.5,
                        label="arm1_start @{}".format(debug["arm1_start"]))
    axes[3].set_ylabel("arm1 joint vel")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    colors = {1: "tab:orange", 2: "tab:blue"}
    for phase_val in [1, 2]:
        mask = phases == phase_val
        axes[4].fill_between(t, 0, 1, where=mask, alpha=0.5,
                             color=colors[phase_val], label="Phase {}".format(phase_val))
    axes[4].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    axes[4].set_ylabel("Phase")
    axes[4].set_xlabel("Time Step")
    axes[4].set_yticks([])
    axes[4].legend(loc="upper right")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "phase_ep{:03d}.png".format(ep_id))
        plt.savefig(path, dpi=120, bbox_inches="tight")
        print("  Saved: {}".format(path))
    plt.close(fig)


def write_phases_to_parquet(ep_id, phases):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = os.path.join(DATA_DIR, "episode_{:06d}.parquet".format(ep_id))
    table = pq.read_table(path)

    if "phase" in table.column_names:
        table = table.drop("phase")

    phase_col = pa.array([float(p) for p in phases], type=pa.float32())
    table = table.append_column("phase", phase_col)
    pq.write_table(table, path)


def update_task_index_in_parquet(ep_id, phases):
    """Update task_index to match phase (phase 1 -> task_index 1, phase 2 -> task_index 2)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = os.path.join(DATA_DIR, "episode_{:06d}.parquet".format(ep_id))
    table = pq.read_table(path)

    if "task_index" in table.column_names:
        table = table.drop("task_index")

    task_indices = pa.array([int(p) for p in phases], type=pa.int64())
    table = table.append_column("task_index", task_indices)
    pq.write_table(table, path)


def update_info_json(dataset_dir):
    """Add phase feature declaration to meta/info.json."""
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    changed = False

    if "phase" not in info["features"]:
        info["features"]["phase"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        changed = True

    if info.get("total_tasks", 1) < 3:
        info["total_tasks"] = 3
        changed = True

    if changed:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print("Updated meta/info.json.")


def update_tasks_jsonl(dataset_dir):
    """Create/update tasks.jsonl with phase-specific prompts."""
    tasks_path = os.path.join(dataset_dir, "meta", "tasks.jsonl")
    tasks = [
        {"task_index": 0, "task": "Place the cubes."},
        {"task_index": 1, "task": "Place the cubes. | Wait for the other robots."},
        {"task_index": 2, "task": "Wait for the other robots. | Place the cubes."},
    ]
    with open(tasks_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print("Updated meta/tasks.jsonl with {} tasks.".format(len(tasks)))


def main():
    parser = argparse.ArgumentParser(description="Set phases for place_two_cubes_v2 dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--dry-run", action="store_true", help="Detect only, don't write")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode IDs (default: all)")
    parser.add_argument("--no-task-index", action="store_true",
                        help="Skip updating task_index column")
    args = parser.parse_args()

    parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
    all_ep_ids = [int(f.split("_")[1].split(".")[0]) for f in parquet_files]

    ep_ids = [int(x) for x in args.episodes.split(",")] if args.episodes else all_ep_ids
    vis_dir = os.path.join(DATASET_DIR, "phase_plots") if args.visualize else None

    print("Processing {} episodes{}...".format(len(ep_ids), " (dry run)" if args.dry_run else ""))
    print("=" * 100)

    warnings = []
    boundaries = []

    for ep_id in ep_ids:
        table = load_episode(ep_id)
        arm1_g, arm2_g, arm1_joints, arm2_joints = extract_data(table)
        phases, boundary, debug = detect_phases(arm1_g, arm2_g, arm1_joints, arm2_joints)

        n = len(phases)
        p1 = int((phases == 1).sum())
        p2 = int((phases == 2).sum())
        warn = debug.get("warning", "")
        method = debug.get("method", "")

        print("ep {:3d} | {:4d} steps | boundary {:4d} | P1: {:4d} ({:5.1f}%) | P2: {:4d} ({:5.1f}%) | {} {}".format(
            ep_id, n, boundary, p1, 100 * p1 / n, p2, 100 * p2 / n,
            method, "| WARNING: " + warn if warn else ""))

        if warn:
            warnings.append((ep_id, warn))
        boundaries.append(boundary)

        if args.visualize:
            visualize_episode(ep_id, arm1_g, arm2_g, arm1_joints, arm2_joints,
                              phases, boundary, debug, vis_dir)

        if not args.dry_run:
            write_phases_to_parquet(ep_id, phases)
            if not args.no_task_index:
                update_task_index_in_parquet(ep_id, phases)

    print("=" * 100)
    print("Boundary stats: min={}, max={}, mean={:.0f}, std={:.0f}".format(
        min(boundaries), max(boundaries), np.mean(boundaries), np.std(boundaries)))

    if warnings:
        print("\nWarnings:")
        for ep_id, w in warnings:
            print("  ep {}: {}".format(ep_id, w))

    if not args.dry_run:
        update_info_json(DATASET_DIR)
        update_tasks_jsonl(DATASET_DIR)
        print("\nPhase data written to parquet files.")
        if not args.no_task_index:
            print("task_index updated (phase 1 -> task_index 1, phase 2 -> task_index 2).")
    else:
        print("\n(Dry run - no files modified)")


if __name__ == "__main__":
    main()
