"""
Detect and write 2-phase labels for the stackbowls_v2 dataset.

Phase 1: arm2 is active (from start until arm2 finishes and arm1 begins)
Phase 2: arm1 is active (from arm1 start until end)

Detection strategy:
  - Primary signal: arm1's first close->open gripper transition (the clearest
    indicator that arm1 starts its "place" action).
  - Look back from that point using joint velocity to find when arm1 first
    started moving (the "approach" phase before gripper opens).
  - That start-of-arm1-movement is the phase boundary.
  - Handles SO100 gripper noise where grippers rest at closed position.

Usage:
    python scripts/so101/set_phases_stackbowls.py --dry-run --visualize
    python scripts/so101/set_phases_stackbowls.py
"""

import argparse
import os
import numpy as np

DATASET_DIR = "/18141169908/zhangzaibin/multivla-pi0/dataset/realworld/stackbowls_v2"
DATA_DIR = os.path.join(DATASET_DIR, "data", "chunk-000")

ARM1_GRIPPER_IDX = 5
ARM2_GRIPPER_IDX = 11

ARM1_OPEN_THRESH = 10.0
ARM1_CLOSED_THRESH = 5.0
JOINT_VEL_THRESH = 1.0
SMOOTH_WINDOW = 15
LOOKBACK_MARGIN = 30


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
    arm1_joints = states[:, :5]
    return arm1_g, arm2_g, arm1_joints


def smooth(x, w=SMOOTH_WINDOW):
    return np.convolve(x, np.ones(w) / w, mode="same")


def find_arm1_first_open_transition(arm1_g):
    """Find the first frame where arm1 gripper transitions from closed to open.
    Handles ep0-like cases where gripper starts open by requiring a preceding
    closed state."""
    n = len(arm1_g)
    was_closed = arm1_g[0] < ARM1_CLOSED_THRESH
    for i in range(1, n):
        if arm1_g[i] < ARM1_CLOSED_THRESH:
            was_closed = True
        if was_closed and arm1_g[i] > ARM1_OPEN_THRESH:
            return i
    return None


def find_arm1_movement_start(arm1_joints, anchor, margin=LOOKBACK_MARGIN):
    """Look backwards from `anchor` to find when arm1 joints first started
    moving significantly (smoothed velocity > threshold)."""
    vel = smooth(np.linalg.norm(np.diff(arm1_joints, axis=0), axis=1))
    search_start = max(0, anchor - 1)
    for i in range(search_start, -1, -1):
        if i < len(vel) and vel[i] < JOINT_VEL_THRESH:
            return min(i + 1, anchor)
    return 0


def detect_phases(arm1_g, arm2_g, arm1_joints):
    """
    Returns: (phases, boundary, debug_info)
    """
    n = len(arm1_g)
    debug = {}

    arm1_open_transition = find_arm1_first_open_transition(arm1_g)
    debug["arm1_open_transition"] = arm1_open_transition

    if arm1_open_transition is not None:
        arm1_move_start = find_arm1_movement_start(arm1_joints, arm1_open_transition)
        debug["arm1_move_start"] = arm1_move_start
        boundary = arm1_move_start
    else:
        debug["warning"] = "arm1 never opens; assigning all to Phase 1"
        boundary = n - 1

    boundary = max(1, min(boundary, n - 1))
    debug["boundary"] = boundary

    phases = np.ones(n, dtype=np.int32)
    phases[boundary:] = 2

    return phases, boundary, debug


def visualize_episode(ep_id, arm1_g, arm2_g, arm1_joints, phases, boundary, debug, save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(arm1_g)
    t = np.arange(n)
    arm1_vel = smooth(np.linalg.norm(np.diff(arm1_joints, axis=0), axis=1))

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, arm2_g, "r-", linewidth=1.5, label="arm2 gripper")
    axes[0].axvline(x=boundary, color="k", linestyle="--", linewidth=2, label="boundary")
    axes[0].set_ylabel("arm2 gripper")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Episode {} | boundary @ step {}".format(ep_id, boundary))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, arm1_g, "b-", linewidth=1.5, label="arm1 gripper")
    axes[1].axhline(y=ARM1_OPEN_THRESH, color="b", linestyle="--", alpha=0.5, label="open threshold")
    axes[1].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    if debug.get("arm1_open_transition") is not None:
        axes[1].axvline(x=debug["arm1_open_transition"], color="g", linestyle=":", linewidth=1.5,
                        label="1st open @{}".format(debug["arm1_open_transition"]))
    axes[1].set_ylabel("arm1 gripper")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.arange(len(arm1_vel)), arm1_vel, "purple", linewidth=1, label="arm1 joint vel (smoothed)")
    axes[2].axhline(y=JOINT_VEL_THRESH, color="purple", linestyle="--", alpha=0.5)
    axes[2].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    axes[2].set_ylabel("arm1 joint vel")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    colors = {1: "tab:orange", 2: "tab:blue"}
    for phase_val in [1, 2]:
        mask = phases == phase_val
        axes[3].fill_between(t, 0, 1, where=mask, alpha=0.5,
                             color=colors[phase_val], label="Phase {}".format(phase_val))
    axes[3].axvline(x=boundary, color="k", linestyle="--", linewidth=2)
    axes[3].set_ylabel("Phase")
    axes[3].set_xlabel("Time Step")
    axes[3].set_yticks([])
    axes[3].legend(loc="upper right")

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

    # Store as float32 scalar (not list) so LeRobot can cast it correctly.
    # LeRobot maps shape:[1] non-image features to plain scalars, not sequences.
    phase_col = pa.array([float(p) for p in phases], type=pa.float32())
    table = table.append_column("phase", phase_col)
    pq.write_table(table, path)


def update_info_json(dataset_dir):
    """Add the phase feature declaration to meta/info.json if not already present."""
    import json

    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    if "phase" in info["features"]:
        return  # already declared, nothing to do

    info["features"]["phase"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print("Updated meta/info.json: added 'phase' feature declaration.")


def main():
    parser = argparse.ArgumentParser(description="Set phases for stackbowls_v2 dataset")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument("--dry-run", action="store_true", help="Detect only, don't write")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode IDs (default: all)")
    args = parser.parse_args()

    parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
    all_ep_ids = [int(f.split("_")[1].split(".")[0]) for f in parquet_files]

    ep_ids = [int(x) for x in args.episodes.split(",")] if args.episodes else all_ep_ids
    vis_dir = os.path.join(DATASET_DIR, "phase_plots") if args.visualize else None

    print("Processing {} episodes{}...".format(len(ep_ids), " (dry run)" if args.dry_run else ""))
    print("=" * 80)

    warnings = []
    boundaries = []
    for ep_id in ep_ids:
        table = load_episode(ep_id)
        arm1_g, arm2_g, arm1_joints = extract_data(table)
        phases, boundary, debug = detect_phases(arm1_g, arm2_g, arm1_joints)

        n = len(phases)
        p1 = int((phases == 1).sum())
        p2 = int((phases == 2).sum())
        warn = debug.get("warning", "")

        print("ep {:3d} | {:4d} steps | boundary {:4d} | P1: {:4d} ({:5.1f}%) | P2: {:4d} ({:5.1f}%){}".format(
            ep_id, n, boundary, p1, 100 * p1 / n, p2, 100 * p2 / n,
            " | WARNING: " + warn if warn else ""))

        if warn:
            warnings.append((ep_id, warn))
        boundaries.append(boundary)

        if args.visualize:
            visualize_episode(ep_id, arm1_g, arm2_g, arm1_joints, phases, boundary, debug, vis_dir)

        if not args.dry_run:
            write_phases_to_parquet(ep_id, phases)

    print("=" * 80)
    print("Boundary stats: min={}, max={}, mean={:.0f}, std={:.0f}".format(
        min(boundaries), max(boundaries), np.mean(boundaries), np.std(boundaries)))
    if warnings:
        print("\nWarnings:")
        for ep_id, w in warnings:
            print("  ep {}: {}".format(ep_id, w))

    if not args.dry_run:
        update_info_json(DATASET_DIR)
        print("\nPhase data written to parquet files.")
    else:
        print("\n(Dry run - no files modified)")


if __name__ == "__main__":
    main()
