"""
Visualize phase labels for a phase-labeled SO101 bi-arm dataset.

Reads the output of set_phases_so101_v2.py and generates per-episode 4-panel plots:
  Panel 1: left gripper state with phase boundaries
  Panel 2: right gripper state with phase boundaries
  Panel 3: phase ID over time (line)
  Panel 4: colored phase bar

Usage:
    python scripts/so101/multitask/visualize_phases_so101_v2.py \\
        --dataset dataset/macg_so101/macg_stack_bowls_v2_phased \\
        [--episodes 0,1,5] \\
        [--output_dir /path/to/plots]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11

PHASE_COLORS = {
    1: "tab:orange",
    2: "tab:red",
    3: "tab:green",
    4: "tab:blue",
    5: "tab:gray",
}
PHASE_LABELS = {
    1: "P1: 1st-arm pick",
    2: "P2: 1st-arm action",
    3: "P3: 2nd-arm pick",
    4: "P4: 2nd-arm action",
    5: "P5: recover",
}


def load_episode(data_dir: Path, ep_id: int):
    path = data_dir / f"episode_{ep_id:06d}.parquet"
    table = pq.read_table(path)
    n = table.num_rows
    states = np.array([np.array(table.column("state")[i].as_py()) for i in range(n)])
    phases = np.array([float(table.column("phase")[i].as_py()) for i in range(n)], dtype=np.int32)
    return states, phases, n


def find_boundaries(phases: np.ndarray) -> list[int]:
    """Return frame indices where phase value changes."""
    boundaries = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            boundaries.append(i)
    return boundaries


def visualize_episode(ep_id: int, states: np.ndarray, phases: np.ndarray, save_path: Path) -> None:
    n = len(phases)
    t = np.arange(n)
    left_g = states[:, LEFT_GRIPPER_IDX]
    right_g = states[:, RIGHT_GRIPPER_IDX]
    boundaries = find_boundaries(phases)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Episode {ep_id} | phases detected | boundaries @ {boundaries}", fontsize=11)

    # Panel 1: left gripper
    axes[0].plot(t, left_g, "b-", linewidth=1.2, label="left gripper")
    for b in boundaries:
        axes[0].axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_ylabel("Left gripper pos")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: right gripper
    axes[1].plot(t, right_g, "r-", linewidth=1.2, label="right gripper")
    for b in boundaries:
        axes[1].axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("Right gripper pos")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: phase ID over time
    axes[2].step(t, phases, where="post", color="purple", linewidth=1.5, label="phase")
    axes[2].set_ylabel("Phase")
    axes[2].set_ylim(0.5, 5.5)
    axes[2].set_yticks([1, 2, 3, 4, 5])
    axes[2].set_yticklabels([PHASE_LABELS[p] for p in [1, 2, 3, 4, 5]], fontsize=7)
    for b in boundaries:
        axes[2].axvline(b, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: colored phase bar
    for phase_val in range(1, 6):
        mask = phases == phase_val
        axes[3].fill_between(
            t, 0, 1, where=mask, alpha=0.65,
            color=PHASE_COLORS[phase_val], label=PHASE_LABELS[phase_val],
        )
    for b in boundaries:
        axes[3].axvline(b, color="black", linestyle="--", linewidth=1.5)
    axes[3].set_ylabel("Phase bar")
    axes[3].set_xlabel("Time Step")
    axes[3].set_yticks([])
    axes[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize phase-labeled SO101 dataset.")
    parser.add_argument("--dataset", required=True, help="Path to phased LeRobot dataset directory")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode IDs to visualize (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save plots (default: <dataset>/phase_plots/)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    data_dir = dataset_path / "data" / "chunk-000"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "phase_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    all_ep_ids = [int(p.stem.split("_")[1]) for p in parquet_files]

    if args.episodes:
        ep_filter = {int(x) for x in args.episodes.split(",")}
        all_ep_ids = [e for e in all_ep_ids if e in ep_filter]

    print(f"Visualizing {len(all_ep_ids)} episodes from {dataset_path}")

    for ep_id in all_ep_ids:
        states, phases, n = load_episode(data_dir, ep_id)
        save_path = output_dir / f"phase_ep{ep_id:03d}.png"
        visualize_episode(ep_id, states, phases, save_path)
        boundaries = find_boundaries(phases)
        phase_counts = {p: int((phases == p).sum()) for p in range(1, 6)}
        print(f"  ep {ep_id:3d} | n={n} | boundaries={boundaries} | counts={phase_counts} -> {save_path.name}")

    print(f"\nDone. {len(all_ep_ids)} plots saved to {output_dir}")


if __name__ == "__main__":
    main()
