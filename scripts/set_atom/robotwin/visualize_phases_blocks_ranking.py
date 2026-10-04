"""
Visualization script for verifying phase labels in the blocks_ranking_rgb dataset.

This script reads the phase-labeled dataset and generates:
1. Gripper state + phase overlay plots for specified episodes
2. Summary statistics across all episodes

Usage:
  python scripts/set_atom/robotwin/visualize_phases_blocks_ranking.py \
    --dataset_dir dataset/robotwin_lerobot/blocks_ranking_rgb_phase \
    --episodes 0 1 2 3 4 \
    --output_dir visualizations/blocks_ranking_phases
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PHASE_INSTRUCTIONS = {
    1: {"name": "pick_red",    "task": "Pick up the red block."},
    2: {"name": "place_red",   "task": "Place the red block on the left position."},
    3: {"name": "pick_green",  "task": "Pick up the green block."},
    4: {"name": "place_green", "task": "Place the green block in the middle position."},
    5: {"name": "pick_blue",   "task": "Pick up the blue block."},
    6: {"name": "place_blue",  "task": "Place the blue block on the right position."},
}

PHASE_COLORS = {
    1: '#FF6B6B',  # pick red - light red
    2: '#CC0000',  # place red - dark red
    3: '#90EE90',  # pick green - light green
    4: '#006400',  # place green - dark green
    5: '#87CEEB',  # pick blue - light blue
    6: '#0000CC',  # place blue - dark blue
}


def load_episode(dataset_dir: str, ep_idx: int):
    """Load a single episode from the phase-labeled dataset."""
    parquet_path = Path(dataset_dir) / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(str(parquet_path))
    
    states = np.array(df['observation.state'].tolist())
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]
    
    phases = df['phase'].values if 'phase' in df.columns else None
    
    return df, left_gripper, right_gripper, phases


def plot_episode(ep_idx, left_gripper, right_gripper, phases, output_dir):
    """Generate a detailed visualization for one episode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(left_gripper)
    time_steps = np.arange(num_frames)
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    
    # --- Left gripper ---
    ax1 = axes[0]
    ax1.plot(time_steps, left_gripper, 'b-', linewidth=1.5, label='Left Gripper')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Left Gripper\n(1=open, 0=closed)', fontsize=11)
    ax1.set_ylim(-0.1, 1.15)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Episode {ep_idx} — Phase Analysis ({num_frames} frames)', fontsize=14, fontweight='bold')
    
    # Add phase background colors to gripper plots
    if phases is not None:
        for phase_id in range(1, 7):
            mask = phases == phase_id
            if mask.any():
                ax1.fill_between(time_steps, -0.1, 1.15, where=mask, alpha=0.15, color=PHASE_COLORS[phase_id])
                ax2 = axes[1]
    
    # --- Right gripper ---
    ax2 = axes[1]
    ax2.plot(time_steps, right_gripper, 'r-', linewidth=1.5, label='Right Gripper')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Right Gripper\n(1=open, 0=closed)', fontsize=11)
    ax2.set_ylim(-0.1, 1.15)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    if phases is not None:
        for phase_id in range(1, 7):
            mask = phases == phase_id
            if mask.any():
                ax2.fill_between(time_steps, -0.1, 1.15, where=mask, alpha=0.15, color=PHASE_COLORS[phase_id])
    
    # --- Phase timeline ---
    ax3 = axes[2]
    if phases is not None:
        for phase_id in range(1, 7):
            mask = phases == phase_id
            if mask.any():
                ax3.fill_between(
                    time_steps, 0, 1, where=mask,
                    alpha=0.6, color=PHASE_COLORS[phase_id],
                    label=f'Phase {phase_id}: {PHASE_INSTRUCTIONS[phase_id]["name"]}',
                )
                # Add text label in the middle of each phase region
                indices = np.where(mask)[0]
                mid = indices[len(indices) // 2]
                ax3.text(
                    mid, 0.5, f'P{phase_id}\n{PHASE_INSTRUCTIONS[phase_id]["name"]}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white' if phase_id in [2, 4, 6] else 'black',
                )
    
    ax3.set_ylabel('Phase', fontsize=11)
    ax3.set_xlabel('Frame Index', fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3)
    
    # Legend for phases
    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS[pid], alpha=0.6, label=f'P{pid}: {pinfo["name"]} — "{pinfo["task"]}"')
        for pid, pinfo in PHASE_INSTRUCTIONS.items()
    ]
    ax3.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=9, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    output_file = output_dir / f"phase_vis_episode_{ep_idx:03d}.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_summary(dataset_dir: str, output_dir: str):
    """Generate summary statistics across all episodes."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    all_phase_lengths = {p: [] for p in range(1, 7)}
    
    for parquet_path in parquet_files:
        df = pd.read_parquet(str(parquet_path))
        if 'phase' not in df.columns:
            print(f"WARNING: {parquet_path.name} has no 'phase' column, skipping")
            continue
        
        phases = df['phase'].values
        for p in range(1, 7):
            count = int(np.sum(phases == p))
            all_phase_lengths[p].append(count)
    
    # Bar chart of average phase lengths
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average phase length
    ax1 = axes[0]
    means = [np.mean(all_phase_lengths[p]) for p in range(1, 7)]
    stds = [np.std(all_phase_lengths[p]) for p in range(1, 7)]
    colors = [PHASE_COLORS[p] for p in range(1, 7)]
    labels = [f'P{p}: {PHASE_INSTRUCTIONS[p]["name"]}' for p in range(1, 7)]
    
    bars = ax1.bar(range(1, 7), means, yerr=stds, color=colors, alpha=0.7, edgecolor='black', capsize=4)
    ax1.set_xlabel('Phase', fontsize=12)
    ax1.set_ylabel('Average Frame Count', fontsize=12)
    ax1.set_title('Average Phase Length (frames)', fontsize=14)
    ax1.set_xticks(range(1, 7))
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Phase proportion per episode (stacked bar)
    ax2 = axes[1]
    n_episodes = len(parquet_files)
    bottoms = np.zeros(n_episodes)
    for p in range(1, 7):
        lengths = np.array(all_phase_lengths[p])
        ax2.bar(range(n_episodes), lengths, bottom=bottoms, color=PHASE_COLORS[p], alpha=0.7,
                label=f'P{p}: {PHASE_INSTRUCTIONS[p]["name"]}')
        bottoms += lengths
    
    ax2.set_xlabel('Episode Index', fontsize=12)
    ax2.set_ylabel('Frame Count', fontsize=12)
    ax2.set_title('Phase Composition per Episode', fontsize=14)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "phase_summary.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {output_file}")
    
    # Print statistics
    print("\n=== Phase Statistics ===")
    for p in range(1, 7):
        lengths = all_phase_lengths[p]
        print(f"Phase {p} ({PHASE_INSTRUCTIONS[p]['name']:12s}): "
              f"avg={np.mean(lengths):6.1f} ± {np.std(lengths):5.1f} frames, "
              f"min={np.min(lengths):4d}, max={np.max(lengths):4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize phase labels for blocks_ranking_rgb")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/robotwin_lerobot/blocks_ranking_rgb_phase",
        help="Path to phase-labeled dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Episode indices to visualize (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations/blocks_ranking_phases",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Also generate summary statistics plot",
    )
    
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}\n")
    
    for ep_idx in args.episodes:
        try:
            df, left_g, right_g, phases = load_episode(args.dataset_dir, ep_idx)
            plot_episode(ep_idx, left_g, right_g, phases, args.output_dir)
        except Exception as e:
            print(f"Error visualizing episode {ep_idx}: {e}")
    
    if args.summary:
        plot_summary(args.dataset_dir, args.output_dir)


