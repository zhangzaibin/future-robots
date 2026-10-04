"""
Phase labeling script for RobotWin blocks_ranking_rgb dataset (LeRobot format).

Task: Arrange red, green, and blue blocks in a row from left to right.

Phase segmentation based on gripper open/close transitions:
  Phase 1 (pick_red):   Start → first gripper close    (approach and grasp red block)
  Phase 2 (place_red):  First close → first open        (move and release red block to left)
  Phase 3 (pick_green): First open → second close       (approach and grasp green block)
  Phase 4 (place_green):Second close → second open      (move and release green block to middle)
  Phase 5 (pick_blue):  Second open → third close       (approach and grasp blue block)
  Phase 6 (place_blue): Third close → third open        (move and release blue block to right)

Arm-wise atomic instruction format:
  "{arm0(left) instruction} | {arm1(right) instruction}"
  The active arm gets the action instruction, the other arm gets "Wait for the other robots."
  Which arm is active varies per episode and is detected from gripper transitions.

The gripper value convention:
  1.0 = open, 0.0 = closed
  arm0 = Left arm:  observation.state[0:7],  gripper = state[6]
  arm1 = Right arm: observation.state[7:14], gripper = state[13]

Usage:
  python scripts/set_atom/robotwin/set_phases_blocks_ranking.py \
    --input_dir dataset/robotwin_lerobot/blocks_ranking_rgb \
    --output_dir dataset/robotwin_lerobot/blocks_ranking_rgb_phase \
    --visualize
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Phase definitions: phase_id -> (phase_name, action_instruction for the active arm)
PHASE_INSTRUCTIONS = {
    1: {
        "name": "pick_red",
        "task": "Pick up the red block.",
    },
    2: {
        "name": "place_red",
        "task": "Place the red block on the left position.",
    },
    3: {
        "name": "pick_green",
        "task": "Pick up the green block.",
    },
    4: {
        "name": "place_green",
        "task": "Place the green block in the middle position.",
    },
    5: {
        "name": "pick_blue",
        "task": "Pick up the blue block.",
    },
    6: {
        "name": "place_blue",
        "task": "Place the blue block on the right position.",
    },
}

WAIT_INSTRUCTION = "Wait for the other robots."


def get_arm_wise_instruction(phase_id: int, active_arm: str) -> str:
    """
    Build arm-wise instruction: "{arm0(left) instruction} | {arm1(right) instruction}"

    Args:
        phase_id: Phase ID (1-6)
        active_arm: Which arm is active ('left' or 'right')

    Returns:
        Combined instruction string with ' | ' separator.
    """
    action_task = PHASE_INSTRUCTIONS[phase_id]["task"]
    if active_arm == "left":
        # arm0 (left) is active, arm1 (right) waits
        return f"{action_task} | {WAIT_INSTRUCTION}"
    else:
        # arm1 (right) is active, arm0 (left) waits
        return f"{WAIT_INSTRUCTION} | {action_task}"


def get_phase_active_arms(events: list) -> dict:
    """
    Determine which arm is active for each phase based on gripper transition events.

    The close events (events[0], events[2], events[4]) tell us which arm handles
    block 1 (red), block 2 (green), block 3 (blue). Pick and place for the same
    block always use the same arm.

    Args:
        events: List of (frame, event_type, arm) tuples, sorted by frame.

    Returns:
        Dictionary mapping phase_id (1-6) to active arm ('left' or 'right').
    """
    if len(events) != 6:
        return {p: "right" for p in range(1, 7)}  # fallback

    block1_arm = events[0][2]  # 1st close → red block arm
    block2_arm = events[2][2]  # 2nd close → green block arm
    block3_arm = events[4][2]  # 3rd close → blue block arm

    return {
        1: block1_arm,   # pick red
        2: block1_arm,   # place red (same arm)
        3: block2_arm,   # pick green
        4: block2_arm,   # place green (same arm)
        5: block3_arm,   # pick blue
        6: block3_arm,   # place blue (same arm)
    }


def detect_gripper_transitions(left_gripper: np.ndarray, right_gripper: np.ndarray, threshold: float = 0.5):
    """
    Detect gripper open/close transition frames from both arms.
    
    Args:
        left_gripper: Array of left gripper values (1.0=open, 0.0=closed)
        right_gripper: Array of right gripper values (1.0=open, 0.0=closed)
        threshold: Threshold to binarize gripper state (>threshold = open)
    
    Returns:
        events: List of (frame, event_type) tuples sorted by frame.
                event_type is 'close' or 'open'.
    """
    left_binary = (left_gripper > threshold).astype(int)   # 1=open, 0=closed
    right_binary = (right_gripper > threshold).astype(int)
    
    left_diff = np.diff(left_binary)
    right_diff = np.diff(right_binary)
    
    events = []
    
    # Left arm transitions
    for f in np.where(left_diff == -1)[0]:  # open -> close
        events.append((int(f), 'close', 'left'))
    for f in np.where(left_diff == 1)[0]:   # close -> open
        events.append((int(f), 'open', 'left'))
    
    # Right arm transitions
    for f in np.where(right_diff == -1)[0]:  # open -> close
        events.append((int(f), 'close', 'right'))
    for f in np.where(right_diff == 1)[0]:   # close -> open
        events.append((int(f), 'open', 'right'))
    
    # Sort by frame
    events.sort(key=lambda x: x[0])
    
    return events


def assign_phases(num_frames: int, events: list) -> np.ndarray:
    """
    Assign phase labels to each frame based on gripper transition events.
    
    Expected event sequence: close, open, close, open, close, open (6 events)
    This creates 6 phases (+ optional trailing frames after last open).
    
    Phase assignment:
      Phase 1: [0, first_close]          - pick red
      Phase 2: [first_close+1, first_open] - place red  
      Phase 3: [first_open+1, second_close] - pick green
      Phase 4: [second_close+1, second_open] - place green
      Phase 5: [second_open+1, third_close] - pick blue
      Phase 6: [third_close+1, third_open] - place blue
      Remaining frames after third_open: Phase 6 (completion)
    
    Args:
        num_frames: Total number of frames in the episode
        events: List of (frame, event_type, arm) tuples
    
    Returns:
        phases: Array of shape (num_frames,) with phase labels (1-6)
    """
    phases = np.zeros(num_frames, dtype=np.int32)
    
    # Extract just the frame and type (ignore arm info for phase assignment)
    event_frames = [(f, t) for f, t, _ in events]
    
    if len(event_frames) != 6:
        print(f"  WARNING: Expected 6 events, got {len(event_frames)}: {event_frames}")
        # Fallback: assign all frames to phase 1
        phases[:] = 1
        return phases
    
    # Validate expected pattern: close, open, close, open, close, open
    expected_types = ['close', 'open', 'close', 'open', 'close', 'open']
    actual_types = [t for _, t in event_frames]
    if actual_types != expected_types:
        print(f"  WARNING: Unexpected event pattern: {actual_types}")
        phases[:] = 1
        return phases
    
    # Get boundary frames
    first_close = event_frames[0][0]
    first_open = event_frames[1][0]
    second_close = event_frames[2][0]
    second_open = event_frames[3][0]
    third_close = event_frames[4][0]
    third_open = event_frames[5][0]
    
    # Assign phases
    # Phase boundaries: the transition frame belongs to the NEW phase
    # e.g., when gripper closes at frame f, frame f starts the "place" phase
    phases[0:first_close + 1] = 1          # pick red (up to and including the close frame)
    phases[first_close + 1:first_open + 1] = 2   # place red
    phases[first_open + 1:second_close + 1] = 3  # pick green
    phases[second_close + 1:second_open + 1] = 4 # place green
    phases[second_open + 1:third_close + 1] = 5  # pick blue
    phases[third_close + 1:] = 6                  # place blue (includes remaining frames)
    
    return phases


def process_episode(parquet_path: str, threshold: float = 0.5):
    """
    Process a single episode parquet file and return phase labels.
    
    Args:
        parquet_path: Path to the episode parquet file
        threshold: Gripper threshold for binarization
    
    Returns:
        df: The original DataFrame
        phases: Phase label array
        events: List of gripper transition events
    """
    df = pd.read_parquet(parquet_path)
    
    states = np.array(df['observation.state'].tolist())
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]
    
    events = detect_gripper_transitions(left_gripper, right_gripper, threshold)
    phases = assign_phases(len(df), events)
    
    return df, phases, events, left_gripper, right_gripper


def add_phases_to_dataset(
    input_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    visualize: bool = False,
    vis_dir: str = None,
):
    """
    Add phase labels and arm-wise atomic instructions to a LeRobot dataset.

    This function:
    1. Copies the dataset to output_dir
    2. Adds a 'phase' column to each parquet file
    3. Detects which arm is active per phase per episode
    4. Builds arm-wise instructions: "{arm0 instruction} | {arm1 instruction}"
    5. Updates 'task_index' per frame to point to the correct arm-wise instruction
    6. Updates metadata files (info.json, tasks.jsonl, episodes.jsonl)

    Args:
        input_dir: Path to input LeRobot dataset
        output_dir: Path to output dataset with phases
        threshold: Gripper threshold for phase detection
        visualize: Whether to generate visualization plots
        vis_dir: Directory to save visualization plots
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if vis_dir is None:
        vis_dir = output_dir / "visualizations"
    else:
        vis_dir = Path(vis_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata directory
    meta_output = output_dir / "meta"
    meta_output.mkdir(parents=True, exist_ok=True)

    # Create data output directory
    data_output = output_dir / "data" / "chunk-000"
    data_output.mkdir(parents=True, exist_ok=True)

    # Read original metadata
    with open(input_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)

    # ---- Build the full set of arm-wise task strings ----
    # For each phase (1-6), there are 2 possible instructions depending on which arm is active.
    # We pre-build all 12 possible task strings and assign them task_index 0..11.
    # Layout: for phase p, left-active variant at index (p-1)*2, right-active at (p-1)*2+1
    all_tasks = []       # list of {"task_index": int, "task": str}
    task_str_to_idx = {} # task_string -> task_index

    for phase_id in range(1, 7):
        for active_arm in ["left", "right"]:
            task_str = get_arm_wise_instruction(phase_id, active_arm)
            if task_str not in task_str_to_idx:
                idx = len(all_tasks)
                task_str_to_idx[task_str] = idx
                all_tasks.append({"task_index": idx, "task": task_str})

    # Collect parquet files
    parquet_dir = input_dir / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))

    print(f"Processing {len(parquet_files)} episodes from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Gripper threshold: {threshold}")
    print(f"Arm-wise task definitions ({len(all_tasks)} tasks):")
    for t in all_tasks:
        print(f"  task_index {t['task_index']:2d}: {t['task']}")
    print()

    # Process each episode
    episodes_info = []
    total_frames = 0

    for parquet_path in parquet_files:
        ep_idx = int(parquet_path.stem.split("_")[1])
        print(f"Processing episode {ep_idx}...", end=" ")

        # Detect phases
        df, phases, events, left_gripper, right_gripper = process_episode(
            str(parquet_path), threshold
        )

        # Determine which arm is active for each phase in this episode
        phase_active_arms = get_phase_active_arms(events)
        arm_pattern = "/".join(phase_active_arms[p][0].upper() for p in [1, 3, 5])

        # Add phase column
        df['phase'] = phases

        # Build per-frame task_index based on phase + active arm
        task_indices = np.zeros(len(df), dtype=np.int64)
        for i in range(len(df)):
            pid = int(phases[i])
            active_arm = phase_active_arms[pid]
            task_str = get_arm_wise_instruction(pid, active_arm)
            task_indices[i] = task_str_to_idx[task_str]

        df['task_index'] = task_indices

        # Save modified parquet
        output_parquet = data_output / parquet_path.name
        df.to_parquet(str(output_parquet), index=False)

        # Phase distribution
        phase_counts = {int(p): int(np.sum(phases == p)) for p in range(1, 7)}
        print(f"arm_pattern={arm_pattern}  phases: {phase_counts}")

        # Record episode info
        ep_task_strs = []
        for pid in sorted(set(phases)):
            active_arm = phase_active_arms[int(pid)]
            ep_task_strs.append(get_arm_wise_instruction(int(pid), active_arm))
        episodes_info.append({
            "episode_index": ep_idx,
            "tasks": ep_task_strs,
            "length": len(df),
        })
        total_frames += len(df)

        # Visualization
        if visualize and ep_idx < 5:  # Visualize first 5 episodes
            visualize_phases_for_episode(
                ep_idx, left_gripper, right_gripper, phases, events,
                phase_active_arms, vis_dir
            )

    # Update and save metadata
    # Update info.json
    info["total_tasks"] = len(all_tasks)
    info["total_frames"] = total_frames

    # Add phase feature to info
    info["features"]["phase"] = {
        "dtype": "int32",
        "shape": [1],
        "names": None,
    }

    with open(meta_output / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Save tasks.jsonl
    with open(meta_output / "tasks.jsonl", "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")

    # Save episodes.jsonl
    with open(meta_output / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")

    # Copy episodes_stats.jsonl (unchanged)
    src_stats = input_dir / "meta" / "episodes_stats.jsonl"
    if src_stats.exists():
        shutil.copy2(str(src_stats), str(meta_output / "episodes_stats.jsonl"))

    print(f"\nDone! Dataset with arm-wise phases saved to: {output_dir}")
    print(f"Total episodes: {len(parquet_files)}")
    print(f"Total frames: {total_frames}")
    print(f"Total arm-wise tasks: {len(all_tasks)}")


def visualize_phases_for_episode(
    ep_idx: int,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    phases: np.ndarray,
    events: list,
    phase_active_arms: dict,
    output_dir: str or Path,
):
    """
    Generate a visualization plot showing gripper states, detected phases, and arm-wise instructions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    time_steps = np.arange(len(phases))

    phase_colors = {
        1: '#FF6B6B',  # pick red - light red
        2: '#FF0000',  # place red - red
        3: '#90EE90',  # pick green - light green
        4: '#008000',  # place green - green
        5: '#87CEEB',  # pick blue - light blue
        6: '#0000FF',  # place blue - blue
    }

    # Build arm pattern string
    arm_pattern = "/".join(phase_active_arms[p][0].upper() for p in [1, 3, 5])

    # Plot left gripper (arm0)
    ax1 = axes[0]
    ax1.plot(time_steps, left_gripper, 'b-', linewidth=1.5, label='arm0 (Left) Gripper')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax1.set_ylabel('arm0 (Left) Gripper')
    ax1.set_title(f'Episode {ep_idx} — Arm Pattern: {arm_pattern} (Red/Green/Blue)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    # Add phase background
    for pid in range(1, 7):
        mask = phases == pid
        if mask.any():
            ax1.fill_between(time_steps, -0.1, 1.1, where=mask, alpha=0.12, color=phase_colors[pid])

    # Plot right gripper (arm1)
    ax2 = axes[1]
    ax2.plot(time_steps, right_gripper, 'r-', linewidth=1.5, label='arm1 (Right) Gripper')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax2.set_ylabel('arm1 (Right) Gripper')
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    for pid in range(1, 7):
        mask = phases == pid
        if mask.any():
            ax2.fill_between(time_steps, -0.1, 1.1, where=mask, alpha=0.12, color=phase_colors[pid])

    # Plot phases with arm-wise instruction labels
    ax3 = axes[2]
    for phase_id in range(1, 7):
        mask = phases == phase_id
        if mask.any():
            active_arm = phase_active_arms[phase_id]
            arm_label = "L" if active_arm == "left" else "R"
            task_str = get_arm_wise_instruction(phase_id, active_arm)
            ax3.fill_between(
                time_steps, 0, 1,
                where=mask,
                alpha=0.4,
                color=phase_colors[phase_id],
                label=f'P{phase_id}({arm_label}): {PHASE_INSTRUCTIONS[phase_id]["name"]}',
            )
            # Add instruction text in the middle of the phase region
            indices = np.where(mask)[0]
            mid = indices[len(indices) // 2]
            # Shorten display for readability
            short_task = task_str.replace("Wait for the other robots.", "Wait")
            ax3.text(mid, 0.5, f'P{phase_id}({arm_label})\n{short_task}',
                     ha='center', va='center', fontsize=6, fontweight='bold')

    ax3.set_ylabel('Phase')
    ax3.set_xlabel('Frame')
    ax3.legend(loc='upper right', fontsize=7, ncol=3)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Add vertical lines at transition events
    for frame, etype, arm in events:
        color = 'blue' if arm == 'left' else 'red'
        linestyle = '--' if etype == 'close' else '-.'
        for ax in axes:
            ax.axvline(x=frame, color=color, linestyle=linestyle, alpha=0.6, linewidth=0.8)

    plt.tight_layout()
    output_file = output_dir / f"phase_analysis_episode_{ep_idx:03d}.png"
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization: {output_file}")


def analyze_dataset(input_dir: str, threshold: float = 0.5):
    """
    Analyze the dataset without modifying it. Print statistics about gripper transitions.
    """
    input_dir = Path(input_dir)
    parquet_dir = input_dir / "data" / "chunk-000"
    parquet_files = sorted(parquet_dir.glob("episode_*.parquet"))
    
    print(f"Analyzing {len(parquet_files)} episodes from {input_dir}")
    print(f"Gripper threshold: {threshold}\n")
    
    all_good = True
    for parquet_path in parquet_files:
        ep_idx = int(parquet_path.stem.split("_")[1])
        df, phases, events, left_gripper, right_gripper = process_episode(
            str(parquet_path), threshold
        )
        
        event_summary = [(f, t, a) for f, t, a in events]
        types = [t for _, t, _ in events]
        expected = ['close', 'open', 'close', 'open', 'close', 'open']
        
        status = "OK" if (len(events) == 6 and types == expected) else "ANOMALY"
        if status != "OK":
            all_good = False
        
        phase_counts = {int(p): int(np.sum(phases == p)) for p in range(1, 7)}
        print(f"Episode {ep_idx:3d} [{status}] frames={len(df):4d} events={event_summary}")
        print(f"           phases: {phase_counts}")
    
    if all_good:
        print("\n✓ All episodes have the expected 6-event pattern (close-open × 3).")
    else:
        print("\n✗ Some episodes have anomalous patterns. Please check above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add phase labels to RobotWin blocks_ranking_rgb dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="dataset/robotwin_lerobot/blocks_ranking_rgb",
        help="Path to input LeRobot dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/robotwin_lerobot/blocks_ranking_rgb_phase",
        help="Path to output dataset with phases",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Gripper threshold for open/close detection (default: 0.5)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots for first 5 episodes",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Directory for visualization plots (default: output_dir/visualizations)",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze the dataset without modifying it",
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_dataset(args.input_dir, args.threshold)
    else:
        add_phases_to_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            visualize=args.visualize,
            vis_dir=args.vis_dir,
        )

