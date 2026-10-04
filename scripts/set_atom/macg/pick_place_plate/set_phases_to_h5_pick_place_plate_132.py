"""
Phase Detection for pick-and-place-to-plate task — placefactory_132 版本.
每个机械臂分别抓取物体放到盘子里。

与 placefactory_123 使用完全相同的检测逻辑，
但 phase 编号从 11 开始，以便与 123 版本区分：

  Phase 11: Agent0 pick up (approach, grasp, lift)
  Phase 12: Agent0 place  (move to plate, release, return / 归位)
  Phase 13: Agent1 pick up
  Phase 14: Agent1 place
  Phase 15: Agent2 pick up
  Phase 16: Agent2 place
  Phase 17: Wait / Done   (all agents idle)
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---- Import all detection utilities from the base (123) version ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from set_phases_to_h5_pick_place_plate import (
    detect_phases,
    get_per_agent_gripper_thresholds,
    apply_gripper_analysis,
    find_gripper_events,
    detect_motion_state,
    find_still_start_after,
    find_motion_start_after,
)

# ---- Phase offset ----
PHASE_OFFSET = 10  # phases will be 11-17 instead of 1-7


# ========== H5 Processing ==========

def add_phases_to_h5(file_path, output_path=None,
                     close_ratio=0.3, still_threshold=0.02, visualize=False,
                     agent_names=None, phase_offset=PHASE_OFFSET):
    """
    Add phase information to HDF5 file.
    Phases are offset by phase_offset (default 10) so they start from 11.
    """
    if agent_names is None:
        agent_names = ["panda-0", "panda-1", "panda-2"]

    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")
            print(f"Phase offset: +{phase_offset}  (phases will be {1+phase_offset}–{7+phase_offset})")

            traj_keys = sorted(
                [k for k in src_file.keys() if k.startswith("traj_")],
                key=lambda k: int(k.split("_")[1]),
            )
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                print(f"\nProcessing {traj_key}...")

                # Copy the entire trajectory group
                src_file.copy(traj_key, dst_file)

                try:
                    # Load qpos for each agent
                    agents_qpos = []
                    for name in agent_names:
                        qpos = src_file[f"{traj_key}/obs/agent/{name}/qpos"][:]
                        agents_qpos.append(qpos)

                    # Detect phases (1-based)
                    phases, debug_info = detect_phases(
                        agents_qpos,
                        close_ratio=close_ratio,
                        still_threshold=still_threshold,
                    )

                    # Apply offset: 1-7 → 11-17
                    phases_offset = phases + phase_offset

                    # Write phases
                    if f"{traj_key}/phase" in dst_file:
                        del dst_file[f"{traj_key}/phase"]
                    dst_file.create_dataset(f"{traj_key}/phase", data=phases_offset)

                    unique, counts = np.unique(phases_offset, return_counts=True)
                    print(f"  Added phases (with offset): { {int(u): int(c) for u, c in zip(unique, counts)} }")

                    # Optional visualization
                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(
                            agents_qpos, debug_info, phases_offset,
                            traj_key, agent_names, phase_offset=phase_offset,
                        )

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nCompleted! Output saved to: {output_path}")
    return output_path


# ========== Visualization ==========

PHASE_NAMES_132 = {
    11: "Phase 11: Agent0 Pick Up",
    12: "Phase 12: Agent0 Place",
    13: "Phase 13: Agent1 Pick Up",
    14: "Phase 14: Agent1 Place",
    15: "Phase 15: Agent2 Pick Up",
    16: "Phase 16: Agent2 Place",
    17: "Phase 17: Wait / Done",
}


def visualize_phases(agents_qpos, debug_info, phases, traj_key, agent_names,
                     output_dir=".", phase_offset=PHASE_OFFSET):
    """
    Create detailed visualization for a single trajectory (132 version with offset phases).
    """
    n_agents = len(agents_qpos)
    n_steps = len(phases)
    time_steps = np.arange(n_steps)

    colors_line = ["b", "r", "g", "m", "c", "y"]
    phase_colors_list = [
        "lightblue", "lightgreen", "lightyellow", "lightcoral",
        "plum", "lightsalmon", "paleturquoise",
    ]

    n_rows_total = 2 + n_agents + 1
    fig, axes = plt.subplots(n_rows_total, 1, figsize=(18, 4 * n_rows_total))

    # Row 0: Raw gripper positions
    ax = axes[0]
    for i in range(n_agents):
        ax.plot(
            time_steps, agents_qpos[i][:, -1],
            colors_line[i % len(colors_line)] + "-", linewidth=1,
            label=agent_names[i], alpha=0.7,
        )
    ax.set_ylabel("Gripper Position")
    ax.set_title(f"{traj_key} - Pick & Place to Plate Phase Analysis (132, offset +{phase_offset})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Motion indicator
    ax = axes[1]
    for i in range(n_agents):
        ax.plot(
            time_steps, debug_info[f"agent{i}_pos_change"],
            colors_line[i % len(colors_line)] + "-", linewidth=1,
            label=f"{agent_names[i]} motion", alpha=0.7,
        )
    ax.set_ylabel("Position Change")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rows 2..n_agents+1: Binary gripper states
    for i in range(n_agents):
        ax = axes[2 + i]
        ax.plot(
            time_steps, debug_info[f"agent{i}_gripper"],
            colors_line[i % len(colors_line)] + "-", linewidth=2,
            label=f"{agent_names[i]} Gripper State",
        )
        ax.set_ylabel(f"Agent{i} Gripper")
        ax.set_ylim(-0.1, 1.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Last row: Phase plot
    ax = axes[-1]
    min_phase = int(phases.min())
    max_phase = int(phases.max())

    agent_order = debug_info.get("agent_order", list(range(n_agents)))
    phase_labels = {}
    for idx, a in enumerate(agent_order):
        phase_labels[1 + phase_offset + 2 * idx] = f"Phase {1+phase_offset+2*idx} (Agent{a} pick up)"
        phase_labels[2 + phase_offset + 2 * idx] = f"Phase {2+phase_offset+2*idx} (Agent{a} place)"
    phase_labels[max_phase] = f"Phase {max_phase} (Wait/Done)"

    for p in range(min_phase, max_phase + 1):
        mask = phases == p
        if np.any(mask):
            label = phase_labels.get(p, f"Phase {p}")
            color_idx = (p - min_phase) % len(phase_colors_list)
            ax.fill_between(
                time_steps, 0, mask * p, alpha=0.6,
                color=phase_colors_list[color_idx],
                label=label,
            )

    ax.plot(time_steps, phases, "k-", linewidth=2, marker="o", markersize=1)
    ax.set_ylabel("Phase")
    ax.set_xlabel("Time Step")
    ax.set_ylim(min_phase - 0.5, max_phase + 0.5)
    ax.set_yticks(list(range(min_phase, max_phase + 1)))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"phase_analysis_{traj_key}.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to: {output_file}")


# ========== Single Trajectory Analysis ==========

def analyze_single_trajectory(file_path, traj_key,
                              close_ratio=0.3, still_threshold=0.02,
                              agent_names=None, phase_offset=PHASE_OFFSET):
    """
    Analyze a single trajectory (132 version with offset phases).
    """
    if agent_names is None:
        agent_names = ["panda-0", "panda-1", "panda-2"]

    with h5py.File(file_path, "r") as f:
        agents_qpos = []
        for name in agent_names:
            qpos = f[f"{traj_key}/obs/agent/{name}/qpos"][:]
            agents_qpos.append(qpos)

        print(f"\n{'='*50}")
        print(f"Analysis of {traj_key}  (phase offset +{phase_offset})")
        print(f"{'='*50}")
        print(f"Trajectory length: {agents_qpos[0].shape[0]} steps")

        for i, name in enumerate(agent_names):
            qpos = agents_qpos[i]
            gripper_pos = qpos[:, -1]
            print(f"\n--- {name} ---")
            print(f"  qpos shape:     {qpos.shape}")
            print(f"  gripper range:  [{gripper_pos.min():.4f}, {gripper_pos.max():.4f}]")

        # Detect phases (1-based)
        phases, debug_info = detect_phases(
            agents_qpos,
            close_ratio=close_ratio,
            still_threshold=still_threshold,
        )

        # Apply offset
        phases_offset = phases + phase_offset

        unique, counts = np.unique(phases_offset, return_counts=True)
        print(f"\nPhase distribution (with offset +{phase_offset}):")
        for u, c in zip(unique, counts):
            pname = PHASE_NAMES_132.get(int(u), f"Phase {int(u)}")
            print(f"  {pname}: {c} steps ({c / len(phases_offset) * 100:.1f}%)")

        return phases_offset, debug_info, agents_qpos


# ========== Main ==========

if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_132.h5"

    agent_names = ["panda-0", "panda-1", "panda-2"]

    output_dir = "/18141169908/zhangzaibin/multivla-pi0/phase_videos_pick_place_plate_132"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Step 1: Analyze sample trajectories ----------
    print("=" * 60)
    print("Step 1: Analyzing sample trajectories (132, phases 11-17)")
    print("=" * 60)

    for traj_id in ["traj_0", "traj_1"]:
        try:
            phases, debug_info, agents_qpos = analyze_single_trajectory(
                file_path, traj_id,
                close_ratio=0.3,
                still_threshold=0.02,
                agent_names=agent_names,
            )
            # Generate visualization
            visualize_phases(
                agents_qpos, debug_info, phases, traj_id, agent_names,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error analyzing {traj_id}: {e}")
            import traceback
            traceback.print_exc()

    # ---------- Step 2: Add phases to all trajectories ----------
    print("\n" + "=" * 60)
    print("Step 2: Adding phases to all trajectories (132, phases 11-17)")
    print("=" * 60)

    try:
        output_path = add_phases_to_h5(
            file_path,
            close_ratio=0.3,
            still_threshold=0.02,
            visualize=True,
            agent_names=agent_names,
        )
        print(f"\nDone! Check the output file: {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()



















