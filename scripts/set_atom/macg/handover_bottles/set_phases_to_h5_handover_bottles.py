"""
Phase segmentation for two_arms_handover_bottles dataset.

7-phase scheme based purely on gripper state transitions.
Each arm gets its own atomic instruction based on its gripper lifecycle.

  Phase 1: start → A0_close_1         A0: "Grasp the bottle."       A1: "Stay still."
  Phase 2: A0_close_1 → A0_open_1     A0: "Handover the bottle."    A1: "Receive the bottle."
  Phase 3: A0_open_1 → A1_open_1      A0: "Stay still."             A1: "Place the bottle."
  Phase 4: A1_open_1 → A0_close_2     A0: "Grasp the bottle."       A1: "Stay still."
  Phase 5: A0_close_2 → A0_open_2     A0: "Handover the bottle."    A1: "Receive the bottle."
  Phase 6: A0_open_2 → A1_open_2      A0: "Stay still."             A1: "Place the bottle."
  Phase 7: A1_open_2 → end            A0: "Stay still."             A1: "Stay still."

Arm0 lifecycle per bottle: Grasp (open→close) → Handover (close→open)
Arm1 lifecycle per bottle: Stay still → Receive (move→close) → Place (close→open)

Usage:
  python scripts/set_atom/macg/handover_bottles/set_phases_to_h5_handover_bottles.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.036):
    """Return binary array: 1 for closed, 0 for open."""
    gripper_pos = qpos[:, -1]
    return (gripper_pos < gripper_threshold).astype(int)


def detect_gripper_transitions(gripper_states):
    """Return list of (step, event) tuples for gripper 0→1 (CLOSE) and 1→0 (OPEN)."""
    transitions = []
    for i in range(1, len(gripper_states)):
        if gripper_states[i - 1] == 0 and gripper_states[i] == 1:
            transitions.append((i, "CLOSE"))
        elif gripper_states[i - 1] == 1 and gripper_states[i] == 0:
            transitions.append((i, "OPEN"))
    return transitions


def detect_phases(agent0_gripper, agent1_gripper):
    """
    Detect 7 phases based purely on gripper transitions.

    Phase boundaries are the 6 key gripper events:
      A0_close_1, A0_open_1, A1_open_1, A0_close_2, A0_open_2, A1_open_2

    The "both closed" handover moments (~32 steps each) are merged into
    Phase 2 and Phase 5 respectively, since A0 is still in "Handover"
    and A1 is still in "Receive" during those moments.
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    a0_trans = detect_gripper_transitions(agent0_gripper)
    a1_trans = detect_gripper_transitions(agent1_gripper)

    if len(a0_trans) < 4 or len(a1_trans) < 4:
        print(f"  WARNING: Expected 4 transitions per agent, got A0={len(a0_trans)}, A1={len(a1_trans)}")
        phases[:] = 1
        return phases

    a0_close_1 = a0_trans[0][0]  # A0 grasps bottle 1
    a0_open_1 = a0_trans[1][0]   # A0 releases bottle 1
    a0_close_2 = a0_trans[2][0]  # A0 grasps bottle 2
    a0_open_2 = a0_trans[3][0]   # A0 releases bottle 2

    a1_open_1 = a1_trans[1][0]   # A1 places bottle 1
    a1_open_2 = a1_trans[3][0]   # A1 places bottle 2

    phases[:a0_close_1] = 1              # A0: Grasp,    A1: Still
    phases[a0_close_1:a0_open_1] = 2     # A0: Handover, A1: Receive
    phases[a0_open_1:a1_open_1] = 3      # A0: Still,    A1: Place
    phases[a1_open_1:a0_close_2] = 4     # A0: Grasp,    A1: Still
    phases[a0_close_2:a0_open_2] = 5     # A0: Handover, A1: Receive
    phases[a0_open_2:a1_open_2] = 6      # A0: Still,    A1: Place
    phases[a1_open_2:] = 7               # A0: Still,    A1: Still

    return phases


def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.036, visualize=False):
    """Add phase information to HDF5 file based on gripper states."""
    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

            traj_keys = sorted(
                [key for key in src_file.keys() if key.startswith("traj_")],
                key=lambda x: int(x.split("_")[1]),
            )
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                src_file.copy(traj_key, dst_file)

                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]

                    agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
                    agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

                    phases = detect_phases(agent0_gripper, agent1_gripper)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    counts = np.bincount(phases, minlength=8)
                    print(
                        f"  {traj_key}: "
                        + " | ".join(f"P{i}={counts[i]:3d}" for i in range(1, 8))
                    )

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(
                            agent0_gripper, agent1_gripper, phases,
                            agent0_qpos, agent1_qpos, traj_key,
                        )
                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, agent0_qpos, agent1_qpos, traj_key):
    """Visualize gripper states, joint movements, and detected phases."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    time_steps = np.arange(len(phases))

    axes[0].plot(time_steps, agent0_gripper, "b-", linewidth=2, label="Agent0 Gripper")
    axes[0].set_ylabel("A0 Gripper")
    axes[0].set_title(f"{traj_key} - Phase Segmentation (7 phases, gripper-based)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_steps, agent1_gripper, "r-", linewidth=2, label="Agent1 Gripper")
    axes[1].set_ylabel("A1 Gripper")
    axes[1].legend()
    axes[1].grid(True)

    a0_jdiff = np.abs(np.diff(agent0_qpos[:, :7], axis=0)).sum(axis=1)
    a1_jdiff = np.abs(np.diff(agent1_qpos[:, :7], axis=0)).sum(axis=1)
    a0_jdiff = np.append(a0_jdiff, 0)
    a1_jdiff = np.append(a1_jdiff, 0)

    axes[2].plot(time_steps, a0_jdiff, "b-", alpha=0.7, label="A0 Joint Movement")
    axes[2].plot(time_steps, a1_jdiff, "r-", alpha=0.7, label="A1 Joint Movement")
    axes[2].set_ylabel("Joint Movement")
    axes[2].legend()
    axes[2].grid(True)

    phase_colors = {
        1: "#FF6B6B", 2: "#FFA07A", 3: "#FFD700",
        4: "#87CEEB", 5: "#9370DB", 6: "#FF69B4", 7: "#DDA0DD",
    }
    for p in range(1, 8):
        mask = phases == p
        if mask.any():
            axes[3].fill_between(time_steps, 0, 1, where=mask, alpha=0.6,
                                 color=phase_colors[p], label=f"Phase {p}")
    axes[3].set_ylabel("Phase")
    axes[3].set_xlabel("Time Step")
    axes[3].legend(loc="upper right", ncol=4, fontsize=8)
    axes[3].grid(True)

    for ax in axes:
        for p in range(1, 8):
            mask = phases == p
            if mask.any():
                start = np.where(mask)[0][0]
                ax.axvline(x=start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved: phase_analysis_{traj_key}.png")


def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.036):
    """Analyze a single trajectory to understand gripper/joint patterns."""
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")

        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

        phases = detect_phases(agent0_gripper, agent1_gripper)
        counts = np.bincount(phases, minlength=8)

        phase_info = {
            1: ('A0:Grasp  A1:Still',    'start → A0_close'),
            2: ('A0:Handover A1:Receive', 'A0_close → A0_open'),
            3: ('A0:Still  A1:Place',     'A0_open → A1_open'),
            4: ('A0:Grasp  A1:Still',     'A1_open → A0_close'),
            5: ('A0:Handover A1:Receive', 'A0_close → A0_open'),
            6: ('A0:Still  A1:Place',     'A0_open → A1_open'),
            7: ('A0:Still  A1:Still',     'A1_open → end'),
        }

        print("\nPhase distribution:")
        for i in range(1, 8):
            mask = phases == i
            if mask.any():
                start = np.where(mask)[0][0]
                end = np.where(mask)[0][-1]
                desc, boundary = phase_info[i]
                print(f"  Phase {i} [{boundary:>22}] {desc:>26}: "
                      f"steps {start:4d}-{end:4d}, {counts[i]:3d} steps ({counts[i]/len(phases)*100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_handover_bottles/two_arms_handover_bottles.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")
    analyze_single_trajectory(file_path, "traj_2")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        gripper_threshold=0.036,
        visualize=True,
    )

    print(f"\nDone! Output file: {output_path}")
