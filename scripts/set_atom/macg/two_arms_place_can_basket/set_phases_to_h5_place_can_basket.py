"""
Phase segmentation for two_arms_place_can_basket dataset.

4-phase scheme based purely on gripper state transitions.

  Phase 1: start → A1_close          A0: "Approach can"      A1: "Approach basket"
  Phase 2: A1_close → A0_close       A0: "Pick the can"      A1: "Hold the basket"
  Phase 3: A0_close → A0_open        A0: "Place the can"     A1: "Hold the basket"
  Phase 4: A0_open → end             A0: "Return"            A1: "Release basket"

A0 lifecycle: approach → close (pick) → open (place) → return
A1 lifecycle: approach → close (hold basket) → [open at end]

Usage:
  python scripts/set_atom/macg/two_arms_place_can_basket/set_phases_to_h5_place_can_basket.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.03):
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
    Detect 4 phases based purely on gripper transitions.

    Phase boundaries:
      A1_close: Agent1 grasps the basket
      A0_close: Agent0 grasps the can  (PICK)
      A0_open:  Agent0 releases the can (PLACE)
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    a0_trans = detect_gripper_transitions(agent0_gripper)
    a1_trans = detect_gripper_transitions(agent1_gripper)

    # Extract key events
    a1_close = a1_trans[0][0] if len(a1_trans) >= 1 else None
    a0_close = a0_trans[0][0] if len(a0_trans) >= 1 else None
    a0_open  = a0_trans[1][0] if len(a0_trans) >= 2 else None

    if a1_close is None:
        print("  WARNING: A1 never closes (basket not grasped)")
        phases[:] = 1
        return phases

    if a0_close is None:
        print("  WARNING: A0 never closes (can not grasped)")
        phases[:a1_close] = 1
        phases[a1_close:] = 2
        return phases

    if a0_open is None:
        print("  WARNING: A0 never opens (can not released)")
        phases[:a1_close] = 1
        phases[a1_close:a0_close] = 2
        phases[a0_close:] = 3
        return phases

    phases[:a1_close] = 1           # A0: Approach can,   A1: Approach basket
    phases[a1_close:a0_close] = 2   # A0: Pick the can,   A1: Hold the basket
    phases[a0_close:a0_open] = 3    # A0: Place the can,  A1: Hold the basket
    phases[a0_open:] = 4            # A0: Return,         A1: Release basket

    return phases


def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.03, visualize=False):
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

                    counts = np.bincount(phases, minlength=5)
                    print(
                        f"  {traj_key}: "
                        + " | ".join(f"P{i}={counts[i]:3d}" for i in range(1, 5))
                    )

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(agent0_gripper, agent1_gripper, phases,
                                         agent0_qpos, agent1_qpos, traj_key)

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, agent0_qpos, agent1_qpos, traj_key):
    """Visualize gripper states and detected phases."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    time_steps = np.arange(len(phases))

    axes[0].plot(time_steps, agent0_qpos[:, -1], "b-", linewidth=1.5, label="A0 Gripper pos (raw)")
    axes[0].plot(time_steps, agent0_gripper * agent0_qpos[:, -1].max(), "b--",
                 linewidth=1, alpha=0.6, label="A0 Gripper (binary)")
    axes[0].set_ylabel("A0 Gripper")
    axes[0].set_title(f"{traj_key} - Phase Segmentation (4 phases, gripper-based)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].plot(time_steps, agent1_qpos[:, -1], "r-", linewidth=1.5, label="A1 Gripper pos (raw)")
    axes[1].plot(time_steps, agent1_gripper * agent1_qpos[:, -1].max(), "r--",
                 linewidth=1, alpha=0.6, label="A1 Gripper (binary)")
    axes[1].set_ylabel("A1 Gripper")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    a0_jdiff = np.abs(np.diff(agent0_qpos[:, :7], axis=0)).sum(axis=1)
    a1_jdiff = np.abs(np.diff(agent1_qpos[:, :7], axis=0)).sum(axis=1)
    axes[2].plot(time_steps[:-1], a0_jdiff, "b-", alpha=0.7, label="A0 Joint Movement")
    axes[2].plot(time_steps[:-1], a1_jdiff, "r-", alpha=0.7, label="A1 Joint Movement")
    axes[2].set_ylabel("Joint Movement")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    phase_colors = {1: "#87CEEB", 2: "#FFD700", 3: "#FF6B6B", 4: "#90EE90"}
    phase_labels = {
        1: "P1: Approach",
        2: "P2: Pick (A0 closed)",
        3: "P3: Place (A0→open)",
        4: "P4: Return",
    }
    for p in range(1, 5):
        mask = phases == p
        if mask.any():
            axes[3].fill_between(time_steps, 0, 1, where=mask, alpha=0.6,
                                 color=phase_colors[p], label=phase_labels[p])
    axes[3].set_ylabel("Phase")
    axes[3].set_xlabel("Time Step")
    axes[3].legend(loc="upper right", ncol=4, fontsize=8)
    axes[3].grid(True)

    for ax in axes:
        for p in range(1, 5):
            mask = phases == p
            if mask.any():
                ax.axvline(x=np.where(mask)[0][0], color="gray",
                           linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved: phase_analysis_{traj_key}.png")


def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.03):
    """Analyze a single trajectory to verify gripper patterns and phase assignment."""
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")

        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]
        print(f"A0 gripper range: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"A1 gripper range: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")

        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

        a0_trans = detect_gripper_transitions(agent0_gripper)
        a1_trans = detect_gripper_transitions(agent1_gripper)
        print(f"\nA0 gripper transitions: {a0_trans}")
        print(f"A1 gripper transitions: {a1_trans}")

        phases = detect_phases(agent0_gripper, agent1_gripper)
        counts = np.bincount(phases, minlength=5)

        phase_info = {
            1: ("A0:Approach   A1:Approach", "start → A1_close"),
            2: ("A0:Pick       A1:Hold",     "A1_close → A0_close"),
            3: ("A0:Place      A1:Hold",     "A0_close → A0_open"),
            4: ("A0:Return     A1:Release",  "A0_open → end"),
        }

        print("\nPhase distribution:")
        for i in range(1, 5):
            mask = phases == i
            if mask.any():
                start = np.where(mask)[0][0]
                end = np.where(mask)[0][-1]
                desc, boundary = phase_info[i]
                print(f"  Phase {i} [{boundary:>22}] {desc}: "
                      f"steps {start:4d}-{end:4d}, {counts[i]:3d} steps ({counts[i]/len(phases)*100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/two_arms_place_can_basket.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")
    analyze_single_trajectory(file_path, "traj_2")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        gripper_threshold=0.03,
        visualize=True,
    )

    print(f"\nDone! Output file: {output_path}")
