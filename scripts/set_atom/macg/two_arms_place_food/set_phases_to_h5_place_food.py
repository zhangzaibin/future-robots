"""
Phase segmentation for two_arms_place_food dataset.

4-phase scheme based on gripper state transitions + joint movement stability.
Task: Agent1 opens the pot lid, then Agent0 picks up meat and places it into the pot.

  Phase 1: start → A1_close        A0: "Stay still."                       A1: "Grasp the pot lid."
  Phase 2: A1_close → A1_stop      A0: "Stay still."                       A1: "Open the pot lid."
  Phase 3: A1_stop → A0_close      A0: "Pick up the meat."                 A1: "Stay still."
  Phase 4: A0_close → end          A0: "Place the meat into the pot."      A1: "Stay still."

Agent1 lifecycle: Grasp lid (open→close) → Lift lid (move→stop) → Hold (stay still)
Agent0 lifecycle: Stay still → Approach meat → Grasp (open→close) → Place into pot (close→open at last step)

Usage:
  python scripts/set_atom/macg/two_arms_place_food/set_phases_to_h5_place_food.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.035):
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


def detect_movement_stop(qpos, search_start, window_size=3, threshold=0.001):
    """Find the first step where joint movement stabilises for `window_size` consecutive steps."""
    jdiff = np.abs(np.diff(qpos[:, :7], axis=0)).sum(axis=1)
    for i in range(search_start, len(jdiff) - window_size + 1):
        if all(jdiff[j] < threshold for j in range(i, i + window_size)):
            return i
    return None


def detect_phases(agent0_gripper, agent1_gripper, agent1_qpos):
    """
    Detect 4 phases for the place-food task.

    Phase boundaries:
      1→2  A1_close:  Agent1 gripper open→close  (grasps pot lid)
      2→3  A1_stop:   Agent1 joints stabilise     (lid is open, arm stops)
      3→4  A0_close:  Agent0 gripper open→close  (picks up meat)
      end            trajectory end               (Agent0 releases meat into pot)
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    a1_trans = detect_gripper_transitions(agent1_gripper)
    a0_trans = detect_gripper_transitions(agent0_gripper)

    # --- A1_close: first CLOSE event of Agent1 ---
    a1_close = None
    for step, event in a1_trans:
        if event == "CLOSE":
            a1_close = step
            break

    if a1_close is None:
        print("  WARNING: A1_close not detected, marking entire trajectory as Phase 1")
        phases[:] = 1
        return phases

    # --- A1_stop: Agent1 movement stabilises (search 30+ steps after A1_close) ---
    search_start = min(a1_close + 30, n_steps - 3)
    a1_stop = detect_movement_stop(agent1_qpos, search_start)

    if a1_stop is None:
        print("  WARNING: A1_stop not detected")
        a1_stop = a1_close + 44  # fallback based on observed mean

    # --- A0_close: first CLOSE event of Agent0 ---
    a0_close = None
    for step, event in a0_trans:
        if event == "CLOSE":
            a0_close = step
            break

    if a0_close is None:
        print("  WARNING: A0_close not detected")

    # --- Assign phases ---
    phases[:a1_close] = 1                          # A1: Grasp the pot lid
    phases[a1_close:a1_stop] = 2                   # A1: Open the pot lid

    if a0_close is not None:
        phases[a1_stop:a0_close] = 3               # A0: Pick up the meat
        phases[a0_close:] = 4                      # A0: Place the meat into the pot
    else:
        phases[a1_stop:] = 3

    return phases


def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.035, visualize=False):
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

                    phases = detect_phases(agent0_gripper, agent1_gripper, agent1_qpos)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    counts = np.bincount(phases, minlength=5)
                    print(
                        f"  {traj_key}: "
                        + " | ".join(f"P{i}={counts[i]:3d}" for i in range(1, 5))
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
    axes[0].set_title(f"{traj_key} - Phase Segmentation (4 phases, place food)")
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
        1: "#FF6B6B", 2: "#FFA07A", 3: "#87CEEB", 4: "#9370DB",
    }
    phase_labels = {
        1: "P1: A1 Grasp lid",
        2: "P2: A1 Open lid",
        3: "P3: A0 Pick meat",
        4: "P4: A0 Place meat",
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
                start = np.where(mask)[0][0]
                ax.axvline(x=start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot saved: phase_analysis_{traj_key}.png")


def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.035):
    """Analyze a single trajectory to understand gripper/joint patterns."""
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")

        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

        phases = detect_phases(agent0_gripper, agent1_gripper, agent1_qpos)
        counts = np.bincount(phases, minlength=5)

        phase_info = {
            1: ('A0:Still          A1:Grasp lid',  'start → A1_close'),
            2: ('A0:Still          A1:Open lid',   'A1_close → A1_stop'),
            3: ('A0:Pick meat      A1:Still',      'A1_stop → A0_close'),
            4: ('A0:Place meat     A1:Still',      'A0_close → end'),
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
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/MACG/two_arms_place_food/20260317_135714.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")
    analyze_single_trajectory(file_path, "traj_2")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        gripper_threshold=0.035,
        visualize=True,
    )

    print(f"\nDone! Output file: {output_path}")
