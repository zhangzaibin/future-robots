"""
7-Phase Detection for placefactory_123_123 order (gripper-only):
Arm action order: arm0 -> arm1 -> arm2

Phase 1: start        -> arm0 closes
Phase 2: arm0 closes  -> arm0 opens
Phase 3: arm0 opens   -> arm1 closes
Phase 4: arm1 closes  -> arm1 opens
Phase 5: arm1 opens   -> arm2 closes
Phase 6: arm2 closes  -> arm2 opens
Phase 7: arm2 opens   -> end
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


def get_per_agent_gripper_thresholds(gripper_pos, close_ratio=0.3):
    """
    Calculate per-agent gripper thresholds based on its own value range.
    Handles cases where different agents have different gripper ranges.
    """
    gripper_min = gripper_pos.min()
    gripper_max = gripper_pos.max()
    gripper_range = gripper_max - gripper_min

    if gripper_range < 0.005:
        close_threshold = (gripper_min + gripper_max) / 2
        open_threshold = close_threshold + 0.001
    else:
        close_threshold = gripper_min + gripper_range * close_ratio
        open_threshold = gripper_min + gripper_range * (close_ratio + 0.2)

    return close_threshold, open_threshold


def apply_unified_gripper_analysis(gripper_pos, close_threshold, open_threshold):
    """
    Apply gripper thresholds with hysteresis to compute binary gripper state.
    Returns: 1 for closed, 0 for open.
    """
    n_steps = len(gripper_pos)
    gripper_state = np.zeros(n_steps, dtype=int)

    gripper_state[0] = 1 if gripper_pos[0] <= close_threshold else 0

    for i in range(1, n_steps):
        prev_state = gripper_state[i - 1]
        curr_pos = gripper_pos[i]
        if prev_state == 0:
            gripper_state[i] = 1 if curr_pos <= close_threshold else 0
        else:
            gripper_state[i] = 0 if curr_pos >= open_threshold else 1

    return gripper_state


def find_gripper_events(gripper_state):
    """
    Find all gripper close (0->1) and open (1->0) events.
    Returns: (closes, opens) — lists of indices.
    """
    closes, opens = [], []
    for i in range(1, len(gripper_state)):
        if gripper_state[i - 1] == 0 and gripper_state[i] == 1:
            closes.append(i)
        elif gripper_state[i - 1] == 1 and gripper_state[i] == 0:
            opens.append(i)
    return closes, opens


def detect_7_phases(first_gripper, second_gripper, third_gripper, arm_names=("arm0", "arm1", "arm2")):
    """
    Generic 7-phase detection by alternating close/open events of three arms in order.

    Phase 1: start           -> first closes
    Phase 2: first closes    -> first opens
    Phase 3: first opens     -> second closes
    Phase 4: second closes   -> second opens
    Phase 5: second opens    -> third closes
    Phase 6: third closes    -> third opens
    Phase 7: third opens     -> end
    """
    n_steps = len(first_gripper)
    phases = np.ones(n_steps, dtype=int)

    first_closes,  first_opens  = find_gripper_events(first_gripper)
    second_closes, second_opens = find_gripper_events(second_gripper)
    third_closes,  third_opens  = find_gripper_events(third_gripper)

    print(f"  {arm_names[0]} closes: {first_closes},  opens: {first_opens}")
    print(f"  {arm_names[1]} closes: {second_closes}, opens: {second_opens}")
    print(f"  {arm_names[2]} closes: {third_closes},  opens: {third_opens}")

    # --- Phase 1 -> 2: first arm closes ---
    if not first_closes:
        print(f"Warning: {arm_names[0]} close not detected"); return phases
    p1_end = first_closes[0]
    phases[:p1_end] =11

    # --- Phase 2 -> 3: first arm opens ---
    valid = [t for t in first_opens if t > p1_end]
    if not valid:
        print(f"Warning: {arm_names[0]} open not detected"); phases[p1_end:] = 2; return phases
    p2_end = valid[0]
    phases[p1_end:p2_end] = 12

    # --- Phase 3 -> 4: second arm closes ---
    valid = [t for t in second_closes if t >= p2_end]
    if not valid:
        print(f"Warning: {arm_names[1]} close not detected"); phases[p2_end:] = 3; return phases
    p3_end = valid[0]
    phases[p2_end:p3_end] = 13

    # --- Phase 4 -> 5: second arm opens ---
    valid = [t for t in second_opens if t > p3_end]
    if not valid:
        print(f"Warning: {arm_names[1]} open not detected"); phases[p3_end:] = 4; return phases
    p4_end = valid[0]
    phases[p3_end:p4_end] = 14

    # --- Phase 5 -> 6: third arm closes ---
    valid = [t for t in third_closes if t >= p4_end]
    if not valid:
        print(f"Warning: {arm_names[2]} close not detected"); phases[p4_end:] = 5; return phases
    p5_end = valid[0]
    phases[p4_end:p5_end] = 15

    # --- Phase 6 -> 7: third arm opens ---
    valid = [t for t in third_opens if t > p5_end]
    if not valid:
        print(f"Warning: {arm_names[2]} open not detected"); phases[p5_end:] = 6; return phases
    p6_end = valid[0]
    phases[p5_end:p6_end] = 16
    phases[p6_end:] = 17

    phase_counts = {p: int(np.sum(phases == p)) for p in range(1, 8)}
    print(f"  Phase distribution: {phase_counts}")
    return phases


def detect_phases(agent0_gripper, agent1_gripper, agent2_gripper):
    """
    7-phase detection for 123_123 order: agent0 -> agent1 -> agent2.
    """
    return detect_7_phases(
        agent0_gripper, agent1_gripper, agent2_gripper,
        arm_names=("agent0", "agent1", "agent2"),
    )


def add_phases_to_h5(file_path, output_path=None, close_ratio=0.3, visualize=False):
    """
    Add 7-phase labels to HDF5 file based on gripper close/open events.
    """
    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

            traj_keys = [key for key in src_file.keys() if key.startswith("traj_")]
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                print(f"\nProcessing {traj_key}...")

                src_file.copy(traj_key, dst_file)

                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]
                    agent2_qpos = src_file[f"{traj_key}/obs/agent/panda-2/qpos"][:]

                    agent0_close_th, agent0_open_th = get_per_agent_gripper_thresholds(
                        agent0_qpos[:, -1], close_ratio
                    )
                    agent1_close_th, agent1_open_th = get_per_agent_gripper_thresholds(
                        agent1_qpos[:, -1], close_ratio
                    )
                    agent2_close_th, agent2_open_th = get_per_agent_gripper_thresholds(
                        agent2_qpos[:, -1], close_ratio
                    )

                    agent0_gripper = apply_unified_gripper_analysis(
                        agent0_qpos[:, -1], agent0_close_th, agent0_open_th
                    )
                    agent1_gripper = apply_unified_gripper_analysis(
                        agent1_qpos[:, -1], agent1_close_th, agent1_open_th
                    )
                    agent2_gripper = apply_unified_gripper_analysis(
                        agent2_qpos[:, -1], agent2_close_th, agent2_open_th
                    )

                    phases = detect_phases(agent0_gripper, agent1_gripper, agent2_gripper)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)
                    print(f"  Added phases: {np.bincount(phases)}")

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(
                            agent0_qpos[:, -1], agent1_qpos[:, -1], agent2_qpos[:, -1],
                            agent0_gripper, agent1_gripper, agent2_gripper,
                            phases, traj_key,
                        )

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(
    agent0_raw, agent1_raw, agent2_raw,
    agent0_gripper, agent1_gripper, agent2_gripper,
    phases, traj_key, output_dir=".",
):
    """
    Visualize raw gripper positions, binary gripper states, and detected phases.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))
    time_steps = np.arange(len(phases))

    ax = axes[0]
    ax.plot(time_steps, agent0_raw, "b-", linewidth=1, label="panda-0 raw", alpha=0.8)
    ax.plot(time_steps, agent1_raw, "r-", linewidth=1, label="panda-1 raw", alpha=0.8)
    ax.plot(time_steps, agent2_raw, "g-", linewidth=1, label="panda-2 raw", alpha=0.8)
    ax.set_ylabel("Gripper Position")
    ax.set_title(f"{traj_key} - 7 Phase Analysis (123_123: arm0→arm1→arm2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time_steps, agent0_gripper,        "b-", linewidth=2, label="panda-0 (1st)")
    ax.plot(time_steps, agent1_gripper + 0.05, "r-", linewidth=2, label="panda-1 (2nd)")
    ax.plot(time_steps, agent2_gripper + 0.10, "g-", linewidth=2, label="panda-2 (3rd)")
    ax.set_ylabel("Gripper State (Binary)")
    ax.set_ylim(-0.1, 1.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    phase_colors = ["#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A", "#C39BD3", "#F0B27A", "#ABB2B9"]
    phase_labels = [
        "Ph1: approaching (before arm0 grabs)",
        "Ph2: arm0 closed (carrying/placing)",
        "Ph3: arm0 released → arm1 grabs",
        "Ph4: arm1 closed (carrying/placing)",
        "Ph5: arm1 released → arm2 grabs",
        "Ph6: arm2 closed (carrying/placing)",
        "Ph7: completion",
    ]
    for phase in range(1, 8):
        mask = phases == phase
        if np.any(mask):
            ax.fill_between(
                time_steps, 0, mask * phase, alpha=0.7,
                color=phase_colors[phase - 1], label=phase_labels[phase - 1],
            )
    ax.plot(time_steps, phases, "k-", linewidth=1.5)
    ax.set_ylabel("Phase")
    ax.set_xlabel("Time Step")
    ax.set_ylim(0.5, 7.5)
    ax.set_yticks(range(1, 8))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = f"{output_dir}/phase_analysis_{traj_key}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to: {out_file}")


def analyze_single_trajectory(file_path, traj_key, close_ratio=0.3):
    """
    Analyze a single trajectory to verify phase detection.
    """
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
        agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")

        thresholds = []
        for i, qpos in enumerate([agent0_qpos, agent1_qpos, agent2_qpos]):
            close_th, open_th = get_per_agent_gripper_thresholds(qpos[:, -1], close_ratio)
            thresholds.append((close_th, open_th))
            print(
                f"Agent{i} gripper range: [{qpos[:, -1].min():.4f}, {qpos[:, -1].max():.4f}]  "
                f"close_th={close_th:.4f}  open_th={open_th:.4f}"
            )

        agent0_gripper = apply_unified_gripper_analysis(
            agent0_qpos[:, -1], *thresholds[0]
        )
        agent1_gripper = apply_unified_gripper_analysis(
            agent1_qpos[:, -1], *thresholds[1]
        )
        agent2_gripper = apply_unified_gripper_analysis(
            agent2_qpos[:, -1], *thresholds[2]
        )

        phases = detect_phases(agent0_gripper, agent1_gripper, agent2_gripper)

        print("\nPhase distribution:")
        for p in range(1, 8):
            count = int(np.sum(phases == p))
            if count > 0:
                print(f"  Phase {p}: {count} steps ({count / len(phases) * 100:.1f}%)")

        return agent0_gripper, agent1_gripper, agent2_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/three_arms/three_arms_place_objects_123_132/20260411_075959.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        close_ratio=0.3,
        visualize=True,
    )

    print(f"\nDone! Check the output file: {output_path}")
