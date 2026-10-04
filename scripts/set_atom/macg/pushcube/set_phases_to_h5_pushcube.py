import h5py
import matplotlib.pyplot as plt
import numpy as np


def detect_phases(agent0_tcp_z, agent1_tcp_z, agent0_qpos, agent1_qpos,
                  z_threshold=0.05, vel_smooth_window=10, vel_move_threshold=0.003):
    """
    Detect phases for the push-cube task using TCP z-height and joint velocity.

    Unlike place_bowls/stack_cubes, grippers stay closed throughout so we cannot
    rely on gripper transitions.  Instead we use:
      - TCP z-height crossing below z_threshold  -> arm reached table level
      - Smoothed joint velocity                   -> handoff between agents

    Phase 1: Agent0 approaches cube (descends from rest to table level)
    Phase 2: Agent0 pushes cube to target and returns to rest
    Phase 3: Agent1 approaches cube (descends from rest to table level)
    Phase 4: Agent1 pushes cube to target and returns to rest
    """
    n_steps = len(agent0_tcp_z)
    phases = np.zeros(n_steps, dtype=int)

    # --- 1. Agent0 z-drop: first frame where TCP z < z_threshold ----------
    a0_z_drop = None
    for i in range(n_steps):
        if agent0_tcp_z[i] < z_threshold:
            a0_z_drop = i
            break

    # --- 2. Handoff: find when Agent0 stops and Agent1 starts via joint vel -
    a0_joints = agent0_qpos[:, :7]
    a1_joints = agent1_qpos[:, :7]
    a0_jv = np.sum(np.abs(np.diff(a0_joints, axis=0)), axis=1)
    a1_jv = np.sum(np.abs(np.diff(a1_joints, axis=0)), axis=1)

    kernel = np.ones(vel_smooth_window) / vel_smooth_window
    a0_smooth = np.convolve(a0_jv, kernel, mode="valid")
    a1_smooth = np.convolve(a1_jv, kernel, mode="valid")

    a0_last_move = 0
    for i in range(len(a0_smooth) - 1, -1, -1):
        if a0_smooth[i] > vel_move_threshold:
            a0_last_move = i + vel_smooth_window
            break

    a1_first_move = n_steps
    for i in range(len(a1_smooth)):
        if a1_smooth[i] > vel_move_threshold:
            a1_first_move = i
            break

    handoff = (a0_last_move + a1_first_move) // 2

    # --- 3. Agent1 z-drop: first frame (after handoff) where TCP z < z_threshold
    a1_z_drop = None
    for i in range(handoff, n_steps):
        if agent1_tcp_z[i] < z_threshold:
            a1_z_drop = i
            break

    # --- Assign phases -------------------------------------------------------
    if a0_z_drop is not None:
        phases[:a0_z_drop] = 1

        phases[a0_z_drop:handoff] = 2

        if a1_z_drop is not None:
            phases[handoff:a1_z_drop] = 3
            phases[a1_z_drop:] = 4
        else:
            phases[handoff:] = 3
            print("Warning: agent1 z-drop not detected")
    else:
        phases[:handoff] = 1
        phases[handoff:] = 3
        print("Warning: agent0 z-drop not detected, falling back to 2-phase split")

    return phases


def add_phases_to_h5(file_path, output_path=None, z_threshold=0.05, visualize=False):
    """
    Add phase information to HDF5 file based on TCP z-height and joint velocity.
    """
    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

            traj_keys = [key for key in src_file.keys() if key.startswith("traj_")]
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                print(f"Processing {traj_key}...")

                src_file.copy(traj_key, dst_file)

                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]
                    agent0_tcp = src_file[f"{traj_key}/obs/extra/left_arm_tcp"][:]
                    agent1_tcp = src_file[f"{traj_key}/obs/extra/right_arm_tcp"][:]

                    agent0_tcp_z = agent0_tcp[:, 2]
                    agent1_tcp_z = agent1_tcp[:, 2]

                    phases = detect_phases(
                        agent0_tcp_z, agent1_tcp_z,
                        agent0_qpos, agent1_qpos,
                        z_threshold=z_threshold,
                    )

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(
                            agent0_tcp_z, agent1_tcp_z, phases, traj_key,
                            z_threshold=z_threshold,
                        )

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_tcp_z, agent1_tcp_z, phases, traj_key, z_threshold=0.05):
    """
    Visualize TCP z-heights and detected phases.
    """
    plt.figure(figsize=(12, 6))

    time_steps = np.arange(len(phases))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, agent0_tcp_z, "b-", linewidth=2, label="Agent0 TCP z")
    plt.axhline(y=z_threshold, color="gray", linestyle="--", alpha=0.7, label=f"z={z_threshold}")
    plt.ylabel("Agent0 TCP z")
    plt.title(f"{traj_key} - TCP z-height and Phases")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, agent1_tcp_z, "r-", linewidth=2, label="Agent1 TCP z")
    plt.axhline(y=z_threshold, color="gray", linestyle="--", alpha=0.7, label=f"z={z_threshold}")
    plt.ylabel("Agent1 TCP z")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, phases, "g-", linewidth=2, marker="o", markersize=3)
    plt.ylabel("Phase")
    plt.xlabel("Time Step")
    plt.ylim(0.5, 4.5)
    plt.yticks([1, 2, 3, 4])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()


def analyze_single_trajectory(file_path, traj_key, z_threshold=0.05):
    """
    Analyze a single trajectory to understand movement patterns.
    """
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
        agent0_tcp = f[f"{traj_key}/obs/extra/left_arm_tcp"][:]
        agent1_tcp = f[f"{traj_key}/obs/extra/right_arm_tcp"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")

        agent0_tcp_z = agent0_tcp[:, 2]
        agent1_tcp_z = agent1_tcp[:, 2]

        print(f"\nTCP z ranges:")
        print(f"Agent0: [{agent0_tcp_z.min():.4f}, {agent0_tcp_z.max():.4f}]")
        print(f"Agent1: [{agent1_tcp_z.min():.4f}, {agent1_tcp_z.max():.4f}]")

        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]
        print(f"\nGripper ranges (for reference):")
        print(f"Agent0: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"Agent1: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")

        phases = detect_phases(
            agent0_tcp_z, agent1_tcp_z,
            agent0_qpos, agent1_qpos,
            z_threshold=z_threshold,
        )
        phase_counts = np.bincount(phases)

        print("\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count / len(phases) * 100:.1f}%)")

        return agent0_tcp_z, agent1_tcp_z, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_push_cube/two_arms_push_cube.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        z_threshold=0.05,
        visualize=True,
    )

    print(f"\nDone! Check the output file: {output_path}")
