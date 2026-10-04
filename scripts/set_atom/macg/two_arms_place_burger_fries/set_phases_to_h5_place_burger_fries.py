import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, gripper_threshold=0.035):
    """
    Analyze gripper state from qpos data.
    Returns binary array: 1 for closed/gripping, 0 for open
    """
    gripper_pos = qpos[:, -1]
    gripper_closed = gripper_pos < gripper_threshold
    return gripper_closed.astype(int)


def detect_phases(agent0_gripper, agent1_gripper):
    """
    Detect phases based on gripper state transitions only:
    Phase 1: Agent0 approaching (gripper open) -> until agent0 grips
    Phase 2: Agent0 carrying & placing (gripper closed) -> until agent0 releases
    Phase 3: Agent1 approaching (gripper open) -> until agent1 grips
    Phase 4: Agent1 carrying & placing (gripper closed) -> until agent1 releases
    Phase 5: Completion / return to rest
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    agent0_grip_start = None
    agent0_grip_end = None
    agent1_grip_start = None
    agent1_grip_end = None

    for i in range(1, n_steps):
        if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break

    if agent0_grip_start is not None:
        for i in range(agent0_grip_start + 1, n_steps):
            if agent0_gripper[i - 1] == 1 and agent0_gripper[i] == 0:
                agent0_grip_end = i
                break

    search_start = agent0_grip_end if agent0_grip_end is not None else 0
    for i in range(search_start + 1, n_steps):
        if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
            agent1_grip_start = i
            break

    if agent1_grip_start is not None:
        for i in range(agent1_grip_start + 1, n_steps):
            if agent1_gripper[i - 1] == 1 and agent1_gripper[i] == 0:
                agent1_grip_end = i
                break

    if agent0_grip_start is not None:
        phases[:agent0_grip_start] = 1

        if agent0_grip_end is not None:
            phases[agent0_grip_start:agent0_grip_end] = 2

            if agent1_grip_start is not None:
                phases[agent0_grip_end:agent1_grip_start] = 3

                if agent1_grip_end is not None:
                    phases[agent1_grip_start:agent1_grip_end] = 4
                    phases[agent1_grip_end:] = 5
                else:
                    phases[agent1_grip_start:] = 4
                    print("Warning: agent1_grip_end not detected")
            else:
                phases[agent0_grip_end:] = 3
                print("Warning: agent1_grip_start not detected")
        else:
            phases[agent0_grip_start:] = 2
            print("Warning: agent0_grip_end not detected")
    else:
        phases[:] = 1
        print("Warning: agent0_grip_start not detected")

    return phases


def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.035, visualize=False):
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

                    agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
                    agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

                    phases = detect_phases(agent0_gripper, agent1_gripper)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key)

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key):
    plt.figure(figsize=(12, 6))

    time_steps = np.arange(len(phases))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, agent0_gripper, "b-", linewidth=2, label="Agent0 Gripper")
    plt.ylabel("Agent0 Gripper")
    plt.title(f"{traj_key} - Gripper States and Phases")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, agent1_gripper, "r-", linewidth=2, label="Agent1 Gripper")
    plt.ylabel("Agent1 Gripper")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, phases, "g-", linewidth=2, marker="o", markersize=3)
    plt.ylabel("Phase")
    plt.xlabel("Time Step")
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.close()


def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.035):
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")
        print(f"Agent0 qpos shape: {agent0_qpos.shape}")
        print(f"Agent1 qpos shape: {agent1_qpos.shape}")

        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]

        print("\nGripper position ranges:")
        print(f"Agent0 gripper: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"Agent1 gripper: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")

        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)

        print("\nGripper states (closed ratio):")
        print(f"Agent0: {agent0_gripper.mean():.3f}")
        print(f"Agent1: {agent1_gripper.mean():.3f}")

        phases = detect_phases(agent0_gripper, agent1_gripper)
        phase_counts = np.bincount(phases)

        print("\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count / len(phases) * 100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_place_burger_fries/two_arms_place_burger_fries.h5"

    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")

    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        gripper_threshold=0.035,
        visualize=True,
    )

    print(f"\nDone! Check the output file: {output_path}")
