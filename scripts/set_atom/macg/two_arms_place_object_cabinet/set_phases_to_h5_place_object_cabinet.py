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


def compute_total_joint_velocity(qpos):
    """Compute per-step total joint velocity (sum of abs changes across 7 joints)."""
    vel = np.zeros(len(qpos))
    vel[1:] = np.abs(np.diff(qpos[:, :7], axis=0)).sum(axis=1)
    return vel


def find_first_stable_step(velocity, search_start, window_size=3, threshold=0.002):
    """Find the first step where velocity stays below threshold for window_size consecutive frames."""
    n = len(velocity)
    search_start = min(search_start, n - window_size)
    for i in range(search_start, n - window_size + 1):
        if all(velocity[i + j] < threshold for j in range(window_size)):
            return i
    return None


def detect_phases(agent0_gripper, agent1_gripper, agent0_qpos, agent1_qpos):
    """
    Detect 4 phases for the "place object in cabinet" task.

    Task: Agent1 (panda-1) opens the drawer, Agent0 (panda-0) places an object inside.

    Phase 1: Agent1 approaches and grabs the drawer handle
             Agent0 stays idle
    Phase 2: Agent1 pulls the drawer open
             Agent0 stays idle
    Phase 3: Agent0 approaches and grabs the object
             Agent1 holds the drawer open (stationary)
    Phase 4: Agent0 carries and places the object in the drawer
             Agent1 holds the drawer open (stationary)
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    agent1_vel = compute_total_joint_velocity(agent1_qpos)
    agent0_vel = compute_total_joint_velocity(agent0_qpos)

    # --- Phase 1/2 boundary: Agent1 grips handle + stabilizes ---
    agent1_grip_start = None
    for i in range(1, n_steps):
        if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
            agent1_grip_start = i
            break

    agent1_grip_stable = None
    if agent1_grip_start is not None:
        agent1_grip_stable = find_first_stable_step(
            agent1_vel, agent1_grip_start + 3, window_size=3, threshold=0.002
        )

    # --- Phase 2/3 boundary: Agent1 finishes pulling drawer open ---
    agent1_pull_done = None
    if agent1_grip_start is not None:
        agent1_pull_done = find_first_stable_step(
            agent1_vel, agent1_grip_start + 30, window_size=3, threshold=0.002
        )

    # --- Phase 3/4 boundary: Agent0 grips object + stabilizes ---
    agent0_grip_start = None
    for i in range(1, n_steps):
        if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break

    agent0_grip_stable = None
    if agent0_grip_start is not None:
        agent0_grip_stable = find_first_stable_step(
            agent0_vel, agent0_grip_start + 3, window_size=3, threshold=0.002
        )

    # --- Assign phases ---
    if agent1_grip_stable is not None:
        phases[:agent1_grip_stable] = 1

        if agent1_pull_done is not None:
            phases[agent1_grip_stable:agent1_pull_done] = 2

            if agent0_grip_stable is not None:
                phases[agent1_pull_done:agent0_grip_stable] = 3
                phases[agent0_grip_stable:] = 4
            else:
                phases[agent1_pull_done:] = 3
                print("Warning: agent0_grip_stable not detected")
        else:
            phases[agent1_grip_stable:] = 2
            print("Warning: agent1_pull_done not detected")
    else:
        phases[:] = 1
        print("Warning: agent1_grip_stable not detected")

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

                    phases = detect_phases(agent0_gripper, agent1_gripper, agent0_qpos, agent1_qpos)

                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key)

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key):
    plt.figure(figsize=(14, 8))

    time_steps = np.arange(len(phases))

    plt.subplot(3, 1, 1)
    plt.plot(time_steps, agent1_gripper, "r-", linewidth=2, label="Agent1 Gripper (drawer)")
    plt.ylabel("Agent1 Gripper")
    plt.title(f"{traj_key} - Place Object in Cabinet: Gripper States and Phases")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, agent0_gripper, "b-", linewidth=2, label="Agent0 Gripper (object)")
    plt.ylabel("Agent0 Gripper")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    labels = {
        1: "P1: A1 grab handle",
        2: "P2: A1 pull drawer",
        3: "P3: A0 grab object",
        4: "P4: A0 place object",
    }
    for phase_val in [1, 2, 3, 4]:
        mask = phases == phase_val
        if mask.any():
            plt.fill_between(
                time_steps, 0, 1, where=mask,
                alpha=0.4, color=colors[phase_val], label=labels[phase_val],
            )
    plt.ylabel("Phase")
    plt.xlabel("Time Step")
    plt.ylim(0, 1.2)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)

    plt.tight_layout()
    output_dir = "phase_videos_place_object_cabinet"
    import os
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
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

        print("\nKey gripper events:")
        for i in range(1, len(agent1_gripper)):
            if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
                print(f"  Agent1 grip CLOSE at step {i}")
            if agent1_gripper[i - 1] == 1 and agent1_gripper[i] == 0:
                print(f"  Agent1 grip OPEN at step {i}")
        for i in range(1, len(agent0_gripper)):
            if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
                print(f"  Agent0 grip CLOSE at step {i}")
            if agent0_gripper[i - 1] == 1 and agent0_gripper[i] == 0:
                print(f"  Agent0 grip OPEN at step {i}")

        phases = detect_phases(agent0_gripper, agent1_gripper, agent0_qpos, agent1_qpos)
        phase_counts = np.bincount(phases)

        print("\nPhase distribution:")
        phase_names = {
            1: "Agent1 approach+grab handle",
            2: "Agent1 pull drawer open",
            3: "Agent0 approach+grab object",
            4: "Agent0 place object in drawer",
        }
        for i, count in enumerate(phase_counts):
            if count > 0:
                name = phase_names.get(i, "unknown")
                print(f"  Phase {i} ({name}): {count} steps ({count / len(phases) * 100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_place_object_cabinet/two_arms_place_object_cabinet.h5"

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
