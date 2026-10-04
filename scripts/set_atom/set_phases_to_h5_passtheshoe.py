import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_gripper_state(qpos, close_threshold=0.015, open_threshold=0.04):
    """
    Analyze gripper state from qpos data with hysteresis.
    Assuming the last dimension of qpos contains gripper information.
    Returns binary array: 1 for closed/gripping, 0 for open

    Args:
        qpos: Array of joint positions
        close_threshold: Gripper position threshold (<=) considered closed
        open_threshold: Gripper position threshold (>=) considered open (hysteresis)
    """
    # Typically gripper position is in the last column of qpos
    gripper_pos = qpos[:, -1]  # Last joint is usually gripper

    # Initialize state array
    n_steps = len(gripper_pos)
    gripper_state = np.zeros(n_steps, dtype=int)

    # First state based on close threshold
    gripper_state[0] = 1 if gripper_pos[0] <= close_threshold else 0

    # Apply hysteresis for remaining states
    for i in range(1, n_steps):
        prev_state = gripper_state[i - 1]
        curr_pos = gripper_pos[i]

        if prev_state == 0:  # Was open
            gripper_state[i] = 1 if curr_pos <= close_threshold else 0
        else:  # Was closed
            gripper_state[i] = 0 if curr_pos >= open_threshold else 1

    return gripper_state


def detect_phases(agent0_gripper, agent1_gripper):
    """
    Detect phases based on gripper states:
    Phase 1: Before agent0 grips (agent0 open)
    Phase 2: Agent0 gripping to agent0 release
    Phase 3: Agent0 released to agent1 grips
    Phase 4: Agent1 gripping to end
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)

    # Find grip/release events
    agent0_grip_start = None
    agent0_grip_end = None
    agent1_grip_start = None

    # Detect agent0 grip start (transition from 0 to 1)
    for i in range(1, n_steps):
        if agent0_gripper[i - 1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break

    # Detect agent0 grip end (transition from 1 to 0)
    if agent0_grip_start is not None:
        for i in range(agent0_grip_start + 1, n_steps):
            if agent0_gripper[i - 1] == 1 and agent0_gripper[i] == 0:
                agent0_grip_end = i
                break

    # Detect agent1 grip start (transition from 0 to 1)
    start_search = agent0_grip_end if agent0_grip_end is not None else 0
    for i in range(start_search + 1, n_steps):
        if agent1_gripper[i - 1] == 0 and agent1_gripper[i] == 1:
            agent1_grip_start = i
            break

    # Assign phases
    # Phase 1: Before agent0 grips
    if agent0_grip_start is not None:
        phases[:agent0_grip_start] = 1

        # Phase 2: Agent0 gripping
        if agent0_grip_end is not None:
            phases[agent0_grip_start:agent0_grip_end] = 2

            # Phase 3: Between agent0 release and agent1 grip
            if agent1_grip_start is not None:
                phases[agent0_grip_end:agent1_grip_start] = 3
                # Phase 4: Agent1 gripping to end
                phases[agent1_grip_start:] = 4
            else:
                # No agent1 grip detected, rest is phase 3
                phases[agent0_grip_end:] = 3
        else:
            # Agent0 never releases, rest is phase 2
            phases[agent0_grip_start:] = 2
    # No agent0 grip detected, check for agent1 only
    elif agent1_grip_start is not None:
        phases[:agent1_grip_start] = 1
        phases[agent1_grip_start:] = 4
    else:
        # No grips detected, all phase 1
        phases[:] = 1

    return phases


def add_phases_to_h5(file_path, output_path=None, close_threshold=0.015, open_threshold=0.04, visualize=False):
    """
    Add phase information to HDF5 file based on gripper states.
    """
    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    # Read the original file
    with h5py.File(file_path, "r") as src_file:
        # Create new file
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

            # Copy all data and add phases
            traj_keys = [key for key in src_file.keys() if key.startswith("traj_")]
            print(f"Found {len(traj_keys)} trajectories")

            for traj_key in traj_keys:
                print(f"Processing {traj_key}...")

                # Copy the entire trajectory group
                src_file.copy(traj_key, dst_file)

                # Get gripper data
                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]

                    # Analyze gripper states
                    agent0_gripper = analyze_gripper_state(agent0_qpos, close_threshold, open_threshold)
                    agent1_gripper = analyze_gripper_state(agent1_qpos, close_threshold, open_threshold)

                    # Detect phases
                    phases = detect_phases(agent0_gripper, agent1_gripper)

                    # Add phases to the destination file
                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    # Visualization for first few trajectories
                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key)

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")

    print(f"Completed! Output saved to: {output_path}")
    return output_path


def visualize_phases(agent0_gripper, agent1_gripper, phases, traj_key):
    """
    Visualize gripper states and detected phases
    """
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
    plt.ylim(0.5, 4.5)
    plt.yticks([1, 2, 3, 4])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"phase_analysis_{traj_key}.png", dpi=150, bbox_inches="tight")
    plt.show()


def analyze_single_trajectory(file_path, traj_key, close_threshold=0.015, open_threshold=0.04):
    """
    Analyze a single trajectory to understand gripper patterns
    """
    with h5py.File(file_path, "r") as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")
        print(f"Agent0 qpos shape: {agent0_qpos.shape}")
        print(f"Agent1 qpos shape: {agent1_qpos.shape}")

        # Analyze gripper positions
        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]

        print("\nGripper position ranges:")
        print(f"Agent0 gripper: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"Agent1 gripper: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")

        # Analyze gripper states
        agent0_gripper = analyze_gripper_state(agent0_qpos, close_threshold, open_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, close_threshold, open_threshold)

        print("\nGripper states (closed ratio):")
        print(f"Agent0: {agent0_gripper.mean():.3f}")
        print(f"Agent1: {agent1_gripper.mean():.3f}")

        # Detect phases
        phases = detect_phases(agent0_gripper, agent1_gripper)
        phase_counts = np.bincount(phases)

        print("\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")

        return agent0_gripper, agent1_gripper, phases


if __name__ == "__main__":
    # File path
    file_path = "/18141169908/zhangzaibin/multivla-pi0/checkpoints/pass_shoe--only-count-success.h5"

    # First, analyze a sample trajectory to understand the data
    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")

    # Add phases to the entire dataset
    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path,
        close_threshold=0.015,  # Threshold for considering gripper closed
        open_threshold=0.04,  # Threshold for considering gripper open (hysteresis)
        visualize=True,  # Set to True to generate plots for first few trajectories
    )

    print(f"\nDone! Check the output file: {output_path}")
