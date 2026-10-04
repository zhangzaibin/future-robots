"""
Phase Detection for pick-and-place-to-plate task.
每个机械臂分别抓取物体放到盘子里。

Phases (for 3 agents operating sequentially):
  Phase 1: Agent0 pick up (approach, grasp, lift)
  Phase 2: Agent0 place  (move to plate, release, return)
  Phase 3: Agent1 pick up
  Phase 4: Agent1 place
  Phase 5: Agent2 pick up
  Phase 6: Agent2 place
  Phase 7: Wait / Done   (all agents idle)

Detection logic:
  - Per-agent gripper thresholds with hysteresis
  - Motion detection based on joint position changes
  - Phase transitions at gripper events and motion → still transitions
  - Automatically detects agent action order by first gripper-close time
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


# ========== Gripper Analysis ==========

def get_per_agent_gripper_thresholds(gripper_pos, close_ratio=0.3):
    """
    Calculate per-agent gripper thresholds based on its own range.
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


def apply_gripper_analysis(gripper_pos, close_threshold, open_threshold):
    """
    Apply gripper thresholds with hysteresis to determine gripper state.
    Returns: Binary array: 1 for closed, 0 for open
    """
    n_steps = len(gripper_pos)
    gripper_state = np.zeros(n_steps, dtype=int)
    gripper_state[0] = 1 if gripper_pos[0] <= close_threshold else 0

    for i in range(1, n_steps):
        prev_state = gripper_state[i - 1]
        curr_pos = gripper_pos[i]
        if prev_state == 0:  # Was open
            gripper_state[i] = 1 if curr_pos <= close_threshold else 0
        else:  # Was closed
            gripper_state[i] = 0 if curr_pos >= open_threshold else 1

    return gripper_state


def find_gripper_events(gripper_state):
    """
    Find all gripper close (0->1) and open (1->0) events.
    Returns: (closes, opens) - lists of indices
    """
    closes = []
    opens = []
    for i in range(1, len(gripper_state)):
        if gripper_state[i - 1] == 0 and gripper_state[i] == 1:
            closes.append(i)
        elif gripper_state[i - 1] == 1 and gripper_state[i] == 0:
            opens.append(i)
    return closes, opens


# ========== Motion Detection ==========

def detect_motion_state(qpos, window_size=5, still_threshold=0.02):
    """
    Detect motion state based on position changes (joints 0-6, excluding gripper).
    Returns: (is_moving binary array, smoothed position change)
    """
    pos_diff = np.abs(np.diff(qpos[:, :7], axis=0))
    pos_change = pos_diff.sum(axis=1)
    pos_change = np.concatenate([[0], pos_change])

    smoothed = np.convolve(pos_change, np.ones(window_size) / window_size, mode="same")
    is_moving = (smoothed > still_threshold).astype(int)

    return is_moving, smoothed


def find_still_start_after(motion_state, start_idx, min_still_duration=10):
    """
    Find the first time point where robot becomes still after start_idx
    and stays still for at least min_still_duration frames.
    """
    n_steps = len(motion_state)
    start_idx = max(0, min(start_idx, n_steps - 1))

    for i in range(start_idx, n_steps - min_still_duration):
        if np.all(motion_state[i : i + min_still_duration] == 0):
            return i

    return n_steps - 1


def find_motion_start_after(motion_state, start_idx, min_motion_duration=3):
    """
    Find the first time point where robot starts moving again after start_idx
    and keeps moving for at least min_motion_duration frames.
    """
    n_steps = len(motion_state)
    start_idx = max(0, min(start_idx, n_steps - 1))

    for i in range(start_idx, n_steps - min_motion_duration):
        if np.all(motion_state[i : i + min_motion_duration] == 1):
            return i

    return n_steps - 1


# ========== Phase Detection ==========

def detect_phases(agents_qpos, close_ratio=0.3, still_threshold=0.02,
                  min_still_duration=5, lift_delay=20, place_delay=20):
    """
    Detect phases for sequential pick-and-place-to-plate task.

    Each agent performs: pick up → place.
    Agents operate sequentially. Order is auto-detected by first gripper-close time.

    Phase layout (for n agents):
      Phase (2*i + 1): Agent_i pick up
      Phase (2*i + 2): Agent_i place
      Phase (2*n + 1): Wait / Done

    Args:
        agents_qpos:       list of qpos arrays [shape (T, n_joints)] for each agent
        close_ratio:       ratio for gripper threshold calculation
        still_threshold:   threshold for motion detection
        min_still_duration: minimum frames of stillness to confirm stop
        lift_delay:        frames to skip after gripper close before searching for still
        place_delay:       frames to skip after gripper open before searching for still

    Returns:
        phases:     array of phase labels (1-based)
        debug_info: dict with intermediate data for visualization
    """
    n_agents = len(agents_qpos)
    n_steps = agents_qpos[0].shape[0]
    phases = np.ones(n_steps, dtype=int)  # Initialize all as phase 1

    print(f"\n  Phase detection for {n_agents} agents, {n_steps} steps...")

    # ---- Per-agent analysis ----
    grippers = []
    motions = []
    pos_changes = []
    all_closes = []
    all_opens = []

    for i, qpos in enumerate(agents_qpos):
        close_th, open_th = get_per_agent_gripper_thresholds(qpos[:, -1], close_ratio)
        gripper = apply_gripper_analysis(qpos[:, -1], close_th, open_th)
        motion, pos_change = detect_motion_state(qpos, still_threshold=still_threshold)
        closes, opens = find_gripper_events(gripper)

        grippers.append(gripper)
        motions.append(motion)
        pos_changes.append(pos_change)
        all_closes.append(closes)
        all_opens.append(opens)

        print(
            f"  Agent{i}: gripper_th=(close={close_th:.4f}, open={open_th:.4f}), "
            f"closes={closes}, opens={opens}"
        )

    # ---- Determine agent action order by first gripper-close time ----
    first_close_times = []
    for i in range(n_agents):
        if all_closes[i]:
            first_close_times.append((all_closes[i][0], i))
        else:
            first_close_times.append((n_steps, i))  # No close → put at end

    first_close_times.sort(key=lambda x: x[0])
    agent_order = [x[1] for x in first_close_times]
    print(f"  Agent action order: {['Agent' + str(a) for a in agent_order]}")

    # ---- Assign phases sequentially ----
    current_pos = 0
    phase_num = 1

    for agent_idx in agent_order:
        closes = all_closes[agent_idx]
        opens = all_opens[agent_idx]
        motion = motions[agent_idx]

        if not closes:
            print(f"  Warning: Agent{agent_idx} has no gripper close event, skipping")
            continue

        # Find first gripper-close event at or after current_pos
        grip_close = None
        for c in closes:
            if c >= current_pos:
                grip_close = c
                break

        if grip_close is None:
            # Agent's close happened before current_pos (possible overlap); use first close
            grip_close = closes[0]
            print(
                f"  Warning: Agent{agent_idx} gripper close at {grip_close} "
                f"is before current_pos {current_pos}"
            )

        # ---------- Pick up phase ----------
        pick_up_end = find_still_start_after(
            motion, grip_close + lift_delay, min_still_duration=min_still_duration
        )

        phases[current_pos:pick_up_end] = phase_num
        print(f"  Phase {phase_num} (Agent{agent_idx} pick up): step {current_pos} → {pick_up_end}")
        phase_num += 1

        # ---------- Place phase ----------
        # Find gripper-open event (release) after pick up
        grip_open = None
        for o in opens:
            if o >= pick_up_end:
                grip_open = o
                break

        if grip_open is not None:
            # Place phase ends at the SECOND stop after gripper opens:
            #   1st stop = release pause (松开物体后短暂停顿)
            #   2nd stop = return to rest (归位)
            first_stop = find_still_start_after(
                motion, grip_open + place_delay, min_still_duration=min_still_duration
            )
            # Find when the arm starts moving again after the first stop (归位动作开始)
            motion_resume = find_motion_start_after(
                motion, first_stop, min_motion_duration=3
            )
            # Find the second stop = arm has returned to rest (归位完成)
            place_end = find_still_start_after(
                motion, motion_resume, min_still_duration=min_still_duration
            )
            print(
                f"  Agent{agent_idx} place detail: "
                f"grip_open={grip_open}, 1st_stop={first_stop}, "
                f"motion_resume={motion_resume}, 2nd_stop(归位)={place_end}"
            )
        else:
            # No explicit open event; use motion stop after pick_up_end
            place_end = find_still_start_after(
                motion, pick_up_end + place_delay, min_still_duration=min_still_duration
            )
            print(f"  Warning: Agent{agent_idx} has no gripper open event, using motion stop")

        phases[pick_up_end:place_end] = phase_num
        print(f"  Phase {phase_num} (Agent{agent_idx} place):   step {pick_up_end} → {place_end}")
        phase_num += 1

        current_pos = place_end

    # ---------- Wait / Done phase ----------
    if current_pos < n_steps:
        phases[current_pos:] = phase_num
        print(f"  Phase {phase_num} (Wait / Done):            step {current_pos} → {n_steps}")

    # ---- Summary ----
    phase_counts = {}
    for p in range(1, phase_num + 1):
        count = int(np.sum(phases == p))
        phase_counts[p] = count
    print(f"  Phase distribution: {phase_counts}")

    # ---- Debug info ----
    debug_info = {"agent_order": agent_order}
    for i in range(n_agents):
        debug_info[f"agent{i}_gripper"] = grippers[i]
        debug_info[f"agent{i}_motion"] = motions[i]
        debug_info[f"agent{i}_pos_change"] = pos_changes[i]

    return phases, debug_info


# ========== H5 Processing ==========

def add_phases_to_h5(file_path, output_path=None,
                     close_ratio=0.3, still_threshold=0.02, visualize=False,
                     agent_names=None):
    """
    Add phase information to HDF5 file based on gripper states and motion detection.

    Args:
        file_path:       path to the source h5 file
        output_path:     path to the output h5 file (default: *_with_phases.h5)
        close_ratio:     ratio for gripper threshold calculation
        still_threshold: threshold for motion detection
        visualize:       whether to generate plots for the first few trajectories
        agent_names:     list of agent sub-paths, e.g. ["panda-0", "panda-1", "panda-2"]
    """
    if agent_names is None:
        agent_names = ["panda-0", "panda-1", "panda-2"]

    if output_path is None:
        output_path = file_path.replace(".h5", "_with_phases.h5")

    with h5py.File(file_path, "r") as src_file:
        with h5py.File(output_path, "w") as dst_file:
            print(f"Processing {file_path} -> {output_path}")

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

                    # Detect phases
                    phases, debug_info = detect_phases(
                        agents_qpos,
                        close_ratio=close_ratio,
                        still_threshold=still_threshold,
                    )

                    # Write phases
                    if f"{traj_key}/phase" in dst_file:
                        del dst_file[f"{traj_key}/phase"]
                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)

                    print(f"  Added phases: {np.bincount(phases)}")

                    # Optional visualization
                    if visualize and int(traj_key.split("_")[1]) < 5:
                        visualize_phases(
                            agents_qpos, debug_info, phases, traj_key, agent_names
                        )

                except Exception as e:
                    print(f"  Error processing {traj_key}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nCompleted! Output saved to: {output_path}")
    return output_path


# ========== Visualization ==========

def visualize_phases(agents_qpos, debug_info, phases, traj_key, agent_names,
                     output_dir="."):
    """
    Create detailed visualization for a single trajectory.
    """
    n_agents = len(agents_qpos)
    n_steps = len(phases)
    time_steps = np.arange(n_steps)

    n_rows = 2 + n_agents  # gripper raw + motion + one row per agent gripper binary + phase
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 4 * n_rows))

    colors = ["b", "r", "g", "m", "c", "y"]

    # ---- Row 0: Raw gripper positions ----
    ax = axes[0]
    for i in range(n_agents):
        ax.plot(
            time_steps,
            agents_qpos[i][:, -1],
            colors[i % len(colors)] + "-",
            linewidth=1,
            label=agent_names[i],
            alpha=0.7,
        )
    ax.set_ylabel("Gripper Position")
    ax.set_title(f"{traj_key} - Pick & Place to Plate Phase Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Row 1: Motion indicator (position change) ----
    ax = axes[1]
    for i in range(n_agents):
        ax.plot(
            time_steps,
            debug_info[f"agent{i}_pos_change"],
            colors[i % len(colors)] + "-",
            linewidth=1,
            label=f"{agent_names[i]} motion",
            alpha=0.7,
        )
    ax.set_ylabel("Position Change")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Rows 2..n_agents+1: Binary gripper states per agent ----
    for i in range(n_agents):
        ax = axes[2 + i]
        ax.plot(
            time_steps,
            debug_info[f"agent{i}_gripper"],
            colors[i % len(colors)] + "-",
            linewidth=2,
            label=f"{agent_names[i]} Gripper State",
        )
        ax.set_ylabel(f"Agent{i} Gripper")
        ax.set_ylim(-0.1, 1.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ---- Last row: Phases ----
    # (replace the last subplot with phase plot if not enough room, add one more)
    # Actually we already allocated n_rows. Let's add one more row for the phase.
    # Re-create figure with n_rows+1 if needed. For simplicity, let's just
    # add the phase plot to an extra axes.

    # Close and redo with extra row for phases
    plt.close(fig)
    n_rows_total = 2 + n_agents + 1
    fig, axes = plt.subplots(n_rows_total, 1, figsize=(18, 4 * n_rows_total))

    # Re-draw all subplots
    ax = axes[0]
    for i in range(n_agents):
        ax.plot(
            time_steps, agents_qpos[i][:, -1],
            colors[i % len(colors)] + "-", linewidth=1,
            label=agent_names[i], alpha=0.7,
        )
    ax.set_ylabel("Gripper Position")
    ax.set_title(f"{traj_key} - Pick & Place to Plate Phase Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(n_agents):
        ax.plot(
            time_steps, debug_info[f"agent{i}_pos_change"],
            colors[i % len(colors)] + "-", linewidth=1,
            label=f"{agent_names[i]} motion", alpha=0.7,
        )
    ax.set_ylabel("Position Change")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i in range(n_agents):
        ax = axes[2 + i]
        ax.plot(
            time_steps, debug_info[f"agent{i}_gripper"],
            colors[i % len(colors)] + "-", linewidth=2,
            label=f"{agent_names[i]} Gripper State",
        )
        ax.set_ylabel(f"Agent{i} Gripper")
        ax.set_ylim(-0.1, 1.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Phase plot
    ax = axes[-1]
    max_phase = int(phases.max())
    phase_colors_list = [
        "lightblue", "lightgreen", "lightyellow", "lightcoral",
        "plum", "lightsalmon", "paleturquoise",
    ]
    agent_order = debug_info.get("agent_order", list(range(n_agents)))
    phase_labels = []
    for idx, a in enumerate(agent_order):
        phase_labels.append(f"Phase {2*idx+1} (Agent{a} pick up)")
        phase_labels.append(f"Phase {2*idx+2} (Agent{a} place)")
    phase_labels.append(f"Phase {max_phase} (Wait/Done)")

    for p in range(1, max_phase + 1):
        mask = phases == p
        if np.any(mask):
            label = phase_labels[p - 1] if p - 1 < len(phase_labels) else f"Phase {p}"
            ax.fill_between(
                time_steps, 0, mask * p, alpha=0.6,
                color=phase_colors_list[(p - 1) % len(phase_colors_list)],
                label=label,
            )

    ax.plot(time_steps, phases, "k-", linewidth=2, marker="o", markersize=1)
    ax.set_ylabel("Phase")
    ax.set_xlabel("Time Step")
    ax.set_ylim(0.5, max_phase + 0.5)
    ax.set_yticks(list(range(1, max_phase + 1)))
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
                              agent_names=None):
    """
    Analyze a single trajectory to understand gripper & motion patterns.
    """
    if agent_names is None:
        agent_names = ["panda-0", "panda-1", "panda-2"]

    with h5py.File(file_path, "r") as f:
        agents_qpos = []
        for name in agent_names:
            qpos = f[f"{traj_key}/obs/agent/{name}/qpos"][:]
            agents_qpos.append(qpos)

        print(f"\n{'='*50}")
        print(f"Analysis of {traj_key}")
        print(f"{'='*50}")
        print(f"Trajectory length: {agents_qpos[0].shape[0]} steps")

        for i, name in enumerate(agent_names):
            qpos = agents_qpos[i]
            gripper_pos = qpos[:, -1]
            print(f"\n--- {name} ---")
            print(f"  qpos shape:     {qpos.shape}")
            print(f"  gripper range:  [{gripper_pos.min():.4f}, {gripper_pos.max():.4f}]")

        # Detect phases
        phases, debug_info = detect_phases(
            agents_qpos,
            close_ratio=close_ratio,
            still_threshold=still_threshold,
        )

        phase_counts = np.bincount(phases)
        print(f"\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"  Phase {i}: {count} steps ({count / len(phases) * 100:.1f}%)")

        return phases, debug_info, agents_qpos


# ========== Main ==========

if __name__ == "__main__":
    # TODO: Change this path to your actual h5 file
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_123.h5"

    # Agent names in the h5 file (adjust if different)
    agent_names = ["panda-0", "panda-1", "panda-2"]

    # Output directory for visualizations
    output_dir = "/18141169908/zhangzaibin/multivla-pi0/phase_videos_pick_place_plate"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Step 1: Analyze sample trajectories ----------
    print("=" * 60)
    print("Step 1: Analyzing sample trajectories")
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
    print("Step 2: Adding phases to all trajectories")
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


