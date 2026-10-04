import h5py
import numpy as np
import cv2
import os


def create_phase_video(rgb_data, phases, gripper_states, traj_key, camera_name, output_dir="phase_videos_place_burger_fries"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_frames = len(rgb_data)
    height, width = rgb_data[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, f"{traj_key}_{camera_name}_phases.mp4")
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height + 100))

    agent0_gripper, agent1_gripper = gripper_states

    phase_colors = {
        1: (255, 200, 200),
        2: (200, 255, 200),
        3: (200, 200, 255),
        4: (255, 255, 200),
    }

    phase_names = {
        1: "Phase 1: Agent0 Grasp Item",
        2: "Phase 2: Agent0 Place & Return",
        3: "Phase 3: Agent1 Grasp Item",
        4: "Phase 4: Agent1 Place & Return",
    }

    for i in range(n_frames):
        frame = rgb_data[i].copy()

        info_panel = np.ones((100, width, 3), dtype=np.uint8) * 255

        current_phase = phases[i]
        phase_color = phase_colors.get(current_phase, (200, 200, 200))

        overlay = frame.copy()
        overlay[:] = phase_color
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.putText(info_panel, f"Frame: {i}/{n_frames-1}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(info_panel, f"{phase_names.get(current_phase, f'Phase {current_phase}')}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        agent0_grip_color = (0, 150, 0) if agent0_gripper[i] else (150, 0, 0)
        agent1_grip_color = (0, 150, 0) if agent1_gripper[i] else (150, 0, 0)

        cv2.putText(info_panel, f"Agent0 Gripper: {'CLOSED' if agent0_gripper[i] else 'OPEN'}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, agent0_grip_color, 1)
        cv2.putText(info_panel, f"Agent1 Gripper: {'CLOSED' if agent1_gripper[i] else 'OPEN'}",
                    (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, agent1_grip_color, 1)

        phase_progress = (current_phase - 1) / 3 * 200
        cv2.rectangle(info_panel, (10, 90), (210, 98), (200, 200, 200), -1)
        cv2.rectangle(info_panel, (10, 90), (int(10 + phase_progress), 98), (0, 200, 0), -1)
        cv2.putText(info_panel, f"Progress: Phase {current_phase}/4",
                    (220, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        combined_frame = np.vstack([frame, info_panel])

        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")
    return output_path


def visualize_trajectories(file_path, n_trajectories=2, specific_trajs=None):
    with h5py.File(file_path, 'r') as f:
        traj_keys = [key for key in f.keys() if key.startswith('traj_')]

        if specific_trajs:
            selected_trajs = [t for t in specific_trajs if t in traj_keys]
        else:
            selected_trajs = np.random.choice(traj_keys, size=min(n_trajectories, len(traj_keys)), replace=False)

        print(f"Selected trajectories: {selected_trajs}")

        for traj_key in selected_trajs:
            print(f"\n{'='*60}")
            print(f"Processing {traj_key}...")
            print('='*60)

            if f"{traj_key}/phase" not in f:
                print(f"Warning: No phase information found for {traj_key}. Skipping...")
                continue

            phases = f[f"{traj_key}/phase"][:]

            agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
            agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]

            gripper_threshold = 0.035
            agent0_gripper = (agent0_qpos[:, -1] < gripper_threshold).astype(int)
            agent1_gripper = (agent1_qpos[:, -1] < gripper_threshold).astype(int)

            phase_counts = np.bincount(phases)
            print("\nPhase distribution:")
            for i, count in enumerate(phase_counts):
                if count > 0:
                    print(f"  Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")

            print("\nGripper statistics (closed ratio):")
            print(f"  Agent0: {agent0_gripper.mean():.3f}")
            print(f"  Agent1: {agent1_gripper.mean():.3f}")

            camera_names = ['head_camera_global', 'wrist_camera_agent0', 'wrist_camera_agent1']

            for camera_name in camera_names:
                try:
                    rgb_data = f[f"{traj_key}/obs/sensor_data/{camera_name}/rgb"][:]
                    print(f"\nCreating video for {camera_name}...")
                    print(f"  RGB data shape: {rgb_data.shape}")

                    create_phase_video(
                        rgb_data,
                        phases,
                        (agent0_gripper, agent1_gripper),
                        traj_key,
                        camera_name
                    )
                except KeyError as e:
                    print(f"  Camera {camera_name} not found in trajectory: {e}")
                except Exception as e:
                    print(f"  Error creating video for {camera_name}: {e}")


def print_h5_structure(file_path, traj_key='traj_0'):
    print(f"\n{'='*60}")
    print(f"H5 File Structure for {traj_key}")
    print('='*60)

    with h5py.File(file_path, 'r') as f:
        if traj_key in f:
            print(f"\nAvailable groups in {traj_key}:")

            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"  {name}/")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  {name}: {obj.shape} {obj.dtype}")

            f[traj_key].visititems(print_structure)
        else:
            print(f"Trajectory {traj_key} not found!")
            print(f"Available trajectories: {[k for k in f.keys() if k.startswith('traj_')][:5]}...")


if __name__ == "__main__":
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/macg_twoarms/two_arms_place_burger_fries/two_arms_place_burger_fries_with_phases.h5"

    print("=" * 60)
    print("Phase Visualization Tool for Place Burger Fries Task")
    print("=" * 60)

    print_h5_structure(file_path, 'traj_0')

    print("\n\n" + "=" * 60)
    print("Creating Phase Videos")
    print("=" * 60)

    visualize_trajectories(file_path, specific_trajs=['traj_0', 'traj_1'])

    print("\n" + "=" * 60)
    print("Done! Check the phase_videos_place_burger_fries/ directory for output videos.")
    print("=" * 60)
