import h5py
import numpy as np
import matplotlib.pyplot as plt


def analyze_gripper_state(qpos, gripper_threshold=0.02):
    """
    Analyze gripper state from qpos data.
    Returns binary array: 1 for closed/gripping, 0 for open
    """
    gripper_pos = qpos[:, -1]  # Last joint is usually gripper
    gripper_closed = gripper_pos < gripper_threshold
    return gripper_closed.astype(int)


# 函数的输入是两个列表。表示agent0与agent1的夹爪状态,0表示张开，1表示闭合.
def detect_phases(agent0_gripper, agent0_joint_state ,agent1_joint_state, agent2_joint_state):
    """
    Detect phases based on gripper states:
    Phase 1: Agent0 grasp & lift the blue cube ; Agent1 grasp & lift the green cube (ends when all robots are lifted and stationary)
    Phase 2: Agent0 place the blue cube on the target position ; Agent1 Stop
    Phase 3: Agent0 stop ; Agent1 place the green cube on the blue cube
    """
    n_steps = len(agent0_gripper)
    phases = np.zeros(n_steps, dtype=int)
    
    # Find events
    agent0_grip_start = None    # agent0夹爪由开到闭合
    agent2_stop = None          # agent2停止运动
    agent0_stop = None          # agent0停止运动
    agent1_stop = None          # agent1停止运动
    
    # Detect agent0 grip start (transition from 0 to 1)，获得agent0夹爪由开到闭合的step
    for i in range(1, n_steps):
        if agent0_gripper[i-1] == 0 and agent0_gripper[i] == 1:
            agent0_grip_start = i
            break
    
    # # Detect agent0 grip end (transition from 1 to 0),获得agent0夹爪由闭合到打开的step
    # if agent0_grip_start is not None:
    #     for i in range(agent0_grip_start + 1, n_steps):
    #         if agent0_gripper[i-1] == 1 and agent0_gripper[i] == 0:
    #             agent0_grip_end = i
    #             break
    
    # Detect agent1 stop (find first index where joint state stays stable)
    if agent0_grip_start is not None:
        window_size = 50 
        threshold = 0.0005  # 相邻元素差值阈值
        
        for i in range(agent0_grip_start, len(agent2_joint_state) - window_size + 1):
            # 检查连续window_size个元素
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent2_joint_state[j+1] - agent2_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent2_stop = i
                break
            
    # Detect agent0 stop (find first index where joint state stays stable)
    if agent2_stop is not None:
        window_size = 50 
        threshold = 0.0005  # 相邻元素差值阈值
        
        for i in range(agent2_stop, len(agent0_joint_state) - window_size + 1):
            # 检查连续window_size个元素
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent0_joint_state[j+1] - agent0_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent0_stop = i
                break
    # Detect agent1 stop (find first index where joint state stays stable)
    if agent0_stop is not None:
        window_size = 50 
        threshold = 0.0005  # 相邻元素差值阈值
        
        for i in range(agent0_stop, len(agent1_joint_state) - window_size + 1):
            # 检查连续window_size个元素
            is_stable = True
            for j in range(i, i + window_size - 1):
                if abs(agent1_joint_state[j+1] - agent1_joint_state[j]) > threshold:
                    is_stable = False
                    break
            if is_stable:
                agent1_stop = i
                break
                
    
    
    
    # Assign phases (grasp + lift merged into Phase 1)
    if agent2_stop is not None:
        phases[:agent2_stop] = 1  # Phase 1: Grasp & Lift (until all robots lifted and stationary)
        
        if agent0_stop is not None:
            phases[agent2_stop:agent0_stop] = 2  # Phase 2: Agent0 place blue cube; Agent1 Stop
            
            if agent1_stop is not None:
                phases[agent0_stop:agent1_stop] = 3  # Phase 3: Agent0 stop; Agent1 place green cube on blue
                phases[agent1_stop:] = 4  # Done
            else:
                print("Error!!! agent1_stop not detected")
                phases[agent0_stop:] = 3
        else:
            print("Error!!! agent0_stop not detected")
            phases[agent2_stop:] = 2
    else:
        print("Error!!! agent2_stop not detected")
        phases[:] = 1
    
    return phases

def add_phases_to_h5(file_path, output_path=None, gripper_threshold=0.02, visualize=False):
    """
    Add phase information to HDF5 file based on gripper states.
    """
    if output_path is None:
        output_path = file_path.replace('.h5', '_with_phases.h5')
    
    # Read the original file
    with h5py.File(file_path, 'r') as src_file:
        # Create new file
        with h5py.File(output_path, 'w') as dst_file:
            print(f"Processing {file_path} -> {output_path}")
            
            # Copy all data and add phases
            traj_keys = [key for key in src_file.keys() if key.startswith('traj_')]
            print(f"Found {len(traj_keys)} trajectories")
            
            for traj_key in traj_keys:
                print(f"Processing {traj_key}...")
                
                # Copy the entire trajectory group
                src_file.copy(traj_key, dst_file)
                
                # Get gripper data
                try:
                    agent0_qpos = src_file[f"{traj_key}/obs/agent/panda-0/qpos"][:]
                    agent1_qpos = src_file[f"{traj_key}/obs/agent/panda-1/qpos"][:]
                    agent2_qpos = src_file[f"{traj_key}/obs/agent/panda-2/qpos"][:]
                    
                    # Analyze gripper states
                    agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
                    agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)
                    
                    
                    # get a joint state for judge agent 0 stop
                    agent0_joint_state = agent0_qpos[:, 1]
                    agent1_joint_state = agent1_qpos[:, 1]
                    agent2_joint_state = agent2_qpos[:, 1]
                    # Detect phases
                    phases = detect_phases(agent0_gripper, agent0_joint_state, agent1_joint_state, agent2_joint_state)

                    # Add phases to the destination file
                    dst_file.create_dataset(f"{traj_key}/phase", data=phases)
                    
                    print(f"  Added phases: {np.bincount(phases)}")
                    
                    # Visualization for first few trajectories
                    if visualize and int(traj_key.split('_')[1]) < 5:
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
    plt.plot(time_steps, agent0_gripper, 'b-', linewidth=2, label='Agent0 Gripper')
    plt.ylabel('Agent0 Gripper')
    plt.title(f'{traj_key} - Gripper States and Phases')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, agent1_gripper, 'r-', linewidth=2, label='Agent1 Gripper')
    plt.ylabel('Agent1 Gripper')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, phases, 'g-', linewidth=2, marker='o', markersize=3)
    plt.ylabel('Phase')
    plt.xlabel('Time Step')
    plt.ylim(0.5, 4.5)
    plt.yticks([1, 2, 3, 4], ['1: Grasp&Lift', '2: Place Blue', '3: Place Green', '4: Done'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'phase_analysis_{traj_key}.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_single_trajectory(file_path, traj_key, gripper_threshold=0.02):
    """
    Analyze a single trajectory to understand gripper patterns
    """
    with h5py.File(file_path, 'r') as f:
        agent0_qpos = f[f"{traj_key}/obs/agent/panda-0/qpos"][:]
        agent1_qpos = f[f"{traj_key}/obs/agent/panda-1/qpos"][:]
        agent2_qpos = f[f"{traj_key}/obs/agent/panda-2/qpos"][:]
        
        print(f"\n=== Analysis of {traj_key} ===")
        print(f"Trajectory length: {len(agent0_qpos)} steps")
        print(f"Agent0 qpos shape: {agent0_qpos.shape}")
        print(f"Agent1 qpos shape: {agent1_qpos.shape}")
        
        # Analyze gripper positions
        agent0_gripper_pos = agent0_qpos[:, -1]
        agent1_gripper_pos = agent1_qpos[:, -1]
        
        print(f"\nGripper position ranges:")
        print(f"Agent0 gripper: [{agent0_gripper_pos.min():.4f}, {agent0_gripper_pos.max():.4f}]")
        print(f"Agent1 gripper: [{agent1_gripper_pos.min():.4f}, {agent1_gripper_pos.max():.4f}]")
        
        # Analyze gripper states
        agent0_gripper = analyze_gripper_state(agent0_qpos, gripper_threshold)
        agent1_gripper = analyze_gripper_state(agent1_qpos, gripper_threshold)
        
        print(f"\nGripper states (closed ratio):")
        print(f"Agent0: {agent0_gripper.mean():.3f}")
        print(f"Agent1: {agent1_gripper.mean():.3f}")
        
       
        
        
        # get a joint state for judge agent 0 stop
        agent0_joint_state = agent0_qpos[:, 1]
        agent1_joint_state = agent1_qpos[:, 1]
        agent2_joint_state = agent2_qpos[:, 1]
        # Detect phases
        phases = detect_phases(agent0_gripper, agent0_joint_state, agent1_joint_state, agent2_joint_state)
        phase_counts = np.bincount(phases)
        
        
        print(f"\nPhase distribution:")
        for i, count in enumerate(phase_counts):
            if count > 0:
                print(f"Phase {i}: {count} steps ({count/len(phases)*100:.1f}%)")
        
        return agent0_gripper, agent1_gripper, phases

if __name__ == "__main__":
    # File path
    file_path = "/18141169908/zhangzaibin/multivla-pi0/dataset/stackcubes.h5"
    
    # First, analyze a sample trajectory to understand the data
    print("=== Analyzing sample trajectories ===")
    analyze_single_trajectory(file_path, "traj_0")
    analyze_single_trajectory(file_path, "traj_1")
    
    # Add phases to the entire dataset
    print("\n=== Adding phases to all trajectories ===")
    output_path = add_phases_to_h5(
        file_path, 
        gripper_threshold=0.02,  # Adjust threshold as needed
        visualize=True  # Set to True to generate plots for first few trajectories
    )
    
    print(f"\nDone! Check the output file: {output_path}")