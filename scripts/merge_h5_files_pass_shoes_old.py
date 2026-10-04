#!/usr/bin/env python3

import h5py
import numpy as np


def merge_h5_files():
    """
    合并四个H5文件并为每个轨迹添加phase字段
    """
    # 定义输入文件和对应的phase
    file_phase_mapping = [
        ("pass_phase1_grasp_a0.h5", 1),
        ("pass_phase2_place_a0.h5", 2),
        ("pass_phase3_grasp_a1.h5", 3),
        ("pass_phase4_place_a1.h5", 4),
    ]

    output_file = "pass_shoes_merged.h5"

    print("开始合并H5文件...")

    # 创建输出文件
    with h5py.File(output_file, "w") as output_f:
        trajectory_counter = 0

        for file_name, phase in file_phase_mapping:
            print(f"处理文件: {file_name}, Phase: {phase}")

            with h5py.File(file_name, "r") as input_f:
                # 获取所有轨迹键
                traj_keys = [k for k in input_f.keys() if k.startswith("traj_")]
                print(f"  找到 {len(traj_keys)} 个轨迹")

                for traj_key in traj_keys:
                    # 创建新的轨迹名称
                    new_traj_key = f"traj_{trajectory_counter}"

                    # 复制整个轨迹组
                    input_f.copy(traj_key, output_f, new_traj_key)

                    # 添加phase字段到新轨迹
                    traj_group = output_f[new_traj_key]

                    # 获取轨迹长度（从任意一个数据集获取）
                    traj_length = traj_group["terminated"].shape[0]

                    # 创建phase数据集，所有时间步都是相同的phase值
                    phase_data = np.full((traj_length,), phase, dtype=np.int32)
                    traj_group.create_dataset("phase", data=phase_data)

                    trajectory_counter += 1

                    if trajectory_counter % 50 == 0:
                        print(f"  已处理 {trajectory_counter} 个轨迹")

    print(f"合并完成！总共处理了 {trajectory_counter} 个轨迹")
    print(f"输出文件: {output_file}")

    # 验证输出文件
    print("\n验证输出文件结构...")
    with h5py.File(output_file, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"输出文件包含 {len(traj_keys)} 个轨迹")

        # 检查第一个轨迹的结构
        if traj_keys:
            first_traj = f[traj_keys[0]]
            print(f"第一个轨迹 ({traj_keys[0]}) 的字段:")
            for key in first_traj.keys():
                if hasattr(first_traj[key], "shape"):
                    print(f"  {key}: {first_traj[key].shape}")
                else:
                    print(f"  {key}: group")

        # 检查phase分布
        phase_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for traj_key in traj_keys:
            phase_val = f[traj_key]["phase"][0]  # 取第一个时间步的phase值
            phase_counts[phase_val] += 1

        print("\nPhase分布:")
        for phase, count in phase_counts.items():
            print(f"  Phase {phase}: {count} 个轨迹")


if __name__ == "__main__":
    merge_h5_files()
