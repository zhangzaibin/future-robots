#!/usr/bin/env python3

import h5py
import numpy as np


def merge_h5_files():
    """
    合并四个H5文件，将同一索引的四个phase轨迹按时间维连接成一个总轨迹，并为每个时间步添加phase字段
    只保留总的traj_i（每个任务一条），其余行为尽量与原来保持一致
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

    # 打开所有输入文件
    input_files = [h5py.File(fname, "r") for fname, _ in file_phase_mapping]
    try:
        # 收集每个文件中的轨迹键，按索引排序
        traj_keys_per_file = []
        for f in input_files:
            keys = [k for k in f.keys() if k.startswith("traj_")]
            keys_sorted = sorted(keys, key=lambda k: int(k.split("_")[1]))
            traj_keys_per_file.append(keys_sorted)
        counts = [len(k) for k in traj_keys_per_file]

        if len(set(counts)) != 1:
            print(f"警告：各文件轨迹数量不一致 {counts}，将按最小数量对齐进行合并。")
        num_trajs = min(counts) if counts else 0
        print(f"每个文件中可用于对齐的轨迹数: {num_trajs}")

        with h5py.File(output_file, "w") as output_f:
            trajectory_counter = 0

            def merge_groups(output_group: h5py.Group, input_groups: list):
                # 找到所有输入组的共同键，确保结构一致性
                common_keys = set(input_groups[0].keys())
                for g in input_groups[1:]:
                    common_keys &= set(g.keys())

                for key in sorted(common_keys):
                    # 跳过我们将单独创建的 'phase'
                    if key == "phase":
                        continue

                    items = [g[key] for g in input_groups]

                    if isinstance(items[0], h5py.Dataset):
                        # 读取四段数据
                        arrays = [ds[...] for ds in items]
                        # 判断是否可按时间维（axis=0）拼接
                        can_concat = True
                        if arrays[0].ndim == 0:
                            can_concat = False
                        else:
                            base_rest_shape = arrays[0].shape[1:]
                            for arr in arrays:
                                if arr.ndim == 0 or arr.shape[1:] != base_rest_shape:
                                    can_concat = False
                                    break
                        if can_concat:
                            data = np.concatenate(arrays, axis=0)
                            output_group.create_dataset(key, data=data)
                        else:
                            # 非时间序列数据，保留第一段
                            output_group.create_dataset(key, data=arrays[0])
                    elif isinstance(items[0], h5py.Group):
                        sub_out = output_group.create_group(key)
                        merge_groups(sub_out, items)
                    else:
                        # 未知类型，跳过
                        continue

            for traj_idx in range(num_trajs):
                new_traj_key = f"traj_{traj_idx}"
                print(f"处理对齐轨迹: {new_traj_key} 来自四个phase (索引 {traj_idx})")

                # 获取四个phase中的对应轨迹组
                in_groups = [input_files[i][traj_keys_per_file[i][traj_idx]] for i in range(len(input_files))]

                # 计算每一段的长度（基于 'terminated' 数据集）
                segment_lengths = []
                for gi, g in enumerate(in_groups):
                    if "terminated" not in g:
                        raise KeyError(f"输入轨迹 {traj_idx} 缺少 'terminated' 数据集（文件索引 {gi}）")
                    segment_lengths.append(g["terminated"].shape[0])

                # 创建输出轨迹组
                out_group = output_f.create_group(new_traj_key)

                # 合并除 'phase' 外的所有键（递归处理）
                merge_groups(out_group, in_groups)

                # 创建按时间步展开的 phase 数据集
                phase_segments = [
                    np.full(seg_len, file_phase_mapping[i][1], dtype=np.int32)
                    for i, seg_len in enumerate(segment_lengths)
                ]
                phase_series = (
                    np.concatenate(phase_segments, axis=0) if phase_segments else np.array([], dtype=np.int32)
                )
                out_group.create_dataset("phase", data=phase_series)

                trajectory_counter += 1
                if trajectory_counter % 50 == 0:
                    print(f"  已处理 {trajectory_counter} 个轨迹")

        print(f"合并完成！总共处理了 {num_trajs} 个轨迹")
        print(f"输出文件: {output_file}")

    finally:
        for f in input_files:
            try:
                f.close()
            except Exception:
                pass

    # 验证输出文件
    print("\n验证输出文件结构...")
    with h5py.File(output_file, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        print(f"输出文件包含 {len(traj_keys)} 个轨迹")

        # 检查第一个轨迹的结构（仅检查顶层）
        if traj_keys:
            first_traj = f[traj_keys[0]]
            print(f"第一个轨迹 ({traj_keys[0]}) 的字段:")
            for key in first_traj.keys():
                if hasattr(first_traj[key], "shape"):
                    print(f"  {key}: {first_traj[key].shape}")
                else:
                    print(f"  {key}: group")

        # 检查phase分布（按时间步计数）
        phase_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for traj_key in traj_keys:
            phase_vals = f[traj_key]["phase"][...]
            unique_vals, counts = np.unique(phase_vals, return_counts=True)
            for val, cnt in zip(unique_vals.tolist(), counts.tolist()):
                if val in phase_counts:
                    phase_counts[val] += int(cnt)
        print("\nPhase分布(按时间步):")
        for phase, count in phase_counts.items():
            print(f"  Phase {phase}: {count} 个时间步")


if __name__ == "__main__":
    merge_h5_files()
