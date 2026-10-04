# PlaceFactory Phase Analysis Scripts (通用版本)

这些脚本用于分析顺序抓取任务（sequential grasping）的轨迹数据，**自动检测执行顺序**并分配phase标签。

## 🎯 主要特点

1. **自动顺序检测**：根据gripper的release时间点自动判断执行顺序
   - 支持任意顺序：123, 132, 213, 231, 312, 321等
   - Phase标签对应实际执行顺序，而不是固定的agent编号

2. **通用性强**：
   - 命令行参数支持，易于批处理
   - 可调节gripper阈值
   - 灵活的可视化选项

3. **详细统计**：
   - 自动统计所有轨迹的执行顺序分布
   - 生成详细的可视化图表
   - 导出视频展示phase转换

## 📋 Phase定义

**自适应Phase划分**：
- **Phase 1**: 第一个完成的agent（从开始到第一次release）
- **Phase 2**: 第二个完成的agent（第一次release到第二次release）
- **Phase 3**: 第三个完成的agent（第二次release到轨迹结束）

**示例**：
- 如果执行顺序是 `0->1->2` (123):
  - Phase 1 = Agent0抓取放置
  - Phase 2 = Agent1抓取放置
  - Phase 3 = Agent2抓取放置

- 如果执行顺序是 `0->2->1` (132):
  - Phase 1 = Agent0抓取放置
  - Phase 2 = Agent2抓取放置
  - Phase 3 = Agent1抓取放置

## 🚀 使用方法

### 1. 添加Phase标签

#### 基本用法
```bash
cd /18141169908/zhangzaibin/multivla-pi0

# 处理单个文件（默认设置）
python scripts/set_atom/set_phases_to_h5_placefactory.py dataset/placefactory_order_123_20251209.h5

# 指定输出文件路径
python scripts/set_atom/set_phases_to_h5_placefactory.py \
    dataset/placefactory_order_132_20251209.h5 \
    --output dataset/placefactory_132_with_phases.h5

# 调整gripper阈值（默认0.3）
python scripts/set_atom/set_phases_to_h5_placefactory.py \
    dataset/placefactory_order_213_20251209.h5 \
    --close-ratio 0.4

# 跳过可视化（只添加phase标签）
python scripts/set_atom/set_phases_to_h5_placefactory.py \
    dataset/placefactory_order_231_20251209.h5 \
    --no-visualization

# 可视化特定轨迹
python scripts/set_atom/set_phases_to_h5_placefactory.py \
    dataset/placefactory_order_312_20251209.h5 \
    --visualize-trajs 0 5 10 15 20
```

#### 参数说明
- `input_file`: 输入的h5文件路径（必需）
- `--output, -o`: 输出文件路径（可选，默认：`input_file_with_phases.h5`）
- `--close-ratio, -c`: Gripper关闭阈值比例（0-1，默认：0.3）
- `--visualize-trajs, -v`: 要可视化的轨迹索引列表（默认：0 1 5）
- `--no-visualization`: 跳过可视化步骤

#### 输出
- 新的h5文件，包含原始数据和phase标签
- 每个轨迹的phase数据存储在 `traj_X/phase` 数据集中
- 执行顺序存储在 `traj_X/phase` 的 `execution_order` 属性中
- 生成PNG图片：`placefactory_detailed_analysis_traj_X.png`
- 终端输出：执行顺序的统计分布

### 2. 生成可视化视频

```bash
# 可视化特定轨迹
python scripts/set_atom/visualize_phases_placefactory.py \
    dataset/placefactory_order_123_20251209_with_phases.h5 \
    --trajs 0 1 5

# 随机选择3个轨迹可视化
python scripts/set_atom/visualize_phases_placefactory.py \
    dataset/placefactory_order_132_20251209_with_phases.h5 \
    --random 3

# 指定相机视角
python scripts/set_atom/visualize_phases_placefactory.py \
    dataset/placefactory_order_213_20251209_with_phases.h5 \
    --trajs 0 1 2 \
    --cameras head_camera_global head_camera_agent0 head_camera_agent1

# 自定义输出目录
python scripts/set_atom/visualize_phases_placefactory.py \
    dataset/placefactory_order_231_20251209_with_phases.h5 \
    --output-dir my_phase_videos
```

#### 参数说明
- `input_file`: 包含phase信息的h5文件（必需）
- `--trajs, -t`: 要可视化的轨迹索引列表
- `--random, -r`: 随机选择N个轨迹
- `--cameras, -c`: 相机视角列表（默认：head_camera_global head_camera_agent0）
- `--output-dir, -o`: 视频输出目录（默认：phase_videos_placefactory）

#### 输出
- 视频文件保存在指定目录
- 文件名格式：`traj_X_camera_name_phases.mp4`
- 10 FPS，包含phase转换的完整过程

## 📊 批量处理示例

处理多个文件：

```bash
#!/bin/bash
# 批量处理所有placefactory数据集

for order in 123 132 213 231 312 321; do
    echo "Processing order ${order}..."
    python scripts/set_atom/set_phases_to_h5_placefactory.py \
        dataset/placefactory_order_${order}_20251209.h5 \
        --no-visualization
done

echo "All files processed!"
```

## 🔧 算法原理

1. **Gripper状态检测**：
   - 使用统一阈值分析所有agent的gripper位置
   - 应用滞后逻辑（hysteresis）避免噪声
   - 转换为二值状态：1=closed, 0=open

2. **Release检测**：
   - 检测从closed到open的转换点
   - 记录每个agent的第一次release时间

3. **顺序确定**：
   - 根据release时间排序
   - 生成执行顺序字符串（如 "0->1->2"）

4. **Phase分配**：
   - Phase 1: [0, first_release)
   - Phase 2: [first_release, second_release)
   - Phase 3: [second_release, end]

## 📈 输出示例

运行后会看到类似输出：

```
============================================================
Processing /path/to/placefactory_order_123_20251209.h5
Output: /path/to/placefactory_order_123_20251209_with_phases.h5
Close ratio: 0.3
============================================================

Found 100 trajectories

[1/100] Processing traj_0...

  Auto-detecting execution order for 619 steps...
  Agent0 (panda-0) releases at: [150]
  Agent1 (panda-1) releases at: [349]
  Agent2 (panda-2) releases at: [564]

  Detected execution order: 0->1->2

  Phase 1: 0-150 (Agent0 (panda-0) 抓取并放置)
  Phase 2: 150-349 (Agent1 (panda-1) 抓取并放置)
  Phase 3: 349-619 (Agent2 (panda-2) 抓取并放置, release at 564)
  Phase distribution: {1: 150, 2: 199, 3: 270}
  ✓ Added phases: [  0 150 199 270], Order: 0->1->2

...

============================================================
✓ Completed! Output saved to: /path/to/output.h5

📊 Execution Order Statistics:
  0->1->2: 100 trajectories (100.0%)
============================================================
```

## 🐛 故障排查

1. **问题**：所有轨迹都是Phase 1
   - **解决**：调整 `--close-ratio` 参数，尝试 0.2 或 0.4

2. **问题**：检测不到release
   - **解决**：检查gripper数据范围，确认阈值合理

3. **问题**：视频生成失败
   - **解决**：确认安装了 opencv-python: `pip install opencv-python`

## 📝 注意事项

- 脚本会自动创建输出目录
- 输出文件如果已存在会被覆盖
- 可视化图片保存在当前工作目录
- 视频需要 OpenCV 支持

## 🔗 相关文件

- `set_phases_to_h5_placefactory.py`: 主要的phase分析脚本
- `visualize_phases_placefactory.py`: 视频可视化脚本
- `analyze_placefactory.py`: 初步数据分析工具
- `visualize_placefactory_gripper.py`: Gripper状态分析工具


