# Phase-based H5 to LeRobot Conversion

这个文档解释如何使用更新后的`convert_h5_lerobot.py`脚本来处理带有phase字段的合并H5文件。

## 功能概述

新增的`convert_robofactory_to_lerobot_with_phase`函数可以：

1. **读取合并后的H5文件**：处理包含多个phase的合并数据
2. **动态设置instruction**：根据每个时间步的phase值自动设置对应的任务指令
3. **保留phase信息**：在输出的LeRobot数据集中保留phase字段
4. **支持自定义指令**：允许用户自定义每个phase的指令内容

## 使用方法

### 1. 命令行使用

```bash
# 使用默认的phase指令映射
python convert_h5_lerobot.py \
    --h5_path /path/to/merged_file.h5 \
    --output_dir /path/to/output \
    --task_name pass_shoes \
    --use_phase

# 不使用phase功能（原来的方式）
python convert_h5_lerobot.py \
    --h5_path /path/to/original_file.h5 \
    --output_dir /path/to/output \
    --task_name stack_cube \
    --task_instruction_agent0 "Stack the cube with agent 0" \
    --task_instruction_agent1 "Assist agent 0 in stacking"
```

### 2. 程序内调用

```python
from convert_h5_lerobot import convert_robofactory_to_lerobot_with_phase

# 使用默认指令
convert_robofactory_to_lerobot_with_phase(
    h5_path="pass_shoes_merged.h5",
    output_dir="./lerobot_output",
    task_name="pass_shoes"
)

# 使用自定义指令
custom_instructions = {
    1: {
        "agent0": "Phase 1: Grasp the object",
        "agent1": "Phase 1: Wait and observe"
    },
    2: {
        "agent0": "Phase 2: Move to handover position", 
        "agent1": "Phase 2: Prepare to receive"
    },
    # ... 更多phase
}

convert_robofactory_to_lerobot_with_phase(
    h5_path="pass_shoes_merged.h5",
    output_dir="./lerobot_output", 
    task_name="pass_shoes",
    phase_instructions=custom_instructions
)
```

## 默认Phase指令映射

如果不提供自定义指令，系统使用以下默认映射：

```python
{
    1: {
        "agent0": "Grasp the shoes with the first robot arm",
        "agent1": "Wait and observe while the first robot grasps the shoes"
    },
    2: {
        "agent0": "Place the shoes in the handover position for the second robot",
        "agent1": "Prepare to receive the shoes from the first robot"
    },
    3: {
        "agent0": "Release and move away after successful handover",
        "agent1": "Grasp the shoes from the handover position"
    },
    4: {
        "agent0": "Task completed, return to home position",
        "agent1": "Place the shoes in the final target position"
    }
}
```

## 输出数据格式

转换后的LeRobot数据集包含以下字段：

- `image_0`: 机械臂0的摄像头图像 (224, 224, 3)
- `image_1`: 机械臂1的摄像头图像 (224, 224, 3)  
- `global_image`: 全局摄像头图像 (224, 224, 3)
- `joint_position`: 关节位置 (7,)
- `gripper_position`: 夹爪位置 (1,)
- `actions`: 完整动作 (8,)
- `phase`: 当前phase值 (1,) - **新增字段**
- `task`: 根据phase动态生成的任务指令

## 示例：处理PassShoes数据集

```python
# 示例：转换PassShoes合并数据
convert_robofactory_to_lerobot_with_phase(
    h5_path="/18141169908/zhangzaibin/dataset/PassShoes/pass_shoes_merged.h5",
    output_dir="/18141169908/zhangzaibin/dataset/PassShoes/lerobot_output",
    task_name="pass_shoes",
    push_to_hub=False
)
```

这将生成两个数据集：
- `pass_shoes_agent0/`: 包含机械臂0的数据和对应的phase指令
- `pass_shoes_agent1/`: 包含机械臂1的数据和对应的phase指令

## 注意事项

1. **Phase字段要求**：输入的H5文件必须包含`phase`字段
2. **指令一致性**：每个phase中的所有时间步都使用相同的指令
3. **Agent命名**：自定义指令字典中使用`agent0`和`agent1`作为键名
4. **错误处理**：如果某个phase没有对应的指令，会使用默认格式：`"Phase {phase_num} task for agent {agent_id}"` 