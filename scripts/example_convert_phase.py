#!/usr/bin/env python3
"""
示例脚本：展示如何使用phase-based转换功能处理合并后的H5文件
"""

from convert_h5_lerobot import convert_robofactory_to_lerobot_with_phase


def main():
    # 自定义的phase指令映射（可选，如果不提供会使用默认的）
    custom_phase_instructions = {
        1: {
            "agent0": "第一阶段：抓取鞋子，使用机械臂0精确抓取鞋子",
            "agent1": "第一阶段：等待观察，机械臂1保持待机状态观察机械臂0的抓取过程",
        },
        2: {
            "agent0": "第二阶段：传递准备，将鞋子移动到交接位置等待机械臂1接收",
            "agent1": "第二阶段：准备接收，机械臂1移动到合适位置准备接收鞋子",
        },
        3: {
            "agent0": "第三阶段：释放交接，机械臂0释放鞋子并移开避免碰撞",
            "agent1": "第三阶段：接收抓取，机械臂1从交接位置抓取鞋子",
        },
        4: {
            "agent0": "第四阶段：任务完成，机械臂0返回初始位置任务结束",
            "agent1": "第四阶段：最终放置，机械臂1将鞋子放置到最终目标位置",
        },
    }

    # 转换合并后的H5文件
    convert_robofactory_to_lerobot_with_phase(
        h5_path="/18141169908/zhangzaibin/dataset/PassShoes/pass_shoes_merged.h5",
        output_dir="/18141169908/zhangzaibin/dataset/PassShoes/lerobot_output",
        task_name="pass_shoes",
        phase_instructions=custom_phase_instructions,  # 使用自定义指令
        push_to_hub=False,
    )


if __name__ == "__main__":
    main()
