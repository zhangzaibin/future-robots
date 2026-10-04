#!/bin/bash

# 训练第一个配置：place_meat_colla
echo "=========================================="
echo "开始训练 place_meat_colla"
echo "=========================================="
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_droid_place_meat_colla --exp_name place_meat_colla_exp --save_interval 5000

# 检查退出码，如果不是正常退出（比如被手动停止），则退出脚本
if [ $? -ne 0 ]; then
    echo "place_meat_colla 训练失败或被中断，退出脚本"
    exit 1
fi

echo "=========================================="
echo "place_meat_colla 训练完成，开始训练 place_meat_order"
echo "=========================================="

# 训练第二个配置：place_meat_order
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_droid_place_meat_order --exp_name place_meat_order_exp --save_interval 5000

echo "=========================================="
echo "所有训练任务完成"
echo "=========================================="
