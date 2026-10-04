#!/bin/bash
# Example batch processing script for multiple placefactory datasets
# This demonstrates how to use the universal phase detection tool

echo "========================================"
echo "批量处理PlaceFactory数据集"
echo "========================================"

# 数据目录
DATA_DIR="dataset"
SCRIPT="scripts/set_atom/set_phases_to_h5_placefactory.py"
VIS_SCRIPT="scripts/set_atom/visualize_phases_placefactory.py"

# 定义要处理的数据集顺序
ORDERS=(123 132)

# Step 1: 添加phase标签到所有数据集
echo ""
echo "Step 1: 添加Phase标签..."
echo "========================================"

for order in "${ORDERS[@]}"; do
    input_file="${DATA_DIR}/placefactory_order_${order}_20251209.h5"
    
    if [ ! -f "$input_file" ]; then
        echo "⚠ 文件不存在，跳过: $input_file"
        continue
    fi
    
    echo ""
    echo "处理: placefactory_order_${order}..."
    
    python "$SCRIPT" "$input_file" \
        --close-ratio 0.3 \
        --visualize-trajs 0 1 5 \
        --output "${DATA_DIR}/placefactory_order_${order}_20251209_with_phases.h5"
    
    if [ $? -eq 0 ]; then
        echo "✓ 成功: order_${order}"
    else
        echo "✗ 失败: order_${order}"
    fi
done

# Step 2: 为每个数据集生成示例视频
echo ""
echo "Step 2: 生成可视化视频..."
echo "========================================"

for order in "${ORDERS[@]}"; do
    input_file="${DATA_DIR}/placefactory_order_${order}_20251209_with_phases.h5"
    
    if [ ! -f "$input_file" ]; then
        echo "⚠ 文件不存在，跳过: $input_file"
        continue
    fi
    
    echo ""
    echo "可视化: placefactory_order_${order}..."
    
    python "$VIS_SCRIPT" "$input_file" \
        --trajs 0 1 \
        --cameras head_camera_global \
        --output-dir "phase_videos_order_${order}"
    
    if [ $? -eq 0 ]; then
        echo "✓ 视频生成成功: order_${order}"
    else
        echo "✗ 视频生成失败: order_${order}"
    fi
done

echo ""
echo "========================================"
echo "✓ 批量处理完成！"
echo "========================================"
echo ""
echo "输出文件："
echo "  - Phase标签文件: ${DATA_DIR}/*_with_phases.h5"
echo "  - 可视化图片: placefactory_detailed_analysis_traj_*.png"
echo "  - 视频文件: phase_videos_order_*/"
echo ""


