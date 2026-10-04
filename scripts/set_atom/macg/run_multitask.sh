#!/bin/bash
set -e

cd /18141169908/zhangzaibin/multivla-pi0
source .venv/bin/activate

# Step 1: Merge datasets (skip if already merged)
if [ ! -d "dataset/MACG/multi_task/multi_task_combined/data" ]; then
    echo "=== Merging multi-task datasets ==="
    python scripts/set_atom/macg/merge_multitask_datasets.py \
        --sources dataset/MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined \
                  dataset/MACG/two_arms_stack_cube/two_arms_stack_cube_combined \
                  dataset/MACG/two_arms_push_cube/two_arms_push_cube_combined \
        --output dataset/MACG/multi_task/multi_task_combined
else
    echo "=== Merged dataset already exists, skipping merge ==="
fi

# Step 2: Compute norm stats and train (pi05 baseline)
echo "=== Computing norm stats for pi05_two_arms_multi_task ==="
uv run scripts/compute_norm_stats.py --config-name pi05_two_arms_multi_task

echo "=== Training pi05_two_arms_multi_task ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py pi05_two_arms_multi_task \
    --exp-name pi05_two_arms_multi_task --overwrite

# Step 3: Compute norm stats and train (AEVLA variant)
echo "=== Computing norm stats for aevla_pi05_two_arms_multi_task ==="
uv run scripts/compute_norm_stats.py --config-name aevla_pi05_two_arms_multi_task

echo "=== Training aevla_pi05_two_arms_multi_task ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py aevla_pi05_two_arms_multi_task \
    --exp-name aevla_pi05_two_arms_multi_task --overwrite
