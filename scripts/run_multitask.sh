uv run scripts/compute_norm_stats.py --config-name aevla_pi05_two_arms_multi_task
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py aevla_pi05_two_arms_multi_task --exp-name aevla_pi05_two_arms_multi_task --overwrite

# #!/usr/bin/env bash
# set -euo pipefail

# UV_BIN="/root/miniconda3/bin/uv"

# cd /18141169908/zhangzaibin/multivla-pi0
# source .venv/bin/activate

# "$UV_BIN" run scripts/compute_norm_stats.py --config-name aevla_pi05_two_arms_multi_task

# XLA_PYTHON_CLIENT_MEM_FRACTION=1 \
# "$UV_BIN" run scripts/train.py aevla_pi05_two_arms_multi_task \
#   --exp-name aevla_pi05_two_arms_multi_task_0411 \
#   --overwrite
