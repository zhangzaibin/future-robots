#!/bin/bash
export HF_LEROBOT_HOME=/18141169908/zhangzaibin/multivla-pi0/dataset/placefactory
# python scripts/set_atom/macg/placefactory/convert_h5_lerobot_placefactory.py --h5_path /18141169908/zhangzaibin/multivla-pi0/dataset/placefactory_new/merge.h5 --output_dir /18141169908/zhangzaibin/multivla-pi0/dataset/placefactory --task_name placefactory --use_phase
uv run scripts/compute_norm_stats.py --config-name pi0_base_2arms_placebowls_macg
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py pi0_base_2arms_placebowls_macg --exp_name 10000_h50_doubleagent_placebowls_macg --overwrite
