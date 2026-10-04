uv run scripts/compute_norm_stats.py --config-name pi05_ur_two_arms
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py pi05_ur_two_arms --exp-name pi05_ur_two_arms --overwrite