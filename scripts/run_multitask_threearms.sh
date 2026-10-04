uv run scripts/compute_norm_stats.py --config-name aevla_pi05_three_arms_place_objects
XLA_PYTHON_CLIENT_MEM_FRACTION=1 uv run scripts/train.py aevla_pi05_three_arms_place_objects --exp-name aevla_pi05_three_arms_place_objects --overwrite
