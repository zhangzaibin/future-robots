cd /18141169908/zhangzaibin/multivla-pi0 && source .venv/bin/activate

python scripts/set_atom/macg/pushcube/convert_h5_lerobot_pushcube.py \
    --h5_path dataset/macg_twoarms/two_arms_push_cube/two_arms_push_cube_with_phases.h5 \
    --output_dir dataset/lerobot \
    --task_name two_arms_push_cube \
    --use_phase
