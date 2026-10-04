cd /18141169908/zhangzaibin/multivla-pi0 && source .venv/bin/activate

python scripts/set_atom/macg/two_arms_place_can_basket/convert_h5_lerobot_place_can_basket.py \
    --h5_path dataset/two_arms_place_can_basket_with_phases.h5 \
    --output_dir dataset/lerobot \
    --task_name two_arms_place_can_basket \
    --use_phase
