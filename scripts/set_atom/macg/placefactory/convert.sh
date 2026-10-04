cd /18141169908/zhangzaibin/multivla-pi0 && source .venv/bin/activate

python scripts/set_atom/macg/placefactory/convert_h5_lerobot_placefactory_macg.py \
    --h5_path dataset/three_arms/three_arms_merge.h5 \
    --output_dir dataset/lerobot \
    --task_name three_arms_place_objects \
    --use_phase