```bash
# Phase 模式（使用分阶段的原子指令）
python scripts/set_atom/macg/handover_bottles/convert_h5_lerobot_handover_bottles.py \
  --h5_path dataset/macg_twoarms/two_arms_handover_bottles/two_arms_handover_bottles_with_phases.h5 \
  --output_dir dataset/lerobot \
  --task_name two_arms_handover_bottles \
  --use_phase

# Regular 模式（整条轨迹用同一条指令）
python scripts/set_atom/macg/handover_bottles/convert_h5_lerobot_handover_bottles.py \
  --h5_path dataset/MACG/two_arms_handover_bottles/two_arms_handover_bottles.h5 \
  --output_dir dataset/lerobot \
  --task_name two_arms_handover_bottles \
  --task_instruction_agent0 "Pick up the bottle and hand it over." \
  --task_instruction_agent1 "Receive the bottle and place it."

python scripts/set_atom/macg/two_arms_stack_bowls/convert_h5_lerobot_placebowls.py \
  --h5_path dataset/macg_twoarms/two_arms_stack_bowl/two_arms_stack_bowl_with_phases.h5 \
  --output_dir dataset/lerobot \
  --task_name two_arms_stack_bowl \
  --use_phase

```