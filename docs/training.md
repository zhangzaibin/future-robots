# Training recipes

[Documentation](README.md) · [Prepare data](data.md) · [Real-world training](real_world.md) · [Deployment](deployment.md)

Run the recipes after [installation](installation.md). Each wrapper resolves the repository root, computes statistics with training augmentations disabled, then calls `scripts/train.py`. Exact model and optimization settings remain in [`config.py`](../src/openpi/training/config.py).

## MA-VLA

The canonical entry is [`scripts/macg_new/train_mavla.sh`](../scripts/macg_new/train_mavla.sh):

```bash
EXP_NAME=mavla_run1 bash scripts/macg_new/train_mavla.sh
```

It selects `pi05_two_arms_multi_task_mavla_shuffle06_mask04`: a standard Pi0.5 backbone, phase-derived atomic instructions, arm shuffle probability **0.6**, image masking probability **0.4**, and delta joint actions. `mask_random_one=True` selects one image to mask when masking is applied. Wait augmentation is disabled. The release restores those augmentation values to match the recipe name.

This is the current two-arm, multi-task Pi0.5 implementation of the MA-VLA recipe. The paper also covers other settings; do not treat this single command as reproduction of every paper table.

The trained weights are available as [MrBean2024/MA-VLA-pi05-shuffle06-mask04](https://huggingface.co/MrBean2024/MA-VLA-pi05-shuffle06-mask04). To run them directly, follow [released MA-VLA checkpoint deployment](deployment.md#released-ma-vla-checkpoint).

## AE-VLA

The [AE-VLA paper](https://arxiv.org/abs/2610.06184) combines arm-token grouping, SkillLoRA, and arm-wise attention. The **full model** uses `Pi0SkillLoRAComposerConfig`; this is also the architecture of the [public checkpoint](https://huggingface.co/MrBean2024/AE-VLA-full):

```bash
EXP_NAME=aevla_full_run1 bash scripts/macg_new/train_skill_lora.sh
```

The [public training dataset](data.md#public-training-dataset) includes `meta/skill_labels.jsonl` for router supervision. The attention-only variant uses `Pi0MultiAgentAttnConfig`, without SkillLoRA:

```bash
EXP_NAME=aevla_run1 bash scripts/macg_new/train_aevla.sh
# Equivalent original entry:
# bash scripts/macg_new/train/train_aevla_pi05_two_arms_multi_task.sh
```

| Variant | Configuration | Entry |
| --- | --- | --- |
| **Full AE-VLA: SkillLoRA + arm-wise attention** | `aevla_pi05_skill_lora_two_arms_multitask` | [`train_skill_lora.sh`](../scripts/macg_new/train_skill_lora.sh) |
| Arm-wise attention, without SkillLoRA | `aevla_pi05_two_arms_multi_task` | [`train_aevla.sh`](../scripts/macg_new/train_aevla.sh) |
| Full attention, per-agent token groups | `aevla_pi05_two_arms_multi_task_full_attn` | [`train_aevla_pi05_two_arms_multi_task_full_attn.sh`](../scripts/macg_new/train/train_aevla_pi05_two_arms_multi_task_full_attn.sh) |
| Skill-LoRA with full attention | `aevla_pi05_skill_lora_two_arms_multitask_full_attn` | [`train_fullattn.sh`](../scripts/macg_new/train_fullattn.sh) |

The older `train_tokengrouponly.sh` name is an alias for the full-attention, per-agent-token configuration. When preparing your own data, include `meta/skill_labels.jsonl` for SkillLoRA and follow the [Skill-LoRA notes](skill_lora.md). The attention-only recipe does not require that file. See [released checkpoint deployment](deployment.md#released-ae-vla-checkpoint) to run the full model without training.

## Separate Pi05

Train one policy for each physical arm using the same combined dataset:

```bash
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_left_multi_task.sh
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_right_multi_task.sh
```

| Policy | Configuration | Arm selection | Normalization asset ID |
| --- | --- | --- | --- |
| Left | `pi05_single_arm_left_multi_task` | `single_arm_id=0` | `multi_task_combined_left` |
| Right | `pi05_single_arm_right_multi_task` | `single_arm_id=1` | `multi_task_combined_right` |

The transforms select the relevant state, actions, camera, and instruction. Shuffling, masking, and wait augmentation are disabled. Each arm has its own statistics; neither recipe overwrites the shared two-arm statistics.

Both recipes default to two FSDP devices. To run independent jobs on individual GPUs, when memory permits:

```bash
CUDA_VISIBLE_DEVICES=0 FSDP_DEVICES=1 BATCH_SIZE=8 \
  bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_left_multi_task.sh
# Run in another terminal:
CUDA_VISIBLE_DEVICES=1 FSDP_DEVICES=1 BATCH_SIZE=8 \
  bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_right_multi_task.sh
```

These batch-size changes affect optimization; record them when comparing methods. Serve the resulting models with the [separate-policy deployment workflow](deployment.md#separate-policies).

## Unified Pi05

```bash
bash scripts/macg_new/pi05/train.sh
```

This selects `pi05_two_arms_multi_task`, with a shared standard Pi0.5 model, delta joint actions, and no shuffle, image masking, or wait augmentation. It is the unaugmented unified baseline.

## Common options

```bash
# Preview; no uv, data, weights, or GPU needed.
bash scripts/macg_new/train_mavla.sh --dry-run

# Use an external dataset and a new experiment name.
DATASET_DIR=/data/my_combined EXP_NAME=mavla_seed42 \
  bash scripts/macg_new/train_mavla.sh --seed 42

# Statistics only, or training with statistics already prepared.
bash scripts/macg_new/train_aevla.sh --norm-only
bash scripts/macg_new/train_aevla.sh --skip-norm

# Restore model, optimizer, and step from the same experiment.
EXP_NAME=mavla_seed42 NUM_TRAIN_STEPS=60000 \
  bash scripts/macg_new/train_mavla.sh --resume

# Explicitly discard and replace an existing experiment.
EXP_NAME=throwaway bash scripts/macg_new/train_aevla.sh --overwrite
```

`--resume` skips statistics computation by default, preserving the original normalization. `NUM_TRAIN_STEPS` is the cumulative training target, not additional steps. For DAgger with changed data, use the [dedicated preparation workflow](real_world.md#dagger-and-resuming).

`BATCH_SIZE`, `NUM_TRAIN_STEPS`, `FSDP_DEVICES`, and `EXP_NAME` override recipe defaults; additional arguments go directly to `train.py`. Set a single GPU with `CUDA_VISIBLE_DEVICES=0 FSDP_DEVICES=1`. Normalization uses the config's own batch size and all available frames (small datasets use a smaller statistics batch).

The main MA-VLA, AE-VLA attention, unified Pi0.5, and separate Pi0.5 configs use 30,000 steps, global batch 32, two FSDP devices, and a 30,000-step save interval. Checkpoints are stored under `<FUTURE_CHECKPOINT_ROOT>/<config>/<experiment>/<step>/`. Step directory names follow the trainer's numbering; inspect the saved directory instead of assuming a specific final number.

Changing dataset paths is supported by the common environment variables. For consistent statistics, prefer them over passing a training-only `--data.repo-id` override. Never combine `--resume` with `--overwrite`.
