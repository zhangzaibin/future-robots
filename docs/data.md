# Data preparation

[Documentation](README.md) · [Training](training.md) · [Real-world data](real_world.md)

## Public training dataset

Download [**MrBean2024/macg_data**](https://huggingface.co/datasets/MrBean2024/macg_data) from Hugging Face using the [README download command](../README.md#2-download-the-training-data). The release is already converted to LeRobot v2.1, with images embedded in parquet files, and can be placed directly at `dataset/multi_task_combined` for the simulation multi-task recipes.

According to the public [dataset metadata](https://huggingface.co/datasets/MrBean2024/macg_data/blob/main/meta/info.json), it contains 700 episodes, 395,759 frames at 30 FPS, and 34 instruction entries, with all 700 episodes in the training split. The metadata identifies the robot type as `panda`. Each frame includes `global_image`, `image_agent0`, and `image_agent1` at 224 × 224 resolution, joint and gripper states for both arms, 16-dimensional actions, and a phase label.

```text
dataset/multi_task_combined/
├── data/chunk-000/episode_*.parquet
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── skill_labels.jsonl
└── norm_stats.json
```

`meta/skill_labels.jsonl` maps task indices to per-arm skill IDs and atomic instructions for full AE-VLA training. Preserve its correspondence with `meta/tasks.jsonl`; changes to task indexing require matching label updates. The training recipes recompute normalization by default. Serving the released model uses its checkpoint's own assets.

Set `DATASET_DIR=/path/to/download` to train from another location. This public upload is the training dataset; use the [paper and evaluation guide](evaluation.md) for held-out composition definitions. Keep the Hub revision in experiment records. To prepare new trajectories, continue below.

## Simulation data

The simulation workflow is **raw HDF5 → task-specific phase labels → LeRobot conversion → multi-task merge → normalization**. Phase boundaries and atomic instructions are task-dependent; inspect the generated visualizations before training.

| Task | Phase labeling and conversion directory |
| --- | --- |
| Two-arm bowl stacking | [`two_arms_stack_bowls`](../scripts/set_atom/macg/two_arms_stack_bowls/) |
| Two-arm cube stacking | [`twostackcubes`](../scripts/set_atom/macg/twostackcubes/) |
| Two-arm cube pushing | [`pushcube`](../scripts/set_atom/macg/pushcube/) |
| Three-arm placefactory | [`placefactory`](../scripts/set_atom/macg/placefactory/) · [walkthrough](MACG/README_placefactory.md) |
| Bottle handover | [`handover_bottles`](../scripts/set_atom/macg/handover_bottles/) |

Converters have task-specific arguments. For example, after phase labeling a bowl trajectory:

```bash
uv run python scripts/set_atom/macg/two_arms_stack_bowls/convert_h5_lerobot_placebowls.py --help
```

Use the converter's `--h5_path`, `--output_dir`, `--task_name`, and `--use_phase` options as appropriate. Some older phase-labeling scripts configure input paths inside the file. See [phase conversion notes](../scripts/README_phase_conversion.md) for the annotation format.

Merge prepared datasets with the same feature schema, image sizes, FPS, and embodiment:

```bash
uv run python scripts/set_atom/macg/merge_multitask_datasets.py \
  --sources dataset/MACG/two_arms_stack_bowl/two_arms_stack_bowl_combined \
            dataset/MACG/two_arms_stack_cube/two_arms_stack_cube_combined \
            dataset/MACG/two_arms_push_cube/two_arms_push_cube_combined \
  --output dataset/multi_task_combined
```

Use a new output directory. Existing outputs are rejected unless `--overwrite` is explicitly passed. Inputs are checked for matching feature schema, FPS, robot type, and episode files before writing. The merger reindexes episodes, frames, and task IDs and writes combined metadata. Its current implementation targets the converted **embedded-image parquet** format used by these recipes; it does not copy arbitrary external video trees. Do not pass an unrelated video-backed LeRobot dataset without adapting the converter/merger.

## Observation and action contract

The simulation two-arm recipes use 7 joint values + 1 gripper per arm. SO101/Bi-SO100 uses 5 joints + 1 gripper per arm. Model tensors are padded to `action_dim=32`; padding does not add physical joints. Recipes predict action chunks of length 50.

A typical compositional instruction is:

```text
agent0: Stack the bowl. | agent1: Push the cube.
```

The arm order must match state, action, and wrist-camera order. AE-VLA tokenizes instructions per agent; separate Pi0.5 selects only the target arm. Joint deltas and absolute gripper commands are handled by the configured transforms. Preserve those transforms through training, statistics computation, and serving.

## Normalization

```bash
bash scripts/macg_new/train_mavla.sh --norm-only
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_left_multi_task.sh --norm-only
bash scripts/macg_new/per-arm-policy/train_pi05_single_arm_right_multi_task.sh --norm-only
```

The statistics script disables arm shuffling, image masking/noise, and synthetic wait augmentation while retaining arm selection and delta-action transforms. It writes to the same `assets_dir / asset_id` used by training. With default paths:

| Recipe | Statistics file |
| --- | --- |
| Shared simulation policies | `dataset/multi_task_combined/norm_stats.json` |
| Separate left policy | `dataset/multi_task_combined_left/norm_stats.json` |
| Separate right policy | `dataset/multi_task_combined_right/norm_stats.json` |
| SO101 multi-task | `dataset/macg_so101/macg_multitask_phased/norm_stats.json` |
| Towel folding | `dataset/foldtowel_v2/norm_stats.json` |

The left/right directories contain **statistics**, not separate copies of the training dataset. Pi0.5 requires quantile statistics (`q01`, `q99`) in OpenPI's `norm_stats` JSON envelope. A raw mean/std/min/max JSON file is not interchangeable.

Changing `DATASET_DIR` changes which trajectories are read. Set `NORM_ASSETS_DIR` to a new assets root when switching experiments or datasets that must keep separate statistics. For example, `NORM_ASSETS_DIR=/data/run2/stats` stores left-arm statistics at `/data/run2/stats/multi_task_combined_left/norm_stats.json`.

Statistics are copied into checkpoints for inference. Use only the training split to compute them; reserve held-out compositions for [evaluation](evaluation.md).
