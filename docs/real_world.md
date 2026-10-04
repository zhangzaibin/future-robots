# Real-world training

[Documentation](README.md) · [Installation](installation.md) · [Deployment](deployment.md)

The included SO101/Bi-SO100 pipeline uses **12 physical dimensions**: left arm 5 joints + gripper, then right arm 5 joints + gripper. This differs from the simulation recipes' 16 dimensions. Use the real-world configurations so the converters, delta masks, arm selection, and action outputs agree.

## Convert LeRobot v3 recordings

For a dual-arm recording:

```bash
uv run python scripts/convert_lerobot_v3_to_v2.py \
  --src /data/recordings/foldtowel \
  --dst dataset/foldtowel_v2 \
  --robot-type bi_so100_follower \
  --default-prompt 'fold the towel'
```

The converter supports `--camera-mapping` and single-arm data; inspect `--help` for your recording schema. Verify camera/arm correspondence, joint units, gripper direction, and a few reconstructed trajectories before using the converted data. The generated v2 data uses embedded images and the flattened keys expected by these training recipes.

## SO101 multi-task

Prepare these v2 datasets under `dataset/macg_so101/`:

- `macg_push_cubes_v2`
- `macg_stack_bowls_v2`
- `macg_stack_cubes_v2`

The pipeline detects phases, writes labeled copies, merges them, and computes OpenPI normalization statistics using the actual training transforms:

```bash
bash scripts/so101/multitask/run_phase_and_merge.sh
bash scripts/so101/train_multitask.sh --skip-norm
```

The configuration is `aevla_pi05_so101_multitask`; it reads `dataset/macg_so101/macg_multitask_phased`, uses 6 dimensions per arm, and defaults to 15,000 steps with two FSDP devices. For a different data root:

```bash
DS_ROOT=/data/so101 bash scripts/so101/multitask/run_phase_and_merge.sh
DATASET_DIR=/data/so101/macg_multitask_phased \
  bash scripts/so101/train_multitask.sh --skip-norm
```

The pipeline's `DS_ROOT` also determines the dataset read by the statistics step. Statistics still use the selected config's assets location; set `FUTURE_ASSETS_DIR` consistently if you want a different assets root.

Inspect a single dataset without writing labels first:

```bash
uv run python scripts/so101/multitask/set_phases_so101_v2.py \
  --source dataset/macg_so101/macg_stack_bowls_v2 \
  --output dataset/macg_so101/macg_stack_bowls_v2_phased \
  --atoms 'pick=Pick up the bowl.,action=Stack the bowl.,recover=Recover.' \
  --dry-run --visualize
```

See [phase detection notes](../scripts/so101/README.md) for task assumptions. Output directories should be fresh when rerunning phase labeling or merging; this multi-task pipeline is not an incremental DAgger merger.

## Towel folding

To fine-tune from the public Pi0.5 initialization, put the raw v3 recording at `dataset/foldtowel`, then run:

```bash
EXP_NAME=foldtowel_run1 bash scripts/so101/fold_towel/train.sh
```

This converts to `dataset/foldtowel_v2` if necessary, computes statistics, then trains `pi05_bi_so100_flodtowel`. The historical `flodtowel` spelling is retained so existing checkpoint paths remain usable. The default instruction is `fold the towel`.

```bash
SRC_DATASET=/data/raw/towels DST_DATASET=/data/converted/towels_v2 \
  EXP_NAME=foldtowel_run2 bash scripts/so101/fold_towel/train.sh
```

`DST_DATASET` is forwarded to the training configuration, so conversion, normalization, and training use the same data. An existing conversion is reused; changing the source requires a new destination. `PROMPT` sets conversion text; the default towel configs use a fixed `fold the towel` prompt, so adapting to another task also requires changing the model prompt/config.

The default towel config uses 30,000 steps, batch 48, two FSDP devices, and saves every 15,000 steps. Use `BATCH_SIZE`, `NUM_TRAIN_STEPS`, and `FSDP_DEVICES` to override these. It preserves an existing experiment unless `--overwrite` is explicitly passed. The common launcher's dry-run previews training only; running the towel preparation wrapper itself can still convert data.

## DAgger and resuming

The DAgger workflow starts with the base v3 recording plus an experience archive:

```bash
BASE_DATASET=/data/raw/foldtowel \
EXPERIENCE_ARCHIVE=/data/dagger.zip \
EXPERIENCE_DATASET=/data/dagger_extracted \
DEST_DATASET=/data/converted/foldtowel_v2 \
  bash scripts/so101/fold_towel/prepare_dataset.sh
```

The preparation script fingerprints appended experience data to avoid duplicate appends, reindexes episodes/frames/tasks, and recomputes statistics after new data is added. Its archive handling uses Linux tools (`sha256sum`, `unzip`). Default filenames retain the original experiment names: `dataset/dagger0916.zip` and `dataset/dagger0916`.

**Resume a complete checkpoint**, including optimizer and step:

```bash
CONFIG_NAME=pi05_bi_so100_flodtowel EXP_NAME=foldtowel_run1 \
  SKIP_PREPARE=1 NUM_TRAIN_STEPS=60000 \
  bash scripts/so101/fold_towel/resume_train.sh
```

`SKIP_PREPARE=1` is appropriate after explicitly preparing data/statistics, or when data has not changed. Omit it to run the DAgger preparation first. Use the same `DEST_DATASET` / `FUTURE_DATASET_DIR` and assets root across preparation and training. This resume helper expects a two-GPU FSDP checkpoint and requires at least two visible GPUs. It restores from an experiment containing `train_state/`, not just exported `params/`.

**Start a new single-GPU run from model parameters**, with fresh optimizer and step zero:

```bash
FUTURE_FOLDTOWEL_PARAMS=/data/checkpoints/foldtowel/14999/params \
  SKIP_PREPARE=1 GPU_ID=0 BATCH_SIZE=4 EXP_NAME=foldtowel_1gpu \
  bash scripts/so101/fold_towel/train_1gpu.sh
```

This selects `pi05_bi_so100_flodtowel_pretrained_3k_1gpu`. The initialization checkpoint is not distributed with this repository; provide your own trained parameters. The default source is `checkpoints/pi05_bi_so100_flodtowel/so101_test/14999/params`. An existing output directory is rejected.

## Connect to the robot

Serve the exact training config with the complete saved step directory using [deployment instructions](deployment.md). Keep the robot client separate from the training environment. Check observation keys, physical left/right order, action units, and delta-to-absolute conversion before enabling actuators. Start with recorded observations, then use supervised low-speed rollouts with an accessible stop control.
