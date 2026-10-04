#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_NAME="pi05_bi_so100_flodtowel_pretrained_3k_1gpu"
EXP_NAME="${EXP_NAME:-foldtowel_v2_1gpu}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_ID="${GPU_ID:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SOURCE_PARAMS="${FUTURE_FOLDTOWEL_PARAMS:-${FUTURE_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}/pi05_bi_so100_flodtowel/so101_test/14999/params}"
CHECKPOINT_DIR="${FUTURE_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}/${CONFIG_NAME}/${EXP_NAME}"

cd "${ROOT_DIR}"
source scripts/setup_env.sh
export FUTURE_FOLDTOWEL_PARAMS="${SOURCE_PARAMS}"
export FUTURE_DATASET_DIR="${DEST_DATASET:-${FUTURE_DATASET_DIR:-${FUTURE_DATA_ROOT:-${ROOT_DIR}/dataset}/foldtowel_v2}}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
    CONFIG_NAME="${CONFIG_NAME}" bash scripts/so101/fold_towel/prepare_dataset.sh
fi

if [[ ! -d "${SOURCE_PARAMS}" ]]; then
    echo "Source model parameters not found: ${SOURCE_PARAMS}" >&2
    exit 1
fi

train_args=(
    "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --batch-size "${BATCH_SIZE}"
)

if [[ -d "${CHECKPOINT_DIR}" ]]; then
    echo "Output directory already exists: ${CHECKPOINT_DIR}" >&2
    echo "Set a different EXP_NAME for a new training run." >&2
    exit 1
fi

echo "Starting a new one-GPU run using pretrained weights: ${SOURCE_PARAMS}"
echo "The optimizer and learning-rate schedule will start from step 0."

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
    uv run scripts/train.py "${train_args[@]}"
