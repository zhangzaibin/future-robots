#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-pi05_bi_so100_flodtowel_pretrained_3k}"
EXP_NAME="${EXP_NAME:-foldtowel_pretrained_3k}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
CHECKPOINT_DIR="${FUTURE_CHECKPOINT_ROOT:-${ROOT_DIR}/checkpoints}/${CONFIG_NAME}/${EXP_NAME}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
REQUIRED_GPU_COUNT=2

cd "${ROOT_DIR}"
source scripts/setup_env.sh
export FUTURE_DATASET_DIR="${DEST_DATASET:-${FUTURE_DATASET_DIR:-${FUTURE_DATA_ROOT:-${ROOT_DIR}/dataset}/foldtowel_v2}}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
    CONFIG_NAME="${CONFIG_NAME}" bash scripts/so101/fold_towel/prepare_dataset.sh
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

latest_step=-1
for checkpoint in "${CHECKPOINT_DIR}"/[0-9]*; do
    if [[ -d "${checkpoint}/train_state" ]]; then
        step="${checkpoint##*/}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="${step}"
        fi
    fi
done

if (( latest_step < 0 )); then
    echo "No resumable checkpoint found in: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

if (( NUM_TRAIN_STEPS <= latest_step )); then
    echo "NUM_TRAIN_STEPS (${NUM_TRAIN_STEPS}) must be greater than the latest checkpoint step (${latest_step})." >&2
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    if (( ${#visible_devices[@]} < REQUIRED_GPU_COUNT )) || [[ "${CUDA_VISIBLE_DEVICES}" == -1 ]]; then
        echo "This resume recipe requires at least ${REQUIRED_GPU_COUNT} visible GPUs." >&2
        exit 1
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    if (( gpu_count < REQUIRED_GPU_COUNT )); then
        echo "This checkpoint was saved with ${REQUIRED_GPU_COUNT}-GPU FSDP sharding, but only ${gpu_count} GPU(s) are visible." >&2
        echo "Run this script in a session with at least ${REQUIRED_GPU_COUNT} visible GPUs." >&2
        exit 1
    fi
fi

echo "Resuming ${CONFIG_NAME}/${EXP_NAME} from step ${latest_step} to ${NUM_TRAIN_STEPS}."
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-1}" \
    uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --resume
