#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_DATASET="${SRC_DATASET:-${FUTURE_DATA_ROOT:-${ROOT_DIR}/dataset}/foldtowel}"
DST_DATASET="${DST_DATASET:-${FUTURE_DATA_ROOT:-${ROOT_DIR}/dataset}/foldtowel_v2}"
CONFIG_NAME="${CONFIG_NAME:-pi05_bi_so100_flodtowel}"
EXP_NAME="${EXP_NAME:-so101_test}"
PROMPT="${PROMPT:-fold the towel}"

cd "${ROOT_DIR}"
source scripts/setup_env.sh

if [[ ! -f "${SRC_DATASET}/meta/info.json" ]]; then
    echo "找不到源数据集：${SRC_DATASET}" >&2
    exit 1
fi

# 原始数据是 LeRobot v3；测试时转换并使用全部 episode。
if [[ ! -f "${DST_DATASET}/meta/info.json" ]]; then
    uv run python scripts/convert_lerobot_v3_to_v2.py \
        --src "${SRC_DATASET}" \
        --dst "${DST_DATASET}" \
        --robot-type bi_so100_follower \
        --default-prompt "${PROMPT}"
else
    echo "检测到已转换数据集，跳过转换：${DST_DATASET}"
fi

export DATASET_DIR="${DST_DATASET}"
export EXP_NAME
exec bash scripts/train_recipe.sh "${CONFIG_NAME}" "$@"
