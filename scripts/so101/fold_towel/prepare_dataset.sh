#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${FUTURE_DATA_ROOT:-${ROOT_DIR}/dataset}"
BASE_DATASET="${BASE_DATASET:-${DATA_ROOT}/foldtowel}"
EXPERIENCE_ARCHIVE="${EXPERIENCE_ARCHIVE:-${DATA_ROOT}/dagger0916.zip}"
EXPERIENCE_DATASET="${EXPERIENCE_DATASET:-${DATA_ROOT}/dagger0916}"
DEST_DATASET="${DEST_DATASET:-${FUTURE_DATASET_DIR:-${DATA_ROOT}/foldtowel_v2}}"
CONFIG_NAME="${CONFIG_NAME:-pi05_bi_so100_flodtowel_pretrained_3k}"
PROMPT="${PROMPT:-fold the towel}"
ARCHIVE_STAMP="${EXPERIENCE_DATASET}/.source_archive.sha256"

cd "${ROOT_DIR}"
source scripts/setup_env.sh
export FUTURE_DATASET_DIR="${DEST_DATASET}"

if [[ ! -f "${BASE_DATASET}/meta/info.json" ]]; then
    echo "Base dataset not found: ${BASE_DATASET}" >&2
    exit 1
fi
if [[ ! -f "${EXPERIENCE_ARCHIVE}" ]]; then
    echo "Experience archive not found: ${EXPERIENCE_ARCHIVE}" >&2
    exit 1
fi

archive_hash="$(sha256sum "${EXPERIENCE_ARCHIVE}" | awk '{print $1}')"
extracted_hash=""
if [[ -f "${ARCHIVE_STAMP}" ]]; then
    extracted_hash="$(<"${ARCHIVE_STAMP}")"
fi

if [[ ! -f "${EXPERIENCE_DATASET}/meta/info.json" ]]; then
    temporary_dir="$(mktemp -d "${DATA_ROOT}/.foldtowel_extract.XXXXXX")"
    trap 'rm -rf "${temporary_dir}"' EXIT
    unzip -q "${EXPERIENCE_ARCHIVE}" -d "${temporary_dir}"
    archive_root="${temporary_dir}/$(unzip -Z1 "${EXPERIENCE_ARCHIVE}" | awk -F/ 'NF {print $1; exit}')"
    if [[ ! -f "${archive_root}/meta/info.json" ]]; then
        echo "Archive does not contain a valid LeRobot dataset: ${EXPERIENCE_ARCHIVE}" >&2
        exit 1
    fi
    mv "${archive_root}" "${EXPERIENCE_DATASET}"
    printf '%s\n' "${archive_hash}" >"${ARCHIVE_STAMP}"
    rm -rf "${temporary_dir}"
    trap - EXIT
elif [[ -n "${extracted_hash}" && "${extracted_hash}" != "${archive_hash}" ]]; then
    echo "The archive changed after extraction: ${EXPERIENCE_ARCHIVE}" >&2
    echo "Choose a new EXPERIENCE_DATASET path or remove the old extracted copy after checking it." >&2
    exit 1
else
    echo "Using extracted experience dataset: ${EXPERIENCE_DATASET}"
fi

if [[ ! -f "${DEST_DATASET}/meta/info.json" ]]; then
    uv run python scripts/convert_lerobot_v3_to_v2.py \
        --src "${BASE_DATASET}" \
        --dst "${DEST_DATASET}" \
        --robot-type bi_so100_follower \
        --default-prompt "${PROMPT}"
else
    echo "Using existing base conversion: ${DEST_DATASET}"
fi

uv run python scripts/so101/fold_towel/merge_experiences.py \
    "${EXPERIENCE_DATASET}" \
    --destination "${DEST_DATASET}" \
    --prompt "${PROMPT}" \
    --config-name "${CONFIG_NAME}"
