#!/usr/bin/env bash
# Shared entry point; recipe wrappers pass their config as the first argument.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
    cat <<'EOF'
Usage: bash scripts/train_recipe.sh CONFIG [OPTIONS] [TRAIN_ARGS...]
  --dry-run     Print commands without loading data, downloading, or training
  --resume      Restore the latest checkpoint (skip norm stats by default)
  --overwrite   Explicitly replace an existing experiment
  --skip-norm   Use existing normalization statistics
  --norm-only   Compute statistics without starting training
  --help        Show this help

Environment: EXP_NAME, DATASET_DIR, NORM_ASSETS_DIR, BATCH_SIZE,
NUM_TRAIN_STEPS, FSDP_DEVICES, CUDA_VISIBLE_DEVICES, FUTURE_DATA_ROOT,
FUTURE_CHECKPOINT_ROOT, FUTURE_PI05_PARAMS, UV_BIN (default: uv).
All other arguments are passed to scripts/train.py unchanged.
EOF
}

if [[ $# -eq 0 || "${1:-}" == --help ]]; then usage; exit 0; fi
config="$1"
shift
dry_run=0
resume=0
overwrite=0
skip_norm=0
norm_only=0
extra_args=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) dry_run=1 ;;
        --resume) resume=1 ;;
        --overwrite) overwrite=1 ;;
        --skip-norm) skip_norm=1 ;;
        --norm-only) norm_only=1 ;;
        --help|-h) usage; exit 0 ;;
        *) extra_args+=("${arg}") ;;
    esac
done
if (( resume && overwrite )); then
    echo "--resume and --overwrite are mutually exclusive." >&2; exit 2
fi
if (( norm_only && (resume || skip_norm) )); then
    echo "--norm-only cannot be combined with --resume or --skip-norm." >&2; exit 2
fi

source "${ROOT_DIR}/scripts/setup_env.sh"
if [[ -n "${DATASET_DIR:-}" ]]; then export FUTURE_DATASET_DIR="${DATASET_DIR}"; fi
if [[ -n "${NORM_ASSETS_DIR:-}" ]]; then export FUTURE_ASSETS_DIR="${NORM_ASSETS_DIR}"; fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
uv_bin="${UV_BIN:-uv}"

run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if (( ! dry_run )); then "$@"; fi
}

if (( ! dry_run )) && ! command -v "${uv_bin}" >/dev/null 2>&1; then
    echo "uv is required. See docs/installation.md." >&2; exit 127
fi
printf 'Config: %s\nExperiment: %s\n' "${config}" "${EXP_NAME:-${config}}"
if [[ -n "${FUTURE_DATASET_DIR:-}" ]]; then printf 'Dataset: %s\n' "${FUTURE_DATASET_DIR}"; fi
if (( ! skip_norm && ! resume )); then
    run "${uv_bin}" run scripts/compute_norm_stats.py --config-name "${config}"
fi
if (( norm_only )); then exit 0; fi

train_args=("${config}" --exp-name "${EXP_NAME:-${config}}")
if [[ -n "${BATCH_SIZE:-}" ]]; then train_args+=(--batch-size "${BATCH_SIZE}"); fi
if [[ -n "${NUM_TRAIN_STEPS:-}" ]]; then train_args+=(--num-train-steps "${NUM_TRAIN_STEPS}"); fi
if [[ -n "${FSDP_DEVICES:-}" ]]; then train_args+=(--fsdp-devices "${FSDP_DEVICES}"); fi
if (( resume )); then train_args+=(--resume); fi
if (( overwrite )); then train_args+=(--overwrite); fi
run "${uv_bin}" run scripts/train.py "${train_args[@]}" ${extra_args[@]+"${extra_args[@]}"}
