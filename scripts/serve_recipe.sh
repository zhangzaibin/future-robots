#!/usr/bin/env bash
# HTTP compatibility server for the existing robot clients.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
    echo 'Usage: CHECKPOINT_DIR=/path/to/step bash scripts/serve_recipe.sh CONFIG [--dry-run]'
    echo 'Optional: HOST (127.0.0.1), PORT (20011), CUDA_VISIBLE_DEVICES.'
    exit 0
fi
config="$1"
shift
if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    echo 'Set CHECKPOINT_DIR to a trained step directory containing params/ and assets/.' >&2
    exit 2
fi
source scripts/setup_env.sh
args=("${UV_BIN:-uv}" run --with flask deploy_minimal.py --config "${config}"
      --checkpoint-dir "${CHECKPOINT_DIR}" --host "${HOST:-127.0.0.1}" --port "${PORT:-20011}")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${args[@]}"; printf '\n'
    exit 0
fi
if [[ $# -gt 0 ]]; then echo "Unknown argument: $1" >&2; exit 2; fi
exec "${args[@]}"
