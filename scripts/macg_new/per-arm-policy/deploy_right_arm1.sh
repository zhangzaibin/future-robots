#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PORT="${PORT:-20022}"
exec bash "${ROOT_DIR}/scripts/serve_recipe.sh" pi05_single_arm_right_multi_task "$@"
