#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${ROOT_DIR}/scripts/train_recipe.sh" pi05_single_arm_left_multi_task "$@"
