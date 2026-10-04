#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT_DIR}/scripts/train_recipe.sh" aevla_pi05_two_arms_multi_task_full_attn "$@"
