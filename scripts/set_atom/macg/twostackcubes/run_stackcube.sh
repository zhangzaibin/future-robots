#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
bash "${ROOT_DIR}/scripts/train_recipe.sh" aevla_pi05_two_arms_stack_cubes "$@"
bash "${ROOT_DIR}/scripts/train_recipe.sh" pi05_two_arms_stack_cubes "$@"
