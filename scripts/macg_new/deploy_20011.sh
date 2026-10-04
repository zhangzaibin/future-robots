#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PORT="${PORT:-20011}"
exec bash "${ROOT_DIR}/scripts/serve_recipe.sh" aevla_pi05_skill_lora_two_arms_multitask "$@"
