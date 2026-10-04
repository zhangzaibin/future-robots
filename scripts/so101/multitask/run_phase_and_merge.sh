#!/usr/bin/env bash
# run_phase_and_merge.sh
#
# Full pipeline: phase-label all three SO101 v2 datasets, merge them into a
# single multi-task LeRobot dataset, and compute norm_stats.
#
# Usage (from repo root, with .venv active):
#   bash scripts/so101/multitask/run_phase_and_merge.sh
#
# Optional: override dataset root
#   DS_ROOT=/my/path bash scripts/so101/multitask/run_phase_and_merge.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DS_ROOT="${DS_ROOT:-${FUTURE_DATA_ROOT:-${REPO_ROOT}/dataset}/macg_so101}"
SCRIPTS="${REPO_ROOT}/scripts"
cd "${REPO_ROOT}"
source scripts/setup_env.sh

echo "=== Step 1: Phase-label each dataset ==="

uv run python "${SCRIPTS}/so101/multitask/set_phases_so101_v2.py" \
  --source "${DS_ROOT}/macg_push_cubes_v2" \
  --output "${DS_ROOT}/macg_push_cubes_v2_phased" \
  --visualize

uv run python "${SCRIPTS}/so101/multitask/set_phases_so101_v2.py" \
  --source "${DS_ROOT}/macg_stack_bowls_v2" \
  --output "${DS_ROOT}/macg_stack_bowls_v2_phased" \
  --atoms "pick=Pick up the bowl.,action=Stack the bowl.,recover=Recover." \
  --visualize

uv run python "${SCRIPTS}/so101/multitask/set_phases_so101_v2.py" \
  --source "${DS_ROOT}/macg_stack_cubes_v2" \
  --output "${DS_ROOT}/macg_stack_cubes_v2_phased" \
  --atoms "pick=Pick up the cube.,action=Stack the cube.,recover=Recover." \
  --visualize

echo ""
echo "=== Step 2: Merge into multi-task dataset ==="

uv run python "${SCRIPTS}/set_atom/macg/merge_multitask_datasets.py" \
  --sources \
    "${DS_ROOT}/macg_push_cubes_v2_phased" \
    "${DS_ROOT}/macg_stack_bowls_v2_phased" \
    "${DS_ROOT}/macg_stack_cubes_v2_phased" \
  --output "${DS_ROOT}/macg_multitask_phased"

echo ""
echo "=== Step 3: Compute training-compatible normalization statistics ==="
DATASET_DIR="${DS_ROOT}/macg_multitask_phased" \
    bash "${REPO_ROOT}/scripts/train_recipe.sh" "${CONFIG_NAME:-aevla_pi05_so101_multitask}" --norm-only

echo "Merged dataset: ${DS_ROOT}/macg_multitask_phased"
