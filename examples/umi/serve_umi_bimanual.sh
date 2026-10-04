#!/usr/bin/env bash
# Launch the pi0_umi_bimanual VLA policy server (Flask HTTP).
#
# Usage:
#   bash examples/umi/serve_umi_bimanual.sh <checkpoint_dir> [port]
#
# Arguments:
#   checkpoint_dir  Path to the trained checkpoint step directory, e.g.
#                   ./checkpoints/pi0_umi_bimanual/exp/9999
#   port            HTTP port (default: 20022)
#
# Example:
#   bash examples/umi/serve_umi_bimanual.sh \
#       ./checkpoints/pi0_umi_bimanual/exp/9999 \
#       20022

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [port]}"
PORT="${2:-20022}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================================="
echo "  pi0_umi_bimanual VLA Server (Flask HTTP)"
echo "  checkpoint : ${CHECKPOINT_DIR}"
echo "  port       : ${PORT}"
echo "=============================================="

cd "${REPO_ROOT}"
source .venv/bin/activate

python server_agent0.py \
    --config pi0_umi_bimanual \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --port "${PORT}"
