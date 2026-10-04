#!/usr/bin/env bash
# Explicit opt-in transfer using your existing SSH agent or SSH config.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash transport.sh LOCAL_CHECKPOINT_DIR USER@HOST:DESTINATION" >&2
    exit 2
fi
if [[ ! -d "$1" ]]; then echo "Checkpoint directory not found: $1" >&2; exit 1; fi
scp -r -- "$1" "$2"
# Local checkpoints are retained after transfer.
