#!/usr/bin/env bash
# Source this file. User-provided paths and cache settings take precedence.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${FUTURE_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dataset}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/openpi}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# PALIGEMMA_TOKENIZER_PATH is optional; when unset the upstream downloader is used.
