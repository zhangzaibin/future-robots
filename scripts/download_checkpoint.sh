#!/bin/bash
# Script to download checkpoints using gsutil (alternative to Python download)

set -e

MODEL_NAME="${1:-pi05_base}"
GCS_PATH="gs://openpi-assets/checkpoints/${MODEL_NAME}"

# Expand ~ to home directory
if [ -n "${OPENPI_DATA_HOME}" ]; then
    CACHE_DIR="${OPENPI_DATA_HOME}"
else
    CACHE_DIR="${HOME}/.cache/openpi"
fi

LOCAL_PATH="${CACHE_DIR}/openpi-assets/checkpoints/${MODEL_NAME}"

echo "Downloading ${MODEL_NAME} checkpoint..."
echo "Source: ${GCS_PATH}"
echo "Destination: ${LOCAL_PATH}"

# Create directory if it doesn't exist (parent directory for the checkpoint)
mkdir -p "${LOCAL_PATH}"

# Check if gsutil is available
GSUTIL_CMD=""
if command -v gsutil &> /dev/null; then
    GSUTIL_CMD="gsutil"
elif [ -f "${HOME}/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL_CMD="${HOME}/google-cloud-sdk/bin/gsutil"
elif [ -f "/root/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL_CMD="/root/google-cloud-sdk/bin/gsutil"
else
    echo "Error: gsutil is not installed or not found in PATH."
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    echo "Or add it to PATH: export PATH=\$PATH:\$HOME/google-cloud-sdk/bin"
    exit 1
fi

# Download using gsutil with anonymous access
# Ensure destination directory exists and ends with / for gsutil
echo "Starting download with gsutil..."
PARENT_DIR="$(dirname "${LOCAL_PATH}")"
mkdir -p "${PARENT_DIR}"

# Remove existing directory if it exists (to avoid conflicts)
if [ -d "${LOCAL_PATH}" ]; then
    echo "Removing existing directory: ${LOCAL_PATH}"
    rm -rf "${LOCAL_PATH}"
fi

# Download to parent directory, which will create the model_name subdirectory
echo "Using gsutil at: ${GSUTIL_CMD}"
"${GSUTIL_CMD}" -m cp -r "${GCS_PATH}" "${PARENT_DIR}/"

echo "Download completed!"
echo "Checkpoint location: ${LOCAL_PATH}"

