#!/bin/bash
# 下载 paligemma tokenizer 模型

set -e

TOKENIZER_URL="gs://big_vision/paligemma_tokenizer.model"
DEFAULT_DOWNLOAD_DIR="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"

# 可以通过环境变量或参数指定下载目录
if [ -n "${1}" ]; then
    DOWNLOAD_DIR="${1}"
elif [ -n "${TOKENIZER_DOWNLOAD_DIR}" ]; then
    DOWNLOAD_DIR="${TOKENIZER_DOWNLOAD_DIR}"
else
    DOWNLOAD_DIR="${DEFAULT_DOWNLOAD_DIR}"
fi

TOKENIZER_PATH="${DOWNLOAD_DIR}/paligemma_tokenizer.model"

echo "下载 paligemma tokenizer..."
echo "源: ${TOKENIZER_URL}"
echo "目标: ${TOKENIZER_PATH}"
echo ""

# 创建目录
mkdir -p "${DOWNLOAD_DIR}"

# 检查 gsutil
GSUTIL_CMD=""
if command -v gsutil &> /dev/null; then
    GSUTIL_CMD="gsutil"
elif [ -f "${HOME}/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL_CMD="${HOME}/google-cloud-sdk/bin/gsutil"
elif [ -f "/root/google-cloud-sdk/bin/gsutil" ]; then
    GSUTIL_CMD="/root/google-cloud-sdk/bin/gsutil"
else
    echo "错误: gsutil 未安装或不在 PATH 中"
    echo "请安装 Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 下载 tokenizer
echo "使用 gsutil 下载..."
"${GSUTIL_CMD}" cp "${TOKENIZER_URL}" "${TOKENIZER_PATH}"

echo ""
echo "✅ 下载完成！"
echo "Tokenizer 位置: ${TOKENIZER_PATH}"
echo ""
echo "设置环境变量以使用本地 tokenizer:"
echo "  export PALIGEMMA_TOKENIZER_PATH=${TOKENIZER_PATH}"

