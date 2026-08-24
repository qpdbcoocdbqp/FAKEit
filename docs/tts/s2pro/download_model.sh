#!/usr/bin/env bash
# 從 HuggingFace 下載 s2-pro GGUF 模型
# 用法: ./download_model.sh [variant]
# variant 可選: q6_k (預設) | q8_0 | q5_k_m | q4_k_m | q3_k | q2_k | f16

set -e

VARIANT="${1:-q6_k}"
REPO="rodrigomt/s2-pro-gguf"
FILE="s2-pro-${VARIANT}.gguf"
DEST="./models/${FILE}"

mkdir -p ./models

echo "▶ 下載模型: ${FILE}  (來源: huggingface.co/${REPO})"

if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "${REPO}" "${FILE}" --local-dir ./models
elif command -v wget &>/dev/null; then
    wget -c -O "${DEST}" \
        "https://huggingface.co/${REPO}/resolve/main/${FILE}"
else
    curl -L -C - -o "${DEST}" \
        "https://huggingface.co/${REPO}/resolve/main/${FILE}"
fi

echo "✅ 完成: ${DEST}"
echo ""
echo "記得在 docker-compose.yml 中把 MODEL_PATH 改為:"
echo "   /app/models/${FILE}"
