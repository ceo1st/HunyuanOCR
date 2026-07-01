#!/bin/bash
# ============================================================================
# Deploy HunyuanOCR base model with vLLM (autoregressive baseline).
# OpenAI-compatible /v1/chat/completions endpoint.
# ============================================================================

set -e

MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
PORT=${PORT:-8000}
GPU=${GPU:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MEDIA_PATH=${MEDIA_PATH:-/tmp}     # allowed local media path for images

LOG=${LOG:-ar_server_${PORT}.log}

echo "========================================"
echo "  HunyuanOCR AR baseline"
echo "  model     : ${MODEL_PATH}"
echo "  gpu       : ${GPU}"
echo "  port      : ${PORT}"
echo "  media     : ${MEDIA_PATH}"
echo "  log       : ${LOG}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_PATH}" \
    --port ${PORT} \
    --allowed-local-media-path "${MEDIA_PATH}" \
    --attention-backend flash_attn \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Wait for 'Application startup complete' with:  tail -f ${LOG}"
