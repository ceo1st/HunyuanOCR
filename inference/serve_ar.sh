#!/bin/bash
# ============================================================================
# Deploy HunyuanOCR-1.5 (base, autoregressive) with vLLM.
# OpenAI-compatible /v1/chat/completions endpoint on a single GPU.
#
# For the multi-GPU / multi-instance production layout (8 replicas on
# ports 8000~8007), see the reference script at:
#   https://github.com/Tencent-Hunyuan/HunyuanOCR (branch: develop)
#   docs/inference.md → "Multi-GPU deployment"
# ============================================================================

set -e

MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
SERVED_NAME=${SERVED_NAME:-tencent/HunyuanOCR-1-5}
PORT=${PORT:-8000}
GPU=${GPU:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-131072}

LOG=${LOG:-vllm_ar_${PORT}.log}

echo "========================================"
echo "  HunyuanOCR-1.5 vLLM (AR baseline)"
echo "  model         : ${MODEL_PATH}"
echo "  served-as     : ${SERVED_NAME}"
echo "  gpu           : ${GPU}"
echo "  port          : ${PORT}"
echo "  gpu_mem_util  : ${GPU_MEM_UTIL}"
echo "  max_model_len : ${MAX_MODEL_LEN}"
echo "  log           : ${LOG}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4, "video":0}' \
    --trust_remote_code \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Wait for readiness with:"
echo "  curl -sf http://127.0.0.1:${PORT}/v1/models  ||  tail -f ${LOG}"
