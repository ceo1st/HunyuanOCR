#!/bin/bash
# ============================================================================
# Deploy HunyuanOCR-1.5 + DFlash draft with vLLM speculative decoding.
# OpenAI-compatible /v1/chat/completions endpoint on a single GPU.
#
# Requires: a vLLM build with the DFlash speculative-decoding method registered.
# See docs/inference.md for install instructions.
# ============================================================================

set -e

MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
DFLASH_PATH=${DFLASH_PATH:-./hyocr_dflash}      # dir with config.json + model.safetensors + dflash.py
SERVED_NAME=${SERVED_NAME:-tencent/HunyuanOCR-1-5}
PORT=${PORT:-8001}
GPU=${GPU:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-131072}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-15}

LOG=${LOG:-vllm_dflash_${PORT}.log}

# ────────────── Sanity check ──────────────
if [ ! -f "${DFLASH_PATH}/config.json" ] || [ ! -f "${DFLASH_PATH}/model.safetensors" ]; then
    echo "[ERROR] DFlash checkpoint incomplete at: ${DFLASH_PATH}"
    echo "        Expected: config.json + model.safetensors + dflash.py"
    exit 1
fi

echo "========================================"
echo "  HunyuanOCR-1.5 vLLM + DFlash"
echo "  base model    : ${MODEL_PATH}"
echo "  draft model   : ${DFLASH_PATH}"
echo "  served-as     : ${SERVED_NAME}"
echo "  num spec tok  : ${NUM_SPEC_TOKENS}"
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
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Wait for readiness with:"
echo "  curl -sf http://127.0.0.1:${PORT}/v1/models  ||  tail -f ${LOG}"
