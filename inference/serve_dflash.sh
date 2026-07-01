#!/bin/bash
# ============================================================================
# Deploy HunyuanOCR + DFlash draft with vLLM speculative decoding.
# OpenAI-compatible /v1/chat/completions endpoint.
#
# Requires: vllm==0.23.1rc1 with the DFlash patch applied.
# See docs/inference.md for install instructions.
# ============================================================================

set -e

MODEL_PATH=${MODEL_PATH:-/path/to/HunyuanOCR/base/model}
DFLASH_PATH=${DFLASH_PATH:-./hyocr_dflash}      # dir with config.json + model.safetensors + dflash.py
PORT=${PORT:-8001}
GPU=${GPU:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MEDIA_PATH=${MEDIA_PATH:-/tmp}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-15}

LOG=${LOG:-dflash_server_${PORT}.log}

# ────────────── Sanity check ──────────────
if [ ! -f "${DFLASH_PATH}/config.json" ] || [ ! -f "${DFLASH_PATH}/model.safetensors" ]; then
    echo "[ERROR] DFlash checkpoint incomplete at: ${DFLASH_PATH}"
    echo "        Expected: config.json + model.safetensors + dflash.py"
    exit 1
fi

echo "========================================"
echo "  HunyuanOCR + DFlash"
echo "  base model    : ${MODEL_PATH}"
echo "  draft model   : ${DFLASH_PATH}"
echo "  num spec tok  : ${NUM_SPEC_TOKENS}"
echo "  gpu           : ${GPU}"
echo "  port          : ${PORT}"
echo "  media         : ${MEDIA_PATH}"
echo "  log           : ${LOG}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --allowed-local-media-path "${MEDIA_PATH}" \
    --attention-backend flash_attn \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization ${GPU_MEM_UTIL} \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Wait for 'Application startup complete' with:  tail -f ${LOG}"
