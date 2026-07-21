#!/bin/bash
# ============================================================================
# HunyuanOCR-1.5 + DFlash vLLM single-GPU serving (speculative decoding).
#
# This script mounts a DFlash draft model and enables speculative decoding via
# --speculative-config. The clients (infer_vllm_client.py / batch_infer.py) are
# shared with the AR path in ../vLLM/; sampling and post-processing are
# identical, so AR and DFlash outputs are directly comparable.
#
# Prerequisite: the unified inference environment is installed and activated,
# and the model weights are downloaded. See docs/inference/inference.md.
#
# Usage:
#   MODEL_PATH=/path/to/HunyuanOCR GPU=0 PORT=8000 bash serve_DFlash.sh
#
# Environment variables (all overridable):
#   MODEL_PATH      base model weights dir (required)
#   DFLASH_PATH     DFlash draft model dir        default ${MODEL_PATH}/dflash
#   GPU             GPU index                     default 0
#   PORT            service port                  default 8000
#   GPU_MEM_UTIL    GPU memory fraction           default 0.85 (draft adds ~0.7GB)
#   MAX_MODEL_LEN   context window length         default 131072
#   NUM_SPEC_TOKENS speculative tokens per step   default 15 (official recommendation)
#   SERVED_NAME     served model name             default tencent/HunyuanOCR
# ============================================================================
set -e

MODEL_PATH=${MODEL_PATH:?"Please set MODEL_PATH=base model weights dir"}
DFLASH_PATH=${DFLASH_PATH:-${MODEL_PATH}/dflash}
GPU=${GPU:-0}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-15}
SERVED_NAME=${SERVED_NAME:-tencent/HunyuanOCR}
LOG=${LOG:-vllm_dflash_${PORT}.log}

# ────────────── Draft model integrity check ──────────────
if [ ! -f "${DFLASH_PATH}/config.json" ] || [ ! -f "${DFLASH_PATH}/model.safetensors" ]; then
    echo "[ERROR] DFlash draft model is incomplete: ${DFLASH_PATH}"
    echo "        required: config.json + dflash.py + model.safetensors"
    echo "        the draft lives in the dflash/ subfolder of the HF model repo;"
    echo "        make sure you downloaded it together with the base model:"
    echo "        huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude \"v1.0/*\""
    echo "        (then DFLASH_PATH defaults to \${MODEL_PATH}/dflash)"
    exit 1
fi

# Local weights: disable network lookups to speed up startup.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================"
echo "  HunyuanOCR-1.5 vLLM + DFlash"
echo "  base model    : ${MODEL_PATH}"
echo "  draft model   : ${DFLASH_PATH}"
echo "  served-as     : ${SERVED_NAME}"
echo "  num spec tok  : ${NUM_SPEC_TOKENS}"
echo "  gpu / port    : ${GPU} / ${PORT}"
echo "  gpu_mem_util  : ${GPU_MEM_UTIL}"
echo "  max_model_len : ${MAX_MODEL_LEN}"
echo "  log           : ${LOG}"
echo "========================================"

CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    > "${LOG}" 2>&1 &

echo "[started] pid=$!  log=${LOG}"
echo "Readiness check (first load includes torch.compile, ~3-5 min):"
echo "  curl -sf http://127.0.0.1:${PORT}/v1/models || tail -f ${LOG}"
