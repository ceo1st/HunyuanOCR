<div align="center">

# HunyuanOCR-1.5: Towards Efficient and Effective E2E OCR

</div>

> 📝 **Note:** The technical report and model weights of HunyuanOCR-1.5 are **coming very soon**.
> This branch (`develop`) hosts the open-source **training and inference toolkit** for HunyuanOCR-1.5.
> For the original HunyuanOCR 1.0 release, please switch to the `main` branch or refer to
> [`README_v1.0.md`](./README_v1.0.md) / [`README_zh_v1.0.md`](./README_zh_v1.0.md).

---

## 📖 Introduction

**HunyuanOCR-1.5** is a lightweight, end-to-end OCR-specialized vision-language model. It targets a
broad range of text-centric visual tasks and unifies **document parsing, text spotting, information
extraction, text-image translation** within a single
end-to-end VLM.

Building upon the validated lightweight architecture of HunyuanOCR-1.0, HunyuanOCR-1.5 does **not**
redesign the model backbone. Instead, it performs a systematic upgrade around two goals — **making
the model faster and better**:

- ⚡ **Faster — DFlash inference acceleration.**
 End-to-end OCR is often accompanied by long autoregressive decoding, which becomes the major
 bottleneck for dense documents, tables, formulas, and other long structured outputs.
 HunyuanOCR-1.5 adapts a speculative-decoding framework based on **DFlash**: a lightweight
 block-diffusion draft model drafts multiple candidate tokens in parallel, which are then verified
 by the target model in a single pass. This significantly reduces the decoding latency of long
 structured outputs while **preserving the output distribution** of the target model.

- 💻 **PC-side deployment via llama.cpp.**
 Beyond server-grade vLLM, HunyuanOCR-1.5 also supports **CPU / consumer-GPU / laptop** deployment
 through [`llama.cpp`](https://github.com/ggml-org/llama.cpp) with a GGUF-converted checkpoint and
 an OpenAI-compatible `llama-server`. A DFlash-adapted `llama.cpp` fork is provided as well, so the
 same speculative-decoding acceleration is available on PC. See
 [`docs/llama_cpp.md`](docs/llama_cpp.md).

- 🧠 **Better — Agentic Data Flow + upgraded training recipe.**
 On the data side, we propose **Agentic Data Flow**, an agent-driven data-construction system that
 translates model weaknesses into executable data requirements. Agents deeply participate in
 material search, tool-based verification, sample cleaning, and data-pipeline development, and
 iterate in a closed loop with algorithm engineers. In HunyuanOCR-1.5, this system is used for
 targeted long-tail capabilities such as **low-resource OCR, ancient-script OCR, and multi-image
 text-centric QA**.
 On the training side, we systematically upgrade the recipe: pretraining Stage-3 is re-planned to
 incorporate the newly produced capability data, multi-image data, and historical OCR data, with
 maximum image resolution extended to **4K** and context window extended to **128K**; post-training
 refines the SFT data and further explores RL across different OCR tasks to amplify the gains from
 reinforcement learning.

Together, HunyuanOCR-1.5 achieves **both faster inference and broader OCR capability coverage**
while retaining the deployment advantages of a lightweight end-to-end model. This repository
open-sources the SFT / DFlash training pipeline and the transformers / vLLM inference stack, so
that the community can reproduce, fine-tune, and extend OCR-specialized VLMs.

---

## ⚙️ Environment

- Python 3.10+
- PyTorch 2.1+ (CUDA 12.1 recommended)
- transformers 4.57+
- DeepSpeed 0.14+
- vLLM 0.23.1rc1 + DFlash patch (only required for DFlash inference — see [`docs/inference.md`](docs/inference.md))

Install common training / inference dependencies:

```bash
pip install -r requirements.txt
# flash-attn requires manual build:
pip install flash-attn --no-build-isolation
```

---

## 🚀 Training

All training scripts live under `scripts/` and share `scripts/env_common.sh` for distributed env
variables. Multi-node training is supported via the standard
`NNODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` env vars.

### 1. Prepare packed training data

We tokenize each raw OCR JSONL, then pack multiple samples up to `packed_max_length=20480`
tokens into single sequences to maximize GPU utilization.

**Step 1** — fill in `configs/data_list.txt` with one absolute path per line, each pointing to a
raw OCR JSONL file. The JSONL schema is documented in [`docs/data_format.md`](docs/data_format.md).

**Step 2** — run the multi-process count-and-pack pipeline:

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
INPUT_LIST=./configs/data_list.txt \
PACK_LEN=20480 \
NUM_PROCESSES=32 \
THREADS_PER_PROCESS=8 \
bash scripts/pack_data.sh
```

Output: `./data/parsing_packed_20480.jsonl` — a single sequence-packed JSONL ready for training.

The pipeline is implemented in [`tools/pipeline_count_and_pack.py`](tools/pipeline_count_and_pack.py)
and [`tools/pack_from_counted.py`](tools/pack_from_counted.py).

### 2. SFT the HunyuanOCR base model

Full end-to-end SFT (vision encoder + MLP + LLM) on packed OCR sequences.
Default profile: `lr=2e-5`, `epochs=5`, per-GPU batch=1, `packed_max_length=20480`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_base.sh
```

Entry: [`train/train_hunyuan.py`](train/train_hunyuan.py).
Full argument list: see [`docs/training.md`](docs/training.md).

### 3. Train the DFlash draft model — from scratch

Trains a small block-diffusion draft that predicts K speculative tokens for HunyuanOCR.
Default profile: `lr=1e-4`, `epochs=2`, `num_mask_tokens=16`, `sample_block_num=8`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash.sh
```

Entry: [`train/train_draft.py`](train/train_draft.py).

### 4. Continue-finetune from an existing DFlash checkpoint

Use this when adapting a released DFlash draft to a smaller / domain-specific dataset.
Recommended profile: `lr=2e-5`, `epochs=10`, `warmup_ratio=0.05`, `save_steps=500`.

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
DFLASH_INIT=/path/to/hyocr_dflash/existing_checkpoint \
TRAIN_DATA=./data/parsing_packed_20480.jsonl \
NPROC_PER_NODE=8 \
bash scripts/sft_dflash_finetune.sh
```

Entry: [`train/train_draft_from_dflash.py`](train/train_draft_from_dflash.py).

---

## 🧪 Inference

Three paths are provided, all runnable on a **single image** for smoke-testing:

- **A. HuggingFace transformers** — single-image, easy to hack, recommended for correctness debugging.
- **B. vLLM (OpenAI-compatible)** — production serving; required for real DFlash speedup.
- **C. llama.cpp** — CPU / consumer-GPU / laptop deployment (see below).

### A. HuggingFace transformers (single-image debug)

The scripts use the official `HunYuanVLForConditionalGeneration` + `AutoProcessor`
integration shipped in transformers ≥ 4.57 (HunyuanOCR-1.5 series).

**Base model — one image:**

```bash
python inference/infer_base.py \
    --model /path/to/HunyuanOCR/base/model \
    --image /path/to/document.png \
    --max-new-tokens 8000
```

**Base model + DFlash draft — correctness / draft-load check on one image:**

```bash
python inference/infer_dflash.py \
    --model /path/to/HunyuanOCR/base/model \
    --dflash-model ./hyocr_dflash/ \
    --image /path/to/document.png \
    --num-spec-tokens 15
```

The default OCR prompt is:

```
提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，
表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。
```

Override with `--prompt "..."`. Both scripts print load time, generation latency
and the decoded text.

> ℹ️ `infer_dflash.py` only verifies that the DFlash draft checkpoint loads and
> produces a matching AR reference on the single image. Real speculative-decoding
> acceleration is only realized under vLLM (see below).

### B. vLLM production serving (OpenAI-compatible)

The launch scripts mirror the internal deployment: served alias
`tencent/HunyuanOCR-v2`, `-tp 1`, `--limit-mm-per-prompt '{"image":4,"video":0}'`,
`--trust_remote_code`, `--max-model-len 131072`.

**Autoregressive baseline** (HunyuanOCR without DFlash), single GPU:

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
GPU=0 PORT=8000 GPU_MEM_UTIL=0.9 \
bash inference/serve_ar.sh
```

**DFlash speculative decoding** (recommended when a DFlash draft is available):

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
DFLASH_PATH=./hyocr_dflash \
GPU=0 PORT=8001 GPU_MEM_UTIL=0.9 \
NUM_SPEC_TOKENS=15 \
bash inference/serve_dflash.sh
```

Both endpoints expose the standard vLLM OpenAI-compatible routes at
`http://<host>:<port>/v1/chat/completions`. Wait for readiness with:

```bash
curl -sf http://127.0.0.1:8000/v1/models
# or
tail -f vllm_ar_8000.log
```

**Minimal single-image client** — send one image via the shipped script:

```bash
python inference/infer_vllm_client.py \
    --host 127.0.0.1 --port 8000 \
    --model tencent/HunyuanOCR-v2 \
    --image /path/to/document.png
```

Or hand-written with the OpenAI SDK (mirrors what `infer_vllm_client.py` does):

```python
import base64, mimetypes
from openai import OpenAI

def data_url(p):
    mime = mimetypes.guess_type(p)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(open(p,'rb').read()).decode()}"

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")
resp = client.chat.completions.create(
    model="tencent/HunyuanOCR-v2",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url("/path/to/document.png")}},
            {"type": "text", "text": "请提取图片中的文字内容。"},
        ]},
    ],
    max_tokens=4096,
    temperature=0.0,
    top_p=1.0,
    extra_body={"top_k": -1, "repetition_penalty": 1.08, "skip_special_tokens": True},
)
print(resp.choices[0].message.content)
```

> ⚠️ For **multi-image** requests (>1 image per prompt), an extra vLLM shape-fix
> patch is required — this is unrelated to single-image OCR. See
> [`docs/inference.md`](docs/inference.md) if you plan to run multi-image benches.

### C. PC-side deployment via llama.cpp

For **CPU / consumer-GPU / laptop** environments, HunyuanOCR-1.5 can also be deployed through
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) after converting the checkpoint to GGUF.
Both the community `llama.cpp` (HunyuanOCR base only) and a DFlash-adapted fork
([`wendadawen/llama.cpp @ dflash-adapt-hunyuanocr-hunyuanstyle`](https://github.com/wendadawen/llama.cpp/tree/dflash-adapt-hunyuanocr-hunyuanstyle))
are supported.

Minimal build & serve (community, no DFlash):

```bash
# 1. Build
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON     # add -DGGML_CUDA=ON for NVIDIA GPU
cmake --build ./build --config Release -j

# 2. Convert HunyuanOCR to GGUF (base + mmproj)
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/hyocr-f16.gguf        --outtype f16 ./HunyuanOCR
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf --outtype f16 --mmproj ./HunyuanOCR

# 3. Serve (OpenAI-compatible)
build/bin/llama-server \
    --model  ./HunyuanOCR/hyocr-f16.gguf \
    --mmproj ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096
```

DFlash-adapted variant, weight conversion for the draft, and a smoke-test client
([`llama_cpp/chat.py`](llama_cpp/chat.py) with 26 sample OCR images under
[`llama_cpp/test_assets/`](llama_cpp/test_assets)):

see [`docs/llama_cpp.md`](docs/llama_cpp.md) for the complete guide.

---

## 📖 Documentation

- [`docs/training.md`](docs/training.md) — training modes, hyperparameters, distributed setup
- [`docs/data_format.md`](docs/data_format.md) — raw OCR JSONL schema and packing pipeline
- [`docs/inference.md`](docs/inference.md) — vLLM install (with DFlash patch) and deployment tuning
- [`docs/llama_cpp.md`](docs/llama_cpp.md) — PC-side deployment with llama.cpp (community & DFlash-adapted fork)
- [`docs/benchmark.md`](docs/benchmark.md) — end-to-end speed benchmark

---

## 📜 License

HunyuanOCR-1.5 is released under the same license as HunyuanOCR 1.0 —
the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
