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
extraction, text-image translation, and multi-image document understanding** within a single
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

Two paths are provided: **HuggingFace transformers** (single-image, easy to hack, recommended for
debugging) and **vLLM** (production serving, OpenAI-compatible, required for real DFlash speedup).

### A. HuggingFace transformers (single-image debug)

**Base model:**

```bash
python inference/infer_base.py \
    --model /path/to/HunyuanOCR/base/model \
    --image /path/to/document.png \
    --max-new-tokens 8000
```

**Base model + DFlash draft (correctness check):**

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

Override with `--prompt "..."`. Both scripts print latency, completion tokens and tok/s.

> ℹ️ `infer_dflash.py` is designed for correctness verification of a DFlash checkpoint on a
> single image. Real end-to-end speedup is only realized under vLLM (see below), because
> transformers has no CUDA-graph / batched speculative-decoding kernel.

### B. vLLM production serving (OpenAI-compatible)

**Autoregressive baseline** (HunyuanOCR without DFlash):

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
PORT=8000 GPU=0 GPU_MEM_UTIL=0.85 \
MEDIA_PATH=/tmp \
bash inference/serve_ar.sh
```

**DFlash speculative decoding** (recommended):

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
DFLASH_PATH=./hyocr_dflash \
PORT=8001 GPU=1 GPU_MEM_UTIL=0.85 \
NUM_SPEC_TOKENS=15 \
MEDIA_PATH=/tmp \
bash inference/serve_dflash.sh
```

Both endpoints expose the standard vLLM OpenAI-compatible routes at
`http://<host>:<port>/v1/chat/completions`. Wait for `Application startup complete` in the log
before sending requests:

```bash
tail -f dflash_server_8001.log
```

Minimal client example (Python):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="/path/to/HunyuanOCR/base/model",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///tmp/doc.png"}},
            {"type": "text", "text": "Extract all text as markdown."},
        ],
    }],
    max_tokens=8000,
    temperature=0.0,
)
print(resp.choices[0].message.content)
```

**DFlash-specific vLLM knobs** (already set inside `serve_dflash.sh`):

| Flag | Meaning |
|:--|:--|
| `--speculative-config '{"method":"dflash","model":"./hyocr_dflash","num_speculative_tokens":15}'` | Enables DFlash speculative decoding with the given draft dir and K |
| `--attention-backend flash_attn` | Required attention backend |
| `--no-enable-prefix-caching` | Disable prefix cache (currently incompatible with DFlash) |
| `--mm-processor-cache-gb 0` | Disable multimodal processor cache to avoid OOM |
| `--max-num-batched-tokens 16384`, `--max-num-seqs 64` | Recommended batching profile |

Full deployment / tuning guide: [`docs/inference.md`](docs/inference.md).

---

## 📖 Documentation

- [`docs/training.md`](docs/training.md) — training modes, hyperparameters, distributed setup
- [`docs/data_format.md`](docs/data_format.md) — raw OCR JSONL schema and packing pipeline
- [`docs/inference.md`](docs/inference.md) — vLLM install (with DFlash patch) and deployment tuning
- [`docs/benchmark.md`](docs/benchmark.md) — end-to-end speed benchmark

---

## 📜 License

HunyuanOCR-1.5 is released under the same license as HunyuanOCR 1.0 —
the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
