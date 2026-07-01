<div align="center">

# HunyuanOCR-1.5: Towards Efficient and Effective E2E OCR

</div>

<p align="center">
 <img src="./assets/hyocr-head-img.png" width="80%"/> <br>
</p>

<p align="center">
<a href="https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh"><b>🎯 Online Demo</b></a> |
<a href="https://huggingface.co/tencent/HunyuanOCR"><b>📥 Model Download</b></a>
</p>

---

> ℹ️ This branch (`develop`) hosts the **HunyuanOCR-1.5** open-source training & inference toolkit.
> For the original HunyuanOCR 1.0 release, please switch to the `main` branch or refer to
> [`README_v1.0.md`](./README_v1.0.md) / [`README_zh_v1.0.md`](./README_zh_v1.0.md).

---

## 📖 Introduction

**HunyuanOCR-1.5** is the next iteration of Tencent's end-to-end OCR expert VLM, aiming at
**simultaneously higher accuracy and higher inference efficiency** than 1.0. Key upgrades include:

- 🧠 **Stronger E2E OCR quality** — improved document parsing, text spotting, information extraction,
  photo translation and video subtitle extraction, with a single-instruction / single-inference
  interface consistent with 1.0.
- ⚡ **DFlash speculative decoding** — an MTP-style draft head trained jointly / from-scratch on
  packed OCR sequences, delivering up to **2.1× end-to-end speedup** on real-world documents with
  no measurable output-quality loss (< 0.15% token diff vs AR).
- 📚 **Full open-source training pipeline** — SFT of the HunyuanOCR base model, DFlash draft
  training (from-scratch and continue-finetune), and an efficient token-count-then-pack data
  pipeline for large-scale packed sequence training.
- 🚀 **vLLM production deployment** — a single command deploys HunyuanOCR-1.5 (± DFlash) as an
  OpenAI-compatible chat/completions endpoint.

### 🔥 DFlash speedup at a glance

| Metric | HunyuanOCR base (AR) | HunyuanOCR + DFlash | Speedup |
|:--|--:|--:|--:|
| Avg latency / image | 3.03 s | **1.41 s** | **2.14×** |
| Token/s (end-to-end) | 466 | **1002** | 2.15× |
| Page/s | 0.33 | **0.71** | 2.14× |
| Output token diff vs AR | — | < 0.15% | ~lossless |

*Evaluated on 930 real-world document / PPT / book / textbook images at concurrency=1,
max_tokens=8000, on a single NVIDIA H20 (80GB). See [`docs/benchmark.md`](docs/benchmark.md)
for the full 8-way OCR comparison.*

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

Trains a small MTP-style draft that predicts K speculative tokens for HunyuanOCR.
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
Recommended profile (v3): `lr=2e-5`, `epochs=10`, `warmup_ratio=0.05`, `save_steps=500`.

> Empirical result on 14.7k packs: v3 continue-finetune beats v1 (1M packs from-scratch) on
> both acceptance rate (**42% vs 33%**) and end-to-end speedup (**2.14× vs 1.92×**).

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
> single image. **The real ~2.1× speedup is only realized under vLLM** (see below), because
> transformers has no CUDA-graph / batched speculative decoding kernel.

### B. vLLM production serving (OpenAI-compatible)

**Autoregressive baseline** (HunyuanOCR without DFlash):

```bash
MODEL_PATH=/path/to/HunyuanOCR/base/model \
PORT=8000 GPU=0 GPU_MEM_UTIL=0.85 \
MEDIA_PATH=/tmp \
bash inference/serve_ar.sh
```

**DFlash speculative decoding** (recommended, ~2.1× end-to-end speedup):

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
- [`docs/benchmark.md`](docs/benchmark.md) — full end-to-end speed benchmark

---

## 📜 License

HunyuanOCR-1.5 is released under the same license as HunyuanOCR 1.0 —
the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
