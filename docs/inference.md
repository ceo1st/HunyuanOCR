# Inference & Deployment Guide

Three inference modes are supported:

1. **Transformers (Python, single image)** — for debugging
2. **vLLM serve (OpenAI-compatible)** — for production
3. **vLLM + DFlash** — vLLM serve with DFlash speculative decoding

---

## 1. Transformers Inference (single image)

### Base model

```bash
python inference/infer_base.py \
    --image /path/to/document.png \
    --model /path/to/HunyuanOCR/base
```

### With DFlash

```bash
python inference/infer_dflash.py \
    --image /path/to/document.png \
    --model /path/to/HunyuanOCR/base \
    --dflash-model ./hyocr_dflash/
```

**Note**: transformers mode is for debugging only. For real inference use vLLM below.

---

## 2. vLLM Serve — Base Model (AR baseline)

Standard vLLM OpenAI-compatible server, no DFlash.

### Prerequisites — install vLLM (tested recipe)

We use a **dedicated venv** for inference to keep vLLM nightly isolated. The
recipe below is what has been validated end-to-end for HunyuanOCR-1.5 (both AR
and DFlash):

```bash
# (optional) proxy — replace with yours if needed
# export http_proxy=http://your.proxy:3128
# export https_proxy=http://your.proxy:3128

uv venv /dockerdata/venv-vllm --python 3.12
source /dockerdata/venv-vllm/bin/activate

# vLLM nightly (cu130); ships DFlash speculative-decoding support
uv pip install -U vllm \
    --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly

# runai-model-streamer speeds up loading of large safetensors from HF/S3
uv pip install runai-model-streamer
```

> 💡 On CUDA 12.x, replace `--torch-backend=cu130` with the matching tag
> (e.g. `cu121`, `cu124`). Everything else stays the same.

Always `source /dockerdata/venv-vllm/bin/activate` before launching the serve
scripts below.

### Launch
```bash
MODEL_PATH=/path/to/HunyuanOCR/base \
PORT=8000 \
GPU=0 \
    bash inference/serve_ar.sh
```

Wait for `Application startup complete` in `ar_server_8000.log`, then test:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

### Client example (OpenAI SDK)

```python
import base64, mimetypes
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

with open("doc.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

r = client.chat.completions.create(
    model="/path/to/HunyuanOCR/base",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Extract all text from this document image as markdown."},
        ],
    }],
    temperature=0.0,
    extra_body={"top_k": 1},
)
print(r.choices[0].message.content)
```

---

## 3. vLLM Serve — With DFlash (recommended)

Enable DFlash speculative decoding for **~2.1× end-to-end speedup**.

### Prerequisites — vLLM nightly (DFlash included)

DFlash speculative decoding is bundled in the vLLM nightly wheel used above
(§2). If you already followed the §2 install, you're done — no extra patch
needed. Verify:

```bash
source /dockerdata/venv-vllm/bin/activate
python -c "
import vllm, inspect
from vllm.engine.arg_utils import SpeculativeConfig
src = inspect.getsource(SpeculativeConfig)
assert 'dflash' in src, 'DFlash not present in this vLLM build'
print('DFlash OK, vLLM:', vllm.__version__)
"
```

If the assert fails, you're on an older vLLM release; re-run the §2 nightly
install.

### Prepare the DFlash checkpoint directory

The vLLM DFlash config expects a directory with three files:

```
./hyocr_dflash/
├── config.json          # DFlash architecture config
├── dflash.py            # DFlash model class registration
└── model.safetensors    # Trained draft weights (346 MB)
```

- `config.json` + `dflash.py` are provided in this repo under `hyocr_dflash/`
- `model.safetensors` is the output of training (see `docs/training.md`)

Copy your trained checkpoint's `model.safetensors` (or the top-level one from `output/{run_name}/`) into `./hyocr_dflash/`:

```bash
cp output/hyocr_dflash_ft_lr2e-5_ep10_XXXX/model.safetensors ./hyocr_dflash/
```

**Important**: only use `model.safetensors` from the **top-level** `output/{run_name}/` (this is the final draft-only weights, ~350 MB). Do NOT use files inside `checkpoint-XXXX/` — those include the target model too (~2 GB) and won't load correctly.

### Launch

```bash
MODEL_PATH=/path/to/HunyuanOCR/base \
DFLASH_PATH=./hyocr_dflash \
PORT=8001 \
GPU=1 \
NUM_SPEC_TOKENS=15 \
    bash inference/serve_dflash.sh
```

Wait for `Application startup complete` in the log. During inference, you'll see periodic spec decoding metrics like:

```
SpecDecoding metrics: Mean acceptance length: 7.36, ...
Per-position acceptance rate: 0.87, 0.79, 0.71, ..., 0.15
Avg Draft acceptance rate: 42.4%
```

### Client — same as AR (drop-in compatible)

The DFlash server exposes the exact same `/v1/chat/completions` endpoint. **No client code changes needed** — you get the speedup for free.

---

## 4. Performance Tuning

### Key vLLM args

| Arg | Recommended | Notes |
|---|---|---|
| `--attention-backend` | `flash_attn` | Fastest for HunyuanOCR |
| `--no-enable-prefix-caching` | ✓ | Simplifies benchmarking; keep enabled for prod |
| `--mm-processor-cache-gb 0` | ✓ | Disables mm cache; more predictable |
| `--max-num-batched-tokens` | `16384` | Higher = better throughput, more mem |
| `--max-num-seqs` | `64` | Concurrent requests |
| `--gpu-memory-utilization` | `0.85` | Leave 15% for CUDA graph capture |

### DFlash-specific

| Arg | Recommended | Notes |
|---|---|---|
| `num_speculative_tokens` | `15` (default) | Larger = higher potential speedup, more overhead. Sweet spot: 8~15 |
| `method` | `dflash` | Must match model architecture |

### Tuning `num_speculative_tokens`

If your DFlash draft is well-trained (position-15 acceptance > 0.3), try `num_speculative_tokens=10` — might squeeze another 5-10% speedup by cutting off low-yield positions.

Check per-position acceptance in the server log:

```bash
grep "Per-position acceptance rate" dflash_server_*.log | tail -5
```

If position-15 rate < 0.15, reducing to 8~10 will help.

---

## 5. Benchmarking

See `docs/benchmark.md` for the full 8-way OCR speed comparison and reproduction instructions.

Quick single-image speed test:

```python
import time, base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8001/v1")
with open("doc.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

t = time.time()
r = client.chat.completions.create(
    model="/path/to/HunyuanOCR/base",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Extract all text..."},
        ],
    }],
    temperature=0.0,
    extra_body={"top_k": 1},
)
dt = time.time() - t
print(f"Latency: {dt:.2f}s | tokens: {r.usage.completion_tokens} | tok/s: {r.usage.completion_tokens/dt:.1f}")
```

Expected numbers on H20 (80GB):
- AR baseline: ~3s per page, ~460 tok/s
- DFlash v3: **~1.4s per page, ~1000 tok/s** ⚡
