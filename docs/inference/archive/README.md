# HunyuanOCR-1.5 — Legacy Inference Setups (Archived)

[中文阅读](./README_zh.md)

> ⚠️ **Archived.** This document describes the **old three-environment layout**,
> where vLLM AR, DFlash, and native transformers each needed a _separate,
> mutually exclusive_ environment. It is kept for reference and reproducibility.
> For the current unified single-environment setup, see
> [`docs/inference/inference.md`](../inference.md).

The three setups below share the same weights and the same sampling /
post-processing pipeline (task-type prompts + `repetition_penalty=1.08` +
tail-repetition early-stop + markdown normalization), so their outputs are
directly comparable.

---

## Which one should I use?

| Your situation                                   | Use                                 | vLLM AR | DFlash accel. | transformers |
| ------------------------------------------------ | ----------------------------------- | :-----: | :-----------: | :----------: |
| CUDA 12, want the simplest setup                 | [`vLLM`](./vLLM.md)                 |   ✅    |      ❌       |      ❌      |
| Want DFlash speculative decoding (needs CUDA 13) | [`DFlash`](./DFlash.md)             |   ✅    |      ✅       |      ❌      |
| Want native HuggingFace transformers inference   | [`transformers`](./transformers.md) |    —    |       —       |      ✅      |

Each guide ships the full validated environment recipe and usage. The code for
each setup lives under [`inference/`](../../../inference) in
`inference/vLLM/`, `inference/DFlash/`, and `inference/transformers/`.

---

## Why three separate environments?

The key constraints are mutually exclusive, so a single environment (in the old
layout) could not satisfy all of them. This is a validated conclusion, not a
preference:

|                   | `vLLM`                 | `DFlash`                      | `transformers`                    |
| ----------------- | ---------------------- | ----------------------------- | --------------------------------- |
| vLLM              | **0.18.1** (release)   | **nightly** (0.23.1rc1)       | not used                          |
| transformers      | 4.57.6                 | 5.5.3 (must pin)              | **5.13.0**                        |
| CUDA              | 12.x (native, one pip) | 13 (torch cu130 + compat lib) | matches host driver (cu128/cu130) |
| Python            | 3.10                   | 3.12                          | 3.12                              |
| Runs AR           | ✅                     | ✅                            | ✅ (HF generate)                  |
| Runs DFlash       | ❌                     | ✅                            | ❌                                |
| Runs transformers | ❌                     | ❌                            | ✅                                |

1. **DFlash was registered only in vLLM nightly (cu130)** → DFlash required
   nightly + CUDA 13; the 0.18.1 release had no `dflash` method.
2. **Native transformers inference needs the `hunyuan_vl` module → transformers
   ≥ 5.13.0**. vLLM 0.18.1 caps `transformers < 5`, and the nightly path had to
   pin transformers back to 5.5.3 — neither ships `hunyuan_vl`.
3. **transformers 5.13.0 breaks vLLM's HunYuanVL loading**
   (`AutoImageProcessor.register` is called with a string; the 5.13 signature
   change raises `AttributeError`). Installing vLLM nightly pulls transformers
   5.13.0 by default, so the nightly setup had to explicitly downgrade to 5.5.3;
   conversely, a 5.13.0 environment could only do transformers, not vLLM.

**Bottom line (old layout):** vLLM inference (AR / DFlash) and native
transformers inference required two different transformers versions and could
not coexist in one environment. For a three-way comparison, run both vLLM paths
from [`DFlash`](./DFlash.md) and transformers from
[`transformers`](./transformers.md) — same kernel, same weights, same sampling,
so the comparison is methodologically sound.

---

## Common steps (identical across all three)

**Download the model weights** (each guide repeats this):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The download contains both the **base model** and the **`dflash/` draft model**.

**Task types / prompts:** every setup selects the official recommended prompt via
`--task-type` (free-form prompt rewriting is intentionally not exposed, to avoid
degrading quality). There are 12 task types — see the "Task types" section in any
guide.

---

## Performance tuning

The following notes apply on top of the setups above.

### 1. vLLM performance tuning

Key server arguments (set in the `serve*.sh` scripts; override via the documented
environment variables):

| Arg                        | Recommended             | Notes                                                              |
| -------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `--gpu-memory-utilization` | `0.85`                  | Leave headroom for CUDA graph capture; DFlash's draft adds ~0.7 GB |
| `--max-model-len`          | `131072`                | Context window; lower it to save memory if your inputs are short   |
| `--max-num-batched-tokens` | `131072`                | Higher = better throughput, more memory                            |
| `--limit-mm-per-prompt`    | `{"image":4,"video":0}` | Image-only model; video disabled                                   |
| `--trust-remote-code`      | ✓                       | Required to load the HunyuanOCR model code                         |

**Multi-GPU throughput.** vLLM here uses `-tp 1` (one replica per GPU). For
higher throughput, launch one instance per GPU on separate ports
(`GPU=0 PORT=8000`, `GPU=1 PORT=8001`, …) and pass all ports to
`batch_infer.py --ports 8000,8001,...` — the batch client round-robins requests
across endpoints for near-linear scaling.

### 2. DFlash tuning (DFlash setup only)

DFlash speculative decoding is lossless (it preserves the target model's output
distribution) and accelerates long structured outputs the most. The single knob
is `NUM_SPEC_TOKENS` (default 15, the official recommendation):

| `NUM_SPEC_TOKENS`   | Effect                                                         |
| ------------------- | -------------------------------------------------------------- |
| larger (e.g. 15)    | higher potential speedup, more per-step overhead               |
| smaller (e.g. 8–10) | less overhead; can win 5–10% when late positions rarely accept |

Inspect per-position acceptance in the server log to decide:

```bash
grep "Per-position acceptance rate" vllm_dflash_*.log | tail -5
# also look for lines like:
#   SpecDecoding metrics: Mean acceptance length: 7.36, ...
#   Avg Draft acceptance rate: 42.4%
```

If the position-15 acceptance rate is < 0.15, reducing `NUM_SPEC_TOKENS` to 8–10
usually helps; if it stays high (> 0.3), keep 15.

The DFlash server exposes the same OpenAI-compatible `/v1/chat/completions`
endpoint as the AR server, so **no client change is needed** — the clients in all
three setups are shared verbatim.

### 3. Advanced: multi-image requests

The shipped clients target **single-image** OCR (the validated path). For
requests with more than one image per prompt under vLLM, an extra vLLM shape-fix
patch may be required depending on your build; this is unrelated to single-image
OCR. Keep prompts to one image per request unless you specifically need
multi-image benches.

### 4. Benchmarking

See [`docs/benchmark.md`](../../benchmark.md) for the full speed comparison
(AR vs DFlash and cross-model) and a minimal reproduction script.
