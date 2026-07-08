<div align="center">

[中文阅读](./README_zh.md)

</div>

<div align="center">

# HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better

</div>

<p align="center">
 <img src="./assets/hyocr-1.5-head-img.png" width="90%"/> <br>
</p>

<p align="center">
<a href="https://huggingface.co/tencent/HunyuanOCR"><b>🤗 Model</b></a> |
<a href="https://arxiv.org/pdf/2607.04884"><b>📄 Paper</b></a>
</p>

> [!NOTE]
> 👉 Looking for the original **HunyuanOCR 1.0** release? Switch to the
> [`v1.0`](https://github.com/Tencent-Hunyuan/HunyuanOCR/tree/v1.0) branch, or read
> [`README_v1.0.md`](./README_v1.0.md) · [`README_zh_v1.0.md`](./README_zh_v1.0.md).

---

## 🔥 News
- **[2026/07/07]** 🚀 We released **HunyuanOCR-1.5**, a systematic upgrade that makes lightweight end-to-end OCR **faster and better** via DFlash speculative decoding, PC-side llama.cpp deployment, an Agentic Data Flow, and an upgraded training recipe. Check out the [paper](https://arxiv.org/pdf/2607.04884).
- **[2026/06/18]** 🎉 Our work on table parsing has been accepted to the ECCV 2026 Main Conference! Check out the paper: [StrucTab: A Structured Optimization Framework for Table Parsing](https://arxiv.org/abs/2606.29905).
- **[2026/06/02]** 🎉 We have released two new benchmarks. [Chronicles-OCR](https://github.com/VirtualLUOUCAS/Chronicles-OCR) ([arXiv](https://arxiv.org/abs/2605.11960)), an open-source ancient-text perception benchmark covering the evolutionary trajectory of the "Seven Chinese Scripts", is jointly built by the **SSV Digital Culture Lab** and the **SSV Technical Architecture Department**, together with the **Palace Museum** and **Anyang Normal University**. We have also released [ChartArena](https://github.com/pspdada/ChartArena) ([arXiv](https://arxiv.org/abs/2606.01348)), a new chart-parsing benchmark supporting diverse chart types. Welcome to evaluate and provide your valuable feedback!
- **[2026/05/11]** 🎉 We have officially open-sourced two benchmarks on document parsing and text-image machine translation: [Wild-OmniDocBench](https://github.com/VirtualLUOUCAS/Wild_OmniDocBench) and [MMTIT-Bench](https://github.com/VirtualLUOUCAS/MMTIT_Bench). Welcome to evaluate and provide your valuable feedback!
- **[2026/04/08]** 🎉 Our works on document parsing and text-image machine translation have been accepted to the CVPR 2026 Main Conference! Check out the papers: [Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](https://arxiv.org/abs/2603.23885) and [MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation](https://arxiv.org/abs/2603.23896).
- **[2026/01/13]** ⭐ We have released a stable official [online demo](https://hunyuan.tencent.com/chat/HunyuanDefault?modelId=HY-OCR-1.0&mid=308&from=vision-zh), feel free to try it out!
- **[2025/11/28]** 🛠️ We fixed vLLM inference bugs and hyperparameter configuration issues such as system prompt. It is recommended to use the latest vLLM installation steps and the [inference script](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/Hunyuan-OCR-master/Hunyuan-OCR-vllm/run_hy_ocr.py) for performance testing. Currently, there is still a certain accuracy difference between Transformers and the vLLM framework (we are working on fixing this).
- **[2025/11/25]** 📝 Inference code and model weights publicly available.

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

### Training

- Python 3.10+ (3.12 tested)
- PyTorch 2.1+ (CUDA 12.1+; a cu130 build has been tested end-to-end)
- transformers 4.57+
- DeepSpeed 0.14+

```bash
pip install -r requirements.txt
# flash-attn requires a manual build:
pip install flash-attn --no-build-isolation
```

### Inference

Inference is split into **three self-contained, mutually exclusive setups** under
[`inference/`](inference). vLLM (AR / DFlash) and native transformers inference
require different, incompatible `transformers` versions and **cannot share one
environment** — this is a validated constraint, not a preference:

| Setup | vLLM | DFlash accel. | transformers | CUDA | Best for |
|---|:-:|:-:|:-:|---|---|
| [`inference/vllm_0_18_1`](inference/vllm_0_18_1) | 0.18.1 (release) | ❌ | ❌ | 12.x | simplest setup, AR only |
| [`inference/nightly`](inference/nightly) | nightly | ✅ | ❌ | 13 | AR + DFlash acceleration |
| [`inference/transformers`](inference/transformers) | — | — | ✅ 5.13.0 | host driver | native HF inference |

Each subfolder ships its own README and `requirements.txt`. See
[`inference/README.md`](inference/README.md) for the selection guide and the full
rationale, and [`docs/inference.md`](docs/inference.md) for performance tuning.

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

HunyuanOCR-1.5 provides three server/inference setups under [`inference/`](inference),
plus an optional PC-side path via llama.cpp. All three share the same weights and
the same task-type prompts + sampling + post-processing, so their outputs are
directly comparable.

- **A. vLLM 0.18.1 (release, CUDA 12) — AR only.** Simplest to install; native
  HunyuanOCR support, no nightly or patch. → [`inference/vllm_0_18_1`](inference/vllm_0_18_1)
- **B. vLLM nightly (CUDA 13) — AR + DFlash speculative decoding.** Lossless
  acceleration for long outputs; draft config/code bundled, weight pulled from HF. → [`inference/nightly`](inference/nightly)
- **C. HuggingFace transformers 5.13.0 — native multi-GPU inference.** For
  alignment / accuracy checks; no vLLM. → [`inference/transformers`](inference/transformers)
- **D. llama.cpp — CPU / consumer-GPU / laptop.** GGUF deployment (see below).

> ⚠️ Setups A / B / C are **mutually exclusive environments**: vLLM and native
> transformers require incompatible `transformers` versions. Read
> [`inference/README.md`](inference/README.md) before choosing.

### Download the weights

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

The download contains both the base model and the `dflash/` draft model.

### Quick start (vLLM AR, single GPU)

Install per [`inference/vllm_0_18_1/requirements.txt`](inference/vllm_0_18_1/requirements.txt),
then launch the OpenAI-compatible server (served as `tencent/HunyuanOCR`,
`-tp 1`, `--max-model-len 131072`):

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vllm_0_18_1/serve.sh
curl -sf http://127.0.0.1:8000/v1/models     # readiness check
```

Send a single image. The prompt is locked to an official task type via
`--task-type` (run `--list-tasks` to see all 12); sampling
(`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `repetition_penalty=1.08`) and
tail-repetition early-stop / cleanup are built in:

```bash
python inference/vllm_0_18_1/infer_vllm_client.py \
    --image /path/to/document.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

Batch inference over a directory (multi-endpoint concurrency, resumable):

```bash
python inference/vllm_0_18_1/batch_infer.py \
    --image-dir /path/to/images --out-dir /path/to/output \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

For **DFlash acceleration**, use [`inference/nightly`](inference/nightly)
(`serve_dflash.sh`); for **native transformers**, use
[`inference/transformers`](inference/transformers). Each subfolder README has the
full environment recipe, the task-type table, and multi-GPU instructions.

### PC-side deployment via llama.cpp

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
- [`docs/inference.md`](docs/inference.md) — vLLM install (nightly, DFlash included) and deployment tuning
- [`docs/llama_cpp.md`](docs/llama_cpp.md) — PC-side deployment with llama.cpp (community & DFlash-adapted fork)
- [`docs/benchmark.md`](docs/benchmark.md) — end-to-end speed benchmark

---

## 📚 Citation
```
@misc{li2026hunyuanocr15,
      title={HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better},
      author={Gengluo Li and Xingyu Wan and Shangpin Peng and Weinong Wang and Hao Feng and Yongkun Du and Binghong Wu and Zheng Ruan and Zhiqiong Lu and Liang Wu and Pengyuan Lyu and Huawen Shen and Zibin Lin and Shijing Hu and Jieneng Yang and Hongbing Wen and Guanghua Yu and Hong Liu and Bochao Wang and Can Ma and Han Hu and Chengquan Zhang and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2607.04884},
      url={https://arxiv.org/abs/2607.04884},
}

@misc{hunyuanvisionteam2025hunyuanocrtechnicalreport,
      title={HunyuanOCR Technical Report}, 
      author={Hunyuan Vision Team and Pengyuan Lyu and Xingyu Wan and Gengluo Li and Shangpin Peng and Weinong Wang and Liang Wu and Huawen Shen and Yu Zhou and Canhui Tang and Qi Yang and Qiming Peng and Bin Luo and Hower Yang and Xinsong Zhang and Jinnian Zhang and Houwen Peng and Hongming Yang and Senhao Xie and Longsha Zhou and Ge Pei and Binghong Wu and Kan Wu and Jieneng Yang and Bochao Wang and Kai Liu and Jianchen Zhu and Jie Jiang and Linus and Han Hu and Chengquan Zhang},
      year={2025},
      journal={arXiv preprint arXiv:2511.19575},
      url={https://arxiv.org/abs/2511.19575}, 
}

@misc{li2026mmtitbench,
      title={MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation},
      author={Gengluo Li and Chengquan Zhang and Yupu Liang and Huawen Shen and Yaping Zhang and Pengyuan Lyu and Weinong Wang and Xingyu Wan and Gangyan Zeng and Han Hu and Can Ma and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2603.23896},
      url={https://arxiv.org/abs/2603.23896},
}

@misc{li2026towardsrealworlddocument,
      title={Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training},
      author={Gengluo Li and Pengyuan Lyu and Chengquan Zhang and Huawen Shen and Liang Wu and Xingyu Wan and Gangyan Zeng and Han Hu and Can Ma and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2603.23885},
      url={https://arxiv.org/abs/2603.23885},
}

@misc{li2026chronicles,
      title={Chronicles-OCR: A Cross-Temporal Perception Benchmark for the Evolutionary Trajectory of Chinese Characters},
      author={Gengluo Li and Shangping Peng and Xingyu Wan and Chengquan Zhang and Hao Feng and Xin Xu and Pian Wu and Bang Li and Zengmao Ding and Yongge Liu and Yipei Ye and Yang Yang and Zhan Shu and Guojun Yan and Zhe Li and Can Ma and Weiping Wang and Yu Zhou and Han Hu},
      year={2026},
      journal={arXiv preprint arXiv:2605.11960},
      url={https://arxiv.org/abs/2605.11960},
}

@misc{peng2026chartarena,
      title={ChartArena: Benchmarking Chart Parsing across Languages, Scenarios, and Formats},
      author={Shangpin Peng and Gengluo Li and Xingyu Wan and Chengquan Zhang and Hao Feng and Binghong Wu and Huawen Shen and Weinong Wang and Ziyi Cai and Zhuotao Tian and Han Hu and Can Ma and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2606.01348},
      url={https://arxiv.org/abs/2606.01348},
}

@misc{li2026structab,
      title={StrucTab: A Structured Optimization Framework for Table Parsing},
      author={Gengluo Li and Shangpin Peng and Chengquan Zhang and Binghong Wu and Hao Feng and Weinong Wang and Pengyuan Lyu and Huawen Shen and Xingyu Wan and Zhuotao Tian and Han Hu and Can Ma and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2606.29905},
      url={https://arxiv.org/abs/2606.29905},
}
```

---

## 🙏 Acknowledgements

We would like to thank [Qwen](https://github.com/QwenLM/Qwen3.6) and [DFlash](https://github.com/z-lab/dflash) for their valuable models and ideas.

Special thanks to the Hugging Face community for their Day-0 support.

---

## 📜 License

HunyuanOCR-1.5 is released under the same license as HunyuanOCR 1.0 —
the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
