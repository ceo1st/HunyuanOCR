# HunyuanOCR-1.5 旧版推理环境（归档）

[English Version](./README.md)

> ⚠️ **已归档。** 本文档描述**旧的三环境布局**：vLLM AR、DFlash、原生 transformers 各自需要一套独立、互斥的环境。此处保留仅供参考与复现用。当前的统一单环境方案，请见 [`docs/inference/inference_zh.md`](../inference_zh.md)。

下文的三套环境共享同一份权重，也共享同一套采样 / 后处理流水线（task-type prompt + `repetition_penalty=1.08` + 尾部重复早停 + markdown 规整），所以三者输出可以直接对比。

---

## 我该选哪套？

| 你的场景                               | 使用                                   | vLLM AR | DFlash 加速 | transformers |
| -------------------------------------- | -------------------------------------- | :-----: | :---------: | :----------: |
| CUDA 12，想要最省心的方案              | [`vLLM`](./vLLM_zh.md)                 |   ✅    |     ❌      |      ❌      |
| 需要 DFlash 投机解码（要求 CUDA 13）   | [`DFlash`](./DFlash_zh.md)             |   ✅    |     ✅      |      ❌      |
| 需要原生 HuggingFace transformers 推理 | [`transformers`](./transformers_zh.md) |    —    |      —      |      ✅      |

每份指南都提供完整的已验证环境安装与使用步骤。各环境的代码位于 [`inference/`](../../../inference) 下的 `inference/vLLM/`、`inference/DFlash/`、`inference/transformers/`。

---

## 为什么要分成三套环境？

关键依赖之间互斥，因此在旧布局下无法用同一套环境同时满足三种需求。这是实测验证的结论，不是个人偏好：

|                   | `vLLM`                 | `DFlash`                       | `transformers`              |
| ----------------- | ---------------------- | ------------------------------ | --------------------------- |
| vLLM              | **0.18.1**（正式版）   | **nightly**（0.23.1rc1）       | 不使用                      |
| transformers      | 4.57.6                 | 5.5.3（必须锁定）              | **5.13.0**                  |
| CUDA              | 12.x（原生，一条 pip） | 13（torch cu130 + compat lib） | 匹配宿主驱动（cu128/cu130） |
| Python            | 3.10                   | 3.12                           | 3.12                        |
| 支持 AR           | ✅                     | ✅                             | ✅（HF generate）           |
| 支持 DFlash       | ❌                     | ✅                             | ❌                          |
| 支持 transformers | ❌                     | ❌                             | ✅                          |

1. **DFlash 只在 vLLM nightly（cu130）里注册过**：所以 DFlash 需要 nightly + CUDA 13；0.18.1 正式版没有 `dflash` 方法。
2. **原生 transformers 推理需要 `hunyuan_vl` 模块，因此 transformers ≥ 5.13.0**。而 vLLM 0.18.1 要求 `transformers < 5`，nightly 路径又必须把 transformers 锁回 5.5.3，这两版本都不带 `hunyuan_vl`。
3. **transformers 5.13.0 会破坏 vLLM 的 HunYuanVL 加载**（`AutoImageProcessor.register` 被字符串参数调用，5.13 签名变化导致 `AttributeError`）。安装 vLLM nightly 默认拉进 transformers 5.13.0，因此 nightly 路径必须显式降级到 5.5.3；反过来，5.13.0 的环境只能做 transformers，不能做 vLLM。

**结论（旧布局）：** vLLM 推理（AR / DFlash）与原生 transformers 推理需要两个不兼容的 transformers 版本，无法共存于同一环境。若要做三方对比，从 [`DFlash`](./DFlash_zh.md) 跑 vLLM 的两条路径，从 [`transformers`](./transformers_zh.md) 跑 transformers 推理，三者内核、权重、采样都一致，对比方法学上是成立的。

---

## 通用步骤（三套通用）

**下载模型权重**（每份指南也都会重复给出）：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

下载内容同时包含**基座模型**与 **`dflash/` 草稿模型**。

**任务类型 / 提示词：** 每套环境都通过 `--task-type` 选择官方推荐 prompt（未开放自由改写 prompt，以避免降低质量）。共 12 种任务类型，详见任一指南的 "Task types" 章节。

---

## 性能调优

以下补充说明适用于上述所有环境。

### 1. vLLM 性能调优

关键 server 参数（在 `serve*.sh` 里设置，可以通过对应环境变量覆盖）：

| 参数                       | 推荐值                  | 说明                                                           |
| -------------------------- | ----------------------- | -------------------------------------------------------------- |
| `--gpu-memory-utilization` | `0.85`                  | 给 CUDA graph capture 留出空间；DFlash 的草稿额外占用约 0.7 GB |
| `--max-model-len`          | `131072`                | 上下文长度；输入较短时可以调低以省显存                         |
| `--max-num-batched-tokens` | `131072`                | 越大吞吐越高，也越吃显存                                       |
| `--limit-mm-per-prompt`    | `{"image":4,"video":0}` | 纯图片模型；视频禁用                                           |
| `--trust-remote-code`      | ✓                       | 加载 HunyuanOCR 模型代码所必需                                 |

**多卡吞吐**：这里 vLLM 用的是 `-tp 1`（每卡一个实例）。要更高吞吐，就在每张卡上分别启动一个实例，用不同端口（`GPU=0 PORT=8000`、`GPU=1 PORT=8001`、……），然后把所有端口传给 `batch_infer.py --ports 8000,8001,...`。批量客户端会在多端点间轮询请求，接近线性扩展。

### 2. DFlash 调优（仅 DFlash 环境）

DFlash 投机解码是无损的（保留目标模型的输出分布），对长结构化输出加速最明显。唯一的调节旋钮是 `NUM_SPEC_TOKENS`（默认 15，官方推荐值）：

| `NUM_SPEC_TOKENS` | 效果                                               |
| ----------------- | -------------------------------------------------- |
| 较大（如 15）     | 潜在加速上限更高，但每步开销更大                   |
| 较小（如 8–10）   | 每步开销更低；当靠后位置几乎不被接受时可再快 5–10% |

查看每个位置的接受率，据此调整：

```bash
grep "Per-position acceptance rate" vllm_dflash_*.log | tail -5
# 还可以看类似这样的日志行：
#   SpecDecoding metrics: Mean acceptance length: 7.36, ...
#   Avg Draft acceptance rate: 42.4%
```

若位置 15 的接受率 < 0.15，把 `NUM_SPEC_TOKENS` 调到 8–10 通常有帮助；若始终 > 0.3，就保持 15。

DFlash server 与 AR server 暴露同一个 OpenAI 兼容的 `/v1/chat/completions` 端点，因此**客户端不需要改**，三套环境的客户端完全通用。

### 3. 进阶：多图请求

出货的客户端针对**单图** OCR 场景（已验证的路径）。若在 vLLM 下要发送单个 prompt 含多张图片的请求，根据构建版本可能需要一个额外的 vLLM shape 补丁，这与单图 OCR 无关。除非你专门要做多图评测，否则保持每个请求一张图片。

### 4. 基准测试

端到端速度对比（AR vs DFlash 及跨模型）与最小复现脚本见 [`docs/benchmark_zh.md`](../../benchmark_zh.md)。
