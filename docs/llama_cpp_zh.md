# 使用 llama.cpp 的 PC 端部署

[English Version](./llama_cpp.md)

HunyuanOCR-1.5 可通过 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 在 **CPU / 消费级 GPU / 笔记本** 上部署：把基座模型（可选还有 DFlash 草稿）转换为 GGUF 格式，然后用 OpenAI 兼容的 `llama-server` 提供服务。

支持两个版本：

| 版本                    | 仓库                                                                                                                                                     | 使用场景                                     |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------- |
| **社区版（无 DFlash）** | 上游 `ggml-org/llama.cpp`（main 分支）                                                                                                                   | 只需要 HunyuanOCR 基座，最简单最稳定。       |
| **DFlash 适配版**       | fork：[`wendadawen/llama.cpp @ dflash-adapt-hunyuanocr-hunyuanstyle`](https://github.com/wendadawen/llama.cpp/tree/dflash-adapt-hunyuanocr-hunyuanstyle) | 需要在 PC 上用 DFlash 投机解码做端到端加速。 |

> ⚠️ 上游 `llama.cpp` 对投机解码的支持有限，DFlash **尚未**合并进去，且仍有已知 bug。上面这个 fork 是我们针对 HunyuanOCR 的 DFlash 移植，**不是**社区版。

---

## 1. 社区版（HunyuanOCR 基座，不含 DFlash）

### 1.1 克隆并编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 如果有 NVIDIA GPU 并希望 CUDA 加速，追加 -DGGML_CUDA=ON
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

### 1.2 为权重转换准备 Python 环境

```bash
uv venv --python 3.12 venv-llamacpp
source venv-llamacpp/bin/activate
uv pip install huggingface_hub transformers torch openai
```

### 1.3 下载 HunyuanOCR 权重并转换为 GGUF

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"

# 语言 / 解码器权重 → hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/hyocr-f16.gguf \
    --outtype f16 \
    ./HunyuanOCR

# 视觉（mmproj）权重 → mmproj-hyocr-f16.gguf
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --outtype f16 \
    --mmproj \
    ./HunyuanOCR
```

### 1.4 启动 OpenAI 兼容服务

```bash
build/bin/llama-server \
    --model  "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096
```

服务端点为 `http://<host>:8080/v1/chat/completions`，别名 `HYVL`。

---

## 2. DFlash 适配版（HunyuanOCR + DFlash 投机解码）

### 2.1 克隆并编译 DFlash 分支

```bash
git clone -b dflash-adapt-hunyuanocr-hunyuanstyle \
    https://github.com/wendadawen/llama.cpp.git
cd llama.cpp

cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build ./build --config Release -j
```

权重下载与基座 / mmproj 的 GGUF 转换与社区版一致，参见 1.2 和 1.3 节。

### 2.2 把 DFlash 草稿权重转换为 GGUF

`--target-model-dir` 指向 HunyuanOCR 基座的 HF 检查点（用于 tokenizer / config），位置参数指向 DFlash 检查点目录。

```bash
python3 convert_hf_to_gguf.py \
    --outfile ./HunyuanOCR-Dflash/hyocr-dflash-bf16.gguf \
    --outtype bf16 \
    --target-model-dir ./HunyuanOCR \
    ./HunyuanOCR-Dflash
```

### 2.3 启动带 DFlash 的 OpenAI 兼容服务

```bash
build/bin/llama-server \
    --model       "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj      "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --model-draft "./HunyuanOCR-Dflash/hyocr-dflash-bf16.gguf" \
    --dflash --draft-max 16 \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    --parallel 1 \
    --ubatch-size 8192 \
    --batch-size  8192
```

DFlash 相关关键参数：

| 参数                                | 含义                                     |
| :---------------------------------- | :--------------------------------------- |
| `--model-draft <path>`              | DFlash 草稿模型的 GGUF 路径              |
| `--dflash`                          | 启用 DFlash 风格的投机解码               |
| `--draft-max 16`                    | 每个投机步的草稿 token 数（K）           |
| `--parallel 1`                      | 单串行 slot（在 PC 上跑 DFlash 时推荐）  |
| `--ubatch-size / --batch-size 8192` | 大 batch，让目标模型在验证阶段有充足吞吐 |

---

## 3. 快速验证

我们在 [`llama_cpp/`](../llama_cpp) 下附带一个最小的 OpenAI 兼容客户端和 26 张 OCR 测试图片，用于端到端冒烟测试。

### 3.1 安装客户端依赖

```bash
pip install openai
```

### 3.2 运行

```bash
cd llama_cpp
python chat.py
```

`chat.py` 默认连接 `http://127.0.0.1:8080/v1`、别名 `HYVL`（与上面 `llama-server` 的启动命令匹配），读取 `test_assets/data.jsonl`，把第一个 `ocr` 样本发到服务器，打印响应与每条耗时，并把所有内容 tee 到 `logs/chat_<timestamp>.log`。

在 `chat.py` 顶部可调整客户端行为：

```python
BASE_URL     = "http://127.0.0.1:8080/v1"
MODEL        = "HYVL"
MAX_REQUESTS = 10                # 总请求上限
TYPE_LIMITS  = {"ocr": 1}        # 按类型限制；设为 None 关闭
```

### 3.3 示例输出

```
=== [ocr] ocr/0.png ===
Prompt: 请提取文档图片中正文的所有信息用 markdown 格式表示。
ring, and Jacobson semisimple, by Corollary 8.35(ii)]. The factor module
$ J^{q}/J^{q+1} $ is an $ (R/J) $-module; hence, by Corollary 8.43,
$ J^{q}/J^{q+1} $ is a semisimple module, and so it can be decomposed into a
direct sum of (possibly infinitely many) simple $ (R/J) $-modules. ...
...
[elapsed] 6.885s

[total] 1 items (ocr=1), elapsed: 6.885s
```

看到响应被流式返回，末尾出现 `[total]` 行，说明 llama.cpp 部署（含 / 不含 DFlash）工作正常。
