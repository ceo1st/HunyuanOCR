# 速度基准

[English Version](./benchmark.md)

## 测试环境

- **硬件**：单张 NVIDIA H20（80 GB），每台服务器一张
- **数据集**：OmnidocBench，包含文档 / PPT / 图书 / 教科书图片
- **配置**：并发 = 1（单请求延迟），`max_tokens=8000`，`temperature=0.0`
- **指标**：从 HTTP 请求到完整响应的端到端延迟

## HunyuanOCR AR vs DFlash

核心对比：同一基座模型，不同推理路径。

| 配置                                       |    wall time | avg latency |    page/s |      加速 |
| ------------------------------------------ | -----------: | ----------: | --------: | --------: |
| **HunyuanOCR 基座（AR）**                  |     2821.9 s |     3.032 s |     0.330 |     1.00× |
| **HunyuanOCR + DFlash（14.7k pack 微调）** | **1316.5 s** | **1.408 s** | **0.706** | **2.14×** |

**关键结论：**

- DFlash 相对 AR 基线取得**端到端 2.14× 加速**
- 投机解码基本无损：total-tokens 与 AR 的差异 < 0.15%

## 跨模型对比

在相同评测条件下（c=1、`max_tokens=8000`），对比其它开源 OCR VLM：

| 排名 | 模型                      | 延迟（s/图） |    Page/s | 相对 HunyuanOCR AR | 备注                    |
| :--: | ------------------------- | -----------: | --------: | -----------------: | ----------------------- |
|  🥇  | **HunyuanOCR + DFlash**   |     **1.41** | **0.706** |          **2.14×** | 投机解码                |
|  🥈  | GLM-OCR（SDK 页面流水线） |         1.65 |     0.604 |              1.83× | Layout + 区域 OCR 并发  |
|  🥉  | PaddleOCR-VL 1.6（0.9B）  |         1.74 |     0.562 |              1.71× | 0.9B 小模型 + 两阶段    |
|  4   | HunyuanOCR 基座（AR）     |         3.03 |     0.330 |              1.00× | 基线                    |
|  5   | Unlimited-OCR（SGLang）   |         3.66 |     0.255 |              0.77× | 图像切片（gundam 模式） |
|  6   | DeepSeek-OCR-2            |         5.46 |     0.179 |              0.54× | 大模型 + grounding      |
|  7   | dots.ocr                  |         7.15 |     0.136 |              0.41× | 慢但结构化              |

## 复现数据

1. 部署所需的推理服务（详见 `docs/inference/inference_zh.md`）
2. 运行一个兼容的基准脚本。最小示例：

```python
import time, base64, glob, json, os
from openai import OpenAI

# ← 配置以下变量
IMAGE_DIR = "/path/to/eval/images"
SAMPLE_LIST = "./speed_eval_set_930.txt"
SERVER_URL = "http://127.0.0.1:8001/v1"
MODEL_NAME = "/path/to/HunyuanOCR/base"
PROMPT = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"

# 收集图片
want = set(l.strip() for l in open(SAMPLE_LIST))
imgs = sorted(p for p in glob.glob(f"{IMAGE_DIR}/*") if os.path.basename(p) in want)

# 提前 base64 编码（避免 I/O 干扰计时）
items = []
for p in imgs:
    with open(p, "rb") as f:
        items.append(base64.b64encode(f.read()).decode())

client = OpenAI(api_key="EMPTY", base_url=SERVER_URL)
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)

t0 = time.time()
latencies, tokens = [], []
for b64 in items:
    t = time.time()
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        temperature=0.0, max_tokens=8000, extra_body={"top_k": 1},
    )
    latencies.append(time.time() - t)
    tokens.append(r.usage.completion_tokens)

wall = time.time() - t0
n = len(latencies)
print(json.dumps({
    "n": n,
    "wall_s": round(wall, 3),
    "avg_latency_s": round(sum(latencies)/n, 3),
    "page_per_s": round(n/wall, 4),
    "token_per_s": round(sum(tokens)/wall, 2),
}))
```

3. 对比 `page_per_s`：这是真正跨模型公平的指标。

## 公平性说明

- **公平**：`wall time`、`avg_latency`、`page_per_s`，跨模型可比
- **不公平**：`token_per_s`、`total_tokens`，不同分词器计数方式不同
- 各模型使用各自原生的 prompt（强行统一 prompt 会破坏输出质量）
- 所有模型均在同一 H20（80GB）硬件上测试
