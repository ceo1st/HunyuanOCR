"""
Minimal OpenAI-compatible client for a locally deployed HunyuanOCR-1.5 vLLM server.

Sends **one image + one prompt** to the server and prints the response and
per-request latency. Use this for quick smoke-testing after `serve_ar.sh` /
`serve_dflash.sh`.

Usage:
    # 1. start the server (in another terminal)
    bash inference/serve_ar.sh

    # 2. send a single-image inference request
    python inference/infer_vllm_client.py \
        --image /path/to/document.png

Optional flags:
    --host / --port           default 127.0.0.1:8000
    --model                   default 'tencent/HunyuanOCR-v2' (must match --served-model-name)
    --prompt                  default OCR-to-markdown prompt (Chinese)
    --max-tokens              default 4096
    --repetition-penalty      default 1.08
"""
import argparse
import base64
import mimetypes
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，表格用html格式表达，"
    "文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def encode_image_as_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="tencent/HunyuanOCR-v2",
                    help="Must match vLLM --served-model-name")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
        timeout=args.timeout,
    )

    image_url = encode_image_as_data_url(args.image)
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    print(f"[info] POST http://{args.host}:{args.port}/v1/chat/completions "
          f"(model={args.model})")
    t = time.time()
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body={
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "skip_special_tokens": True,
        },
    )
    dt = time.time() - t

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)

    print("=" * 60)
    print(f"Latency          : {dt:.2f}s")
    if usage is not None:
        print(f"Prompt tokens    : {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        if usage.completion_tokens:
            print(f"tok/s            : {usage.completion_tokens / dt:.1f}")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
