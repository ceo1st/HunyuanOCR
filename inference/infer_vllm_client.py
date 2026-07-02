"""
Minimal OpenAI-compatible client for a locally deployed HunyuanOCR-1.5 vLLM server.

Sends **one image + one prompt** to the server and prints the response and
per-request latency. Aligned with the production `vllm/infer_vllm_8gpu.py`
(sampling params, streaming + tail-repetition early stop, tail-repeat cleanup)
so results match what the internal bench pipeline produces on the same server
started by `serve_ar.sh` / `serve_dflash.sh`.

Usage:
    # 1. start the server (in another terminal)
    bash inference/serve_ar.sh          # port 8000, AR baseline
    # or
    bash inference/serve_dflash.sh      # port 8001, DFlash speculative

    # 2. send a single-image inference request
    python inference/infer_vllm_client.py \
        --image /path/to/document.png \
        --port 8000

Optional flags:
    --host / --port           default 127.0.0.1:8000
--model                   default 'tencent/HunyuanOCR-1-5' (must match --served-model-name)
    --prompt                  default OCR-to-markdown prompt (Chinese)
    --max-tokens              default 4096
    --repetition-penalty      default 1.08
    --no-stream               disable streaming + early-stop (one-shot generation)
"""
import argparse
import base64
import sys
import time
from pathlib import Path
from typing import List


DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，表格用html格式表达，"
    "文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


# ---------- tail-repetition helpers (mirror vllm/infer_vllm_8gpu.py) ----------
def has_tail_repetition(text: str, min_repeats: int = 8, max_unit: int = 256) -> bool:
    """Detect if the tail of `text` is stuck in a small repeated unit."""
    n = len(text)
    if n < min_repeats * 2:
        return False
    upper = min(max_unit, n // min_repeats)
    for length in range(1, upper + 1):
        unit = text[-length:]
        if not unit.strip():
            continue
        ok = True
        for k in range(2, min_repeats + 1):
            if text[-length * k:-length * (k - 1)] != unit:
                ok = False
                break
        if ok:
            return True
    return False


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim long repeated suffixes as a final safety net."""
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // min_repeats + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= min_repeats:
            return text[: n - length * (count - 1)]
    return text


# ---------- image encoding ----------
def encode_image_as_data_url(path: str) -> str:
    """Read image → base64 data URL. Mime is fixed to `image/jpeg` to match the
    production `vllm/infer_vllm_8gpu.py` behavior (vLLM does not care about the
    declared mime for base64 image payloads)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------- inference ----------
def infer_stream(client, common_kwargs, repeat_min_repeats: int) -> tuple[str, bool]:
    """Streaming generation with tail-repetition early-stop.

    Returns (text, early_stopped).
    """
    stream = client.chat.completions.create(stream=True, **common_kwargs)
    parts: List[str] = []
    acc_len = 0
    next_check_at = 4000       # start checking after 4k chars
    check_step = 1000
    early_stopped = False

    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        piece = getattr(delta, "content", None)
        if not piece:
            continue
        parts.append(piece)
        acc_len += len(piece)

        if acc_len >= next_check_at:
            next_check_at = acc_len + check_step
            tail = "".join(parts)[-8000:]
            if has_tail_repetition(tail, min_repeats=repeat_min_repeats):
                early_stopped = True
                try:
                    stream.close()
                except Exception:
                    pass
                break

    return "".join(parts), early_stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="tencent/HunyuanOCR-1-5",
                    help="Must match vLLM --served-model-name")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--repeat-min-repeats", type=int, default=8,
                    help="tail-repeats threshold that triggers streaming early-stop")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming + early-stop (one-shot generation)")
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

    common_kwargs = dict(
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

    print(f"[info] POST http://{args.host}:{args.port}/v1/chat/completions "
          f"(model={args.model}, stream={not args.no_stream})")
    t = time.time()

    early_stopped = False
    usage = None
    if args.no_stream:
        resp = client.chat.completions.create(stream=False, **common_kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
    else:
        text, early_stopped = infer_stream(
            client, common_kwargs, args.repeat_min_repeats,
        )

    text = clean_repeated_substrings(text)
    dt = time.time() - t

    print("=" * 60)
    print(f"Latency          : {dt:.2f}s")
    print(f"Output chars     : {len(text)}")
    if usage is not None:
        print(f"Prompt tokens    : {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        if usage.completion_tokens:
            print(f"tok/s            : {usage.completion_tokens / dt:.1f}")
    if early_stopped:
        print("Early-stopped    : yes (tail repetition detected)")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
