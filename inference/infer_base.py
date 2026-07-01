"""
HunyuanOCR base model inference (single image) via HuggingFace transformers.

For production serving, use vLLM instead (see inference/serve_ar.sh).
This script is intended for quick debugging on a single image.

Usage:
    python inference/infer_base.py \
        --image /path/to/document.png \
        --model /path/to/HunyuanOCR/base

Example prompts:
    "Extract all text from this document image as markdown."
    "提取文档图片中正文的所有信息用markdown格式表示，页眉、页脚忽略，
     表格用html格式，公式用latex格式，按照阅读顺序组织。"
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，表格用html格式表达，"
    "文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--model", required=True, help="HunyuanOCR base model directory")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading model from {args.model} ...")
    t0 = time.time()
    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device).eval()
    print(f"[info] model loaded in {time.time()-t0:.1f}s")

    image = Image.open(args.image).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": args.prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(args.device)

    print("[info] generating ...")
    t = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    dt = time.time() - t

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_len:]
    text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)

    print("=" * 60)
    print(f"Latency         : {dt:.2f}s")
    print(f"Completion tokens: {len(completion_ids)}")
    print(f"tok/s           : {len(completion_ids)/dt:.1f}")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
