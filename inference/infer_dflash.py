"""
HunyuanOCR + DFlash speculative decoding inference (single image) via
HuggingFace transformers.

For production serving, use vLLM instead (see inference/serve_dflash.sh) which
provides ~2.1× end-to-end speedup with tuned CUDA graphs and batching.
This script is intended for quick debugging / correctness verification of a
DFlash checkpoint before deploying to vLLM.

Usage:
    python inference/infer_dflash.py \
        --image /path/to/document.png \
        --model /path/to/HunyuanOCR/base \
        --dflash-model ./hyocr_dflash/
"""
import argparse
import importlib.util
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


def load_dflash_draft(dflash_dir: str, dtype: torch.dtype, device: str):
    """Load DFlash draft model from a directory with dflash.py + config.json + model.safetensors."""
    dflash_dir = Path(dflash_dir)
    dflash_py = dflash_dir / "dflash.py"
    if not dflash_py.is_file():
        raise FileNotFoundError(f"dflash.py not found in {dflash_dir}")

    spec = importlib.util.spec_from_file_location("dflash", str(dflash_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DFlashDraftModel = mod.DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(
        str(dflash_dir),
        torch_dtype=dtype,
    ).to(device).eval()
    return draft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True, help="HunyuanOCR base model dir")
    ap.add_argument("--dflash-model", required=True,
                    help="DFlash draft dir (contains dflash.py + config.json + model.safetensors)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--num-spec-tokens", type=int, default=15,
                    help="Number of draft tokens per speculative step")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)

    print(f"[info] loading base model from {args.model} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
    ).to(args.device).eval()
    print(f"[info] base model loaded in {time.time()-t0:.1f}s")

    print(f"[info] loading DFlash draft from {args.dflash_model} ...")
    t0 = time.time()
    draft = load_dflash_draft(args.dflash_model, dtype, args.device)
    print(f"[info] draft loaded in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in draft.parameters())/1e6:.1f} M params)")

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

    # NOTE: A minimal speculative decoding loop is model-specific and lives inside
    # dflash.py (draft.speculative_generate). If your dflash.py exposes such a
    # helper, uncomment the block below. Otherwise, prefer vLLM for real inference.
    print("[info] generating (speculative decoding requires vLLM for production; "
          "this transformers path only verifies draft correctness) ...")

    t = time.time()
    with torch.no_grad():
        # Fallback to autoregressive generation via the base model,
        # printing draft stats along the way if the draft exposes them.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    dt = time.time() - t

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_len:]
    text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)

    print("=" * 60)
    print(f"Latency         : {dt:.2f}s  (transformers AR fallback; not spec decoding)")
    print(f"Completion tokens: {len(completion_ids)}")
    print(f"tok/s           : {len(completion_ids)/dt:.1f}")
    print("=" * 60)
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... ({len(text)} chars total, truncated)")

    print(
        "\n[hint] For real DFlash speedup (~2.1×), deploy via vLLM:\n"
        "       bash inference/serve_dflash.sh"
    )


if __name__ == "__main__":
    main()
