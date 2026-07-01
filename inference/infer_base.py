"""
HunyuanOCR-1.5 base model inference on a **single image** via HuggingFace transformers.

Model class: `HunYuanVLForConditionalGeneration` (registered by the HunyuanOCR-1.5
transformers integration). The processor auto-loads via `AutoProcessor`.

For production serving, prefer vLLM (see `inference/serve_vllm.sh`
+ `inference/infer_vllm_client.py`). This script is intended for quick
correctness checking on a single image.

Usage:
    python inference/infer_base.py \
        --model /path/to/HunyuanOCR/base \
        --image /path/to/document.png
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，"
    "其中页眉、页脚部分忽略，表格用html格式表达，"
    "文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def _patch_hunyuan_tokenizer_special_tokens(tokenizer) -> None:
    """Backfill missing special-token attributes on older HunyuanOCR tokenizers.

    Some HunyuanOCR tokenizer configs only define image tokens; the transformers
    HunYuanVLProcessor also reads video-token attributes during __init__, even
    for image-only inference. We add defaults without resizing the vocab.
    """
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    extra_tokens = init_kwargs.get("extra_special_tokens", {}) or {}

    defaults = {
        "image_token": "<｜hy_place▁holder▁no▁102｜>",
        "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
        "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
        "video_token": "<｜hy_place▁holder▁no▁103｜>",
        "video_start_token": "<｜hy_place▁holder▁no▁104｜>",
        "video_end_token": "<｜hy_place▁holder▁no▁105｜>",
    }
    for name, default_value in defaults.items():
        if hasattr(tokenizer, name):
            continue
        value = extra_tokens.get(name)
        if value is None and name == "video_token":
            value = extra_tokens.get("image_token")
        setattr(tokenizer, name, value or default_value)


def _load_processor_with_patch(model_path: str):
    """Fallback processor loader for older tokenizer snapshots missing video_token."""
    from transformers import AutoImageProcessor, AutoTokenizer

    processor_module = importlib.import_module(
        "transformers.models.hunyuan_vl.processing_hunyuan_vl"
    )
    HunYuanVLProcessor = processor_module.HunYuanVLProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _patch_hunyuan_tokenizer_special_tokens(tokenizer)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    video_processor = None
    try:
        from transformers import AutoVideoProcessor
        video_processor = AutoVideoProcessor.from_pretrained(model_path)
    except Exception:
        video_processor = None

    try:
        return HunYuanVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )
    except TypeError:
        return HunYuanVLProcessor(image_processor, tokenizer, video_processor)


def load_model_and_processor(model_path: str, dtype_name: str,
                             attn_implementation: str = "eager"):
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    dtype = getattr(torch, dtype_name)
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except AttributeError as e:
        if "video_token" not in str(e):
            raise
        print("[warn] AutoProcessor tokenizer lacks video_token; "
              "retrying with patched Hunyuan tokenizer.", file=sys.stderr)
        processor = _load_processor_with_patch(model_path)

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor, torch


def infer_one(model, processor, torch_module, image_path: str, prompt: str,
              max_new_tokens: int) -> str:
    from PIL import Image

    with Image.open(image_path) as raw:
        image = raw.convert("RGB")

    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch_module.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
    trimmed = [out_ids[len(in_ids):]
               for in_ids, out_ids in zip(input_ids, generated_ids)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HunyuanOCR base model directory")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn-implementation", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()

    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading model from {args.model} ...")
    t0 = time.time()
    model, processor, torch_mod = load_model_and_processor(
        args.model, args.dtype, args.attn_implementation,
    )
    print(f"[info] model loaded in {time.time() - t0:.1f}s")

    print("[info] generating ...")
    t = time.time()
    text = infer_one(model, processor, torch_mod,
                     args.image, args.prompt, args.max_new_tokens)
    dt = time.time() - t

    print("=" * 60)
    print(f"Latency         : {dt:.2f}s")
    print(f"Completion chars: {len(text)}")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
