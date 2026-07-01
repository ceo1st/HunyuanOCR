"""
HunyuanOCR-1.5 + DFlash draft correctness check on a **single image** via
HuggingFace transformers.

For real end-to-end acceleration, deploy via vLLM (see `inference/serve_dflash.sh`).
This transformers path is only a correctness / sanity check for a DFlash draft
checkpoint before deploying to vLLM — it does **not** run the speculative
decoding loop; it merely loads the base model + DFlash draft, prints the draft
parameter count, and produces the AR reference output for comparison.

Usage:
    python inference/infer_dflash.py \
        --model /path/to/HunyuanOCR/base \
        --dflash-model ./hyocr_dflash/ \
        --image /path/to/document.png
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

# Reuse the exact loading / prompting logic used by infer_base.py so the AR
# reference output matches the vLLM baseline byte-for-byte.
from infer_base import (  # noqa: E402
    DEFAULT_PROMPT,
    infer_one,
    load_model_and_processor,
)


def load_dflash_draft(dflash_dir: str, dtype, device: str):
    """Load DFlash draft model from a dir with `dflash.py` + `config.json` + weights."""
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
    ap.add_argument("--model", required=True, help="HunyuanOCR base model dir")
    ap.add_argument("--dflash-model", required=True,
                    help="DFlash draft dir (contains dflash.py + config.json + weights)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=8000)
    ap.add_argument("--num-spec-tokens", type=int, default=15,
                    help="Number of draft tokens per speculative step (used only under vLLM)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn-implementation", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()

    if not Path(args.image).is_file():
        print(f"[ERROR] image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading base model from {args.model} ...")
    t0 = time.time()
    model, processor, torch_mod = load_model_and_processor(
        args.model, args.dtype, args.attn_implementation,
    )
    print(f"[info] base model loaded in {time.time() - t0:.1f}s")

    device = str(next(model.parameters()).device)
    print(f"[info] loading DFlash draft from {args.dflash_model} ...")
    t0 = time.time()
    dtype = getattr(torch_mod, args.dtype)
    draft = load_dflash_draft(args.dflash_model, dtype, device)
    n_params = sum(p.numel() for p in draft.parameters()) / 1e6
    print(f"[info] draft loaded in {time.time() - t0:.1f}s ({n_params:.1f} M params)")

    print("[info] generating AR reference (correctness path; "
          "real speculative decoding requires vLLM) ...")
    t = time.time()
    text = infer_one(model, processor, torch_mod,
                     args.image, args.prompt, args.max_new_tokens)
    dt = time.time() - t

    print("=" * 60)
    print(f"Latency         : {dt:.2f}s  (transformers AR reference, not spec decoding)")
    print(f"Completion chars: {len(text)}")
    print("=" * 60)
    print(text[:2000])
    if len(text) > 2000:
        print(f"\n... ({len(text)} chars total, truncated)")

    print(
        "\n[hint] For real DFlash acceleration, deploy via vLLM:\n"
        "       bash inference/serve_dflash.sh"
    )


if __name__ == "__main__":
    main()
