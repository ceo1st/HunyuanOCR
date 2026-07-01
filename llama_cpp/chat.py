"""Minimal OpenAI-compatible client for the local vLLM server."""

from openai import OpenAI
import base64
import mimetypes
import json
import os
import sys
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "HYVL"
MAX_REQUESTS = 10

TYPE_LIMITS = {
    "ocr": 1,
}

_client = OpenAI(base_url=BASE_URL, api_key="empty")


class _Tee:
    """Write to multiple streams at once (e.g. terminal + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _redirect_output_to_log() -> str:
    """Tee stdout/stderr to a timestamped log file in logs/. Returns the path."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"chat_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _image_part(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat(prompt: str, image_paths: list = None, max_tokens: int = 4096, temperature: float = 0, top_p: float = 1, top_k: int = 1, repetition_penalty: float = 1) -> str:
    if image_paths:
        content = [{"type": "text", "text": prompt}, *(_image_part(p) for p in image_paths)]
    else:
        content = prompt

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k, "repetition_penalty": repetition_penalty},
    )
    return resp.choices[0].message.content or ""


def _iter_jsonl(path: str):
    """Yield objects from a pretty-printed (multi-line) JSONL file."""
    with open(path, "r") as f:
        text = f.read().strip()
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        yield obj


if __name__ == "__main__":
    _redirect_output_to_log()

    data_path = "test_assets/data.jsonl"
    base_dir = os.path.dirname(os.path.abspath(data_path))

    total_start = time.perf_counter()
    count = 0
    type_counts = {}  # how many of each type already run
    for item in _iter_jsonl(data_path):
        if MAX_REQUESTS is not None and count >= MAX_REQUESTS:
            break

        item_type = item.get("type", "unknown")

        # Filter by TYPE_LIMITS: skip types not selected, and respect per-type caps.
        if TYPE_LIMITS is not None:
            if item_type not in TYPE_LIMITS:
                continue
            limit = TYPE_LIMITS[item_type]
            if limit is not None and type_counts.get(item_type, 0) >= limit:
                continue

        prompt = item["prompt"]
        image_paths = [os.path.join(base_dir, p) for p in item.get("image_paths", [])]
        label = ", ".join(item.get("image_paths", [])) or "(text-only)"
        print(f"=== [{item_type}] {label} ===")
        print(f"Prompt: {prompt}")
        item_start = time.perf_counter()
        print(chat(prompt, image_paths=image_paths or None))
        print(f"[elapsed] {time.perf_counter() - item_start:.3f}s")
        print()
        count += 1
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    summary = ", ".join(f"{t}={n}" for t, n in type_counts.items()) or "none"
    print(f"[total] {count} items ({summary}), elapsed: {time.perf_counter() - total_start:.3f}s")
