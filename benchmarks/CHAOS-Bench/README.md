<div align="center">

[中文阅读](./README_zh.md)

# CHAOS-Bench

**C**haracter-level **H**allucination **A**ssessment for **O**CR **S**eeing-is-believing

</div>

<p align="center">
<a href="https://arxiv.org/abs/2607.04884"><b>📄 Paper</b></a> •
<a href="https://github.com/Tencent-Hunyuan/HunyuanOCR"><b>💻 HunyuanOCR</b></a>
</p>

---

## 📖 Overview

**CHAOS-Bench** is a diagnostic benchmark for probing the *"seeing-is-believing"* ability of
vision-language models (VLMs) — i.e., whether a model truly **reads what is in the image** rather
than **hallucinating** text from its language prior.

The core idea is simple but revealing: we take real academic-paper page images and inject
**character-level corruptions** into selected words (e.g. `participant` → `qarticipant`,
`about` → `abcut`). These corrupted words are visually present but linguistically implausible.
A model that genuinely perceives the pixels should report the corrupted string; a model that leans
on its language prior will silently "auto-correct" it back to the plausible word — exposing a
perception–cognition gap and a form of OCR hallucination.

## 📊 Statistics

| Item | Value |
|---|---|
| Images | 500 |
| Annotated samples | 500 |
| Domain | Academic paper pages |
| Corruption type | Character-level word edits (with bounding boxes) |

## 🗂️ Data Format

The archive `CHAOS-Bench.zip` contains:

```
CHAOS-Bench/
├── CHAOS-Bench.jsonl     # annotations, one JSON object per line
└── images/               # 500 page images (001.png ... )
```

Each line in `CHAOS-Bench.jsonl` has the following fields:

| Field | Type | Description |
|---|---|---|
| `data_id` | string | Sample id, e.g. `"001"` |
| `image` | string | Relative image path, e.g. `"images/001.png"` |
| `change` | list | One or more injected corruptions in this image |

Each element of `change` describes a single corrupted word:

| Field | Type | Description |
|---|---|---|
| `ocr_ans` | string | The **corrupted** string actually rendered in the image (ground truth to be read) |
| `origin_ans` | string | The **original** correct word before corruption |
| `bbox` | list[int] | Bounding box of the word, `[x1, y1, x2, y2]` |

### Example

```json
{
  "data_id": "002",
  "image": "images/002.png",
  "change": [
    {"ocr_ans": "abcut", "origin_ans": "about", "bbox": [806, 809, 859, 826]},
    {"ocr_ans": "cf",    "origin_ans": "of",    "bbox": [765, 838, 787, 853]}
  ]
}
```

## 🧪 How to Evaluate

1. Unzip `CHAOS-Bench.zip`.
2. For each image, prompt the model to recognize the text at the given region (or perform full-page
   recognition), and locate the model output corresponding to each annotated `bbox`.
3. Compare the model output against `ocr_ans`:
   - Output matches **`ocr_ans`** → the model **truly read the pixels** ✅
   - Output matches **`origin_ans`** (the plausible word) → the model **hallucinated** from its
     language prior ❌
4. Report the rate at which the model faithfully reproduces the corrupted `ocr_ans` as the
   seeing-is-believing / anti-hallucination score.

## 📚 Citation

If you find CHAOS-Bench useful, please cite HunyuanOCR-1.5:

```bibtex
@misc{li2026hunyuanocr15,
      title={HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better},
      author={Gengluo Li and Xingyu Wan and Shangpin Peng and Weinong Wang and Hao Feng and Yongkun Du and Binghong Wu and Zheng Ruan and Zhiqiong Lu and Liang Wu and Pengyuan Lyu and Huawen Shen and Zibin Lin and Shijing Hu and Jieneng Yang and Hongbing Wen and Guanghua Yu and Hong Liu and Bochao Wang and Can Ma and Han Hu and Chengquan Zhang and Yu Zhou},
      year={2026},
      journal={arXiv preprint arXiv:2607.04884},
      url={https://arxiv.org/abs/2607.04884},
}
```
