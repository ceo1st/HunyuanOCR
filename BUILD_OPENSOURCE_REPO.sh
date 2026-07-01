#!/bin/bash
# ============================================================================
# One-shot script: copy the original training / model / dflash code from the
# internal repo into this open-source layout, skipping __pycache__ and other
# non-essentials.
#
# Usage:
#   bash BUILD_OPENSOURCE_REPO.sh /path/to/internal/hyocr_sft_code
#
# The target directory is the current dir (this script's location).
# ============================================================================

set -e

SRC=${1:-}
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "Usage: bash BUILD_OPENSOURCE_REPO.sh /path/to/internal/hyocr_sft_code"
    exit 1
fi

DST="$(cd "$(dirname "$0")" && pwd)"

echo "Source : $SRC"
echo "Target : $DST"
echo "----------------------------------------"

# ─── copy python code (train / tools / hyocr_dflash) ───
for d in train tools hyocr_dflash; do
    if [ -d "$SRC/$d" ]; then
        echo "[copy] $d/"
        mkdir -p "$DST/$d"
        rsync -a --exclude='__pycache__' --exclude='*.pyc' \
              "$SRC/$d/" "$DST/$d/"
    else
        echo "[skip] $SRC/$d (not found)"
    fi
done

# ─── remove old / dev shell scripts (they're superseded by our new ones) ───
echo ""
echo "[info] The following files/dirs already exist in this open-source repo:"
echo "       README.md, requirements.txt, LICENSE, .gitignore"
echo "       scripts/{env_common,pack_data,sft_base,sft_dflash,sft_dflash_finetune}.sh"
echo "       scripts/zero2.json, configs/data_list.txt"
echo "       inference/{serve_ar,serve_dflash}.sh, inference/infer_{base,dflash}.py"
echo "       docs/{training,data_format,inference,benchmark}.md"
echo ""
echo "[done] source code copied. Repo is ready for git init + push."
echo ""
echo "Next steps:"
echo "  1. cd $DST"
echo "  2. Fill in configs/data_list.txt with your data paths"
echo "  3. Edit MODEL_PATH placeholders in scripts/*.sh if needed"
echo "  4. git init && git add . && git commit -m 'initial open-source release'"
