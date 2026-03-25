#!/bin/bash
# Full pipeline: extract → segment → train word vectors.
# Usage: train.sh <thucnews_dir> <output_model> [threads]
set -euo pipefail

THUC_DIR=${1:?usage: train.sh <thucnews_dir> <output_model> [threads]}
OUTPUT=${2:?usage: train.sh <thucnews_dir> <output_model> [threads]}
THREADS=${3:-16}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WAVEC="$SCRIPT_DIR/../build/wavec"
DICT="$SCRIPT_DIR/../../IsmaCut/dict/dict.txt"
WORKDIR="$SCRIPT_DIR/.."

echo "=== Step 1: Extract text ==="
python3 "$SCRIPT_DIR/prepare_thuc.py" "$THUC_DIR" "$WORKDIR/train_thuc.txt"

echo "=== Step 2: Segment ==="
bash "$SCRIPT_DIR/segment.sh" "$DICT" "$WORKDIR/train_thuc.txt" "$WORKDIR/train_thuc_seg.txt" "$THREADS"

echo "=== Step 3: Train word vectors ==="
"$WAVEC" \
    -dim 100 \
    -window 5 \
    -mincount 5 \
    -threads "$THREADS" \
    -iter 5 \
    -sample 1e-3 \
    "$WORKDIR/train_thuc_seg.txt" "$OUTPUT"

echo "=== Done: $OUTPUT ==="
