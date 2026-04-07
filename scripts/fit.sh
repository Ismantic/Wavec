#!/bin/bash
# Full pipeline: segment → train word vectors.
# Usage: train.sh <output_model> [threads]
set -euo pipefail

OUTPUT=${1:?usage: train.sh <output_model> [threads]}
THREADS=${2:-16}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$SCRIPT_DIR/.."
WAVEC="$WORKDIR/build/wavec"
INPUT="$WORKDIR/prepare/News.sentences.txt"
SEG_FILE="$WORKDIR/prepare/News.cut.txt"

echo "=== Step 1: Segment ==="
bash "$SCRIPT_DIR/cut.sh" "$INPUT" "$SEG_FILE" "$THREADS"

echo "=== Step 2: Train word vectors ==="
"$WAVEC" \
    -dim 100 \
    -window 5 \
    -mincount 5 \
    -threads "$THREADS" \
    -iter 5 \
    -sample 1e-3 \
    "$SEG_FILE" "$OUTPUT"

echo "=== Done: $OUTPUT ==="
