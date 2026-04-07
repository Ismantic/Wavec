#!/bin/bash
# Parallel segmentation using iscut.
# Usage: segment.sh <input> <output> [nproc]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREPARE_DIR="$SCRIPT_DIR/../prepare"
ISCUT="$PREPARE_DIR/iscut"
DICT="$PREPARE_DIR/dict.txt"

INPUT=${1:?usage: segment.sh <input> <output> [nproc]}
OUTPUT=${2:?usage: segment.sh <input> <output> [nproc]}
NPROC=${3:-$(nproc)}

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

TOTAL=$(wc -l < "$INPUT")
CHUNK=$(( (TOTAL + NPROC - 1) / NPROC ))

echo "Splitting $TOTAL lines into $NPROC chunks of ~$CHUNK lines..."
split -l "$CHUNK" -d -a 3 "$INPUT" "$TMPDIR/part_"

echo "Segmenting with $NPROC processes..."
PIDS=()
for part in "$TMPDIR"/part_*; do
    out="$part.seg"
    "$ISCUT" --dict "$DICT" --cut "$part" "$out" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "Merging..."
cat "$TMPDIR"/part_*.seg > "$OUTPUT"
echo "Done: $(wc -l < "$OUTPUT") lines -> $OUTPUT"
