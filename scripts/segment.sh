#!/bin/bash
# Parallel segmentation using IsmaCut.
# Usage: segment.sh <dict> <input> <output> [nproc]
set -euo pipefail

DICT=${1:?usage: segment.sh <dict> <input> <output> [nproc]}
INPUT=${2:?usage: segment.sh <dict> <input> <output> [nproc]}
OUTPUT=${3:?usage: segment.sh <dict> <input> <output> [nproc]}
NPROC=${4:-$(nproc)}

ISMACUT="$(dirname "$0")/../../IsmaCut/build/ismacut"
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
    "$ISMACUT" --dict "$DICT" --cut "$part" "$out" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "Merging..."
cat "$TMPDIR"/part_*.seg > "$OUTPUT"
echo "Done: $(wc -l < "$OUTPUT") lines -> $OUTPUT"
