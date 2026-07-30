#!/bin/bash
# Full pipeline: download THUCNews → segment with Wapic → train word vectors.
# Usage: train.sh <output_model> [threads]
set -euo pipefail

OUTPUT=${1:?usage: train.sh <output_model> [threads]}
THREADS=${2:-16}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$SCRIPT_DIR/.."
PYTHON=${PYTHON:-"$WORKDIR/.venv/bin/python"}

make -C "$SCRIPT_DIR" fit \
    PY="$PYTHON" NPROC="$THREADS" THREADS="$THREADS" OUTPUT="$OUTPUT"

echo "=== Done: $OUTPUT ==="
