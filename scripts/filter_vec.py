#!/usr/bin/env python3
"""Filter a .vec model file by word frequency and character length.

Usage: filter_vec.py <seg_file> <model.vec> <output.vec> [--minfreq N] [--minlen N]

Reads word frequencies from <seg_file>, then filters <model.vec> keeping only
words that meet both the minimum frequency and minimum character length.
"""

import sys
import argparse
from collections import Counter


def count_freq(seg_file):
    counter = Counter()
    with open(seg_file) as f:
        for line in f:
            for w in line.split():
                counter[w] += 1
    return counter


def main():
    parser = argparse.ArgumentParser(description="Filter .vec model by frequency and word length")
    parser.add_argument("seg_file", help="Segmented corpus (for counting word freq)")
    parser.add_argument("model_file", help="Input model.vec")
    parser.add_argument("output_file", help="Output filtered.vec")
    parser.add_argument("--minfreq", type=int, default=0, help="Minimum word frequency (default: 0)")
    parser.add_argument("--minlen", type=int, default=0, help="Minimum word length in characters (default: 0)")
    args = parser.parse_args()

    print(f"Counting frequencies from {args.seg_file}...", file=sys.stderr)
    freq = count_freq(args.seg_file)

    kept = []
    with open(args.model_file) as f:
        header = f.readline().split()
        vocab_size, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.split(" ", 1)
            word = parts[0]
            if len(word) < args.minlen:
                continue
            if freq.get(word, 0) < args.minfreq:
                continue
            kept.append(line)

    with open(args.output_file, "w") as f:
        f.write(f"{len(kept)} {dim}\n")
        for line in kept:
            f.write(line)

    print(f"Kept {len(kept)}/{vocab_size} words -> {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
