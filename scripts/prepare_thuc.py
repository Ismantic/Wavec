#!/usr/bin/env python3
"""Extract THUCNews content to plain text, one article per line."""

import json
import glob
import sys

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <thucnews_dir> <output_file>", file=sys.stderr)
        sys.exit(1)

    thuc_dir = sys.argv[1]
    outfile = sys.argv[2]

    count = 0
    with open(outfile, "w") as out:
        for fp in sorted(glob.glob(f"{thuc_dir}/*.jsonl")):
            for line in open(fp):
                d = json.loads(line)
                text = d["content"].replace("\n", " ").strip()
                if text:
                    out.write(text + "\n")
                    count += 1

    print(f"Wrote {count} lines to {outfile}", file=sys.stderr)

if __name__ == "__main__":
    main()
