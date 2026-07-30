#!/usr/bin/env python3
"""Convert THUCNews articles into a line-oriented training corpus."""

import re
import os
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "downloads" / "THUCNews" / "default" / "train"
OUTPUT = HERE / "derived" / "THUCNews.sentences.txt"
BOUNDARY = re.compile(r"(?<=[。！？!?])|[\r\n]+")
SPACE = re.compile(r"\s+")


def sentences(value):
    if not value:
        return
    for part in BOUNDARY.split(value):
        line = SPACE.sub(" ", part).strip()
        if line:
            yield line


def main():
    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files under {INPUT_DIR}; run data/download.py")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    rows = lines = 0
    with temporary.open("w", encoding="utf-8") as output:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=2048, columns=["title", "text"]
            ):
                titles = batch.column("title").to_pylist()
                texts = batch.column("text").to_pylist()
                for title, text in zip(titles, texts):
                    rows += 1
                    for value in (title, text):
                        for line in sentences(value):
                            output.write(line + "\n")
                            lines += 1
            print(f"{path.name}: {rows:,} articles, {lines:,} sentences")
    os.replace(temporary, OUTPUT)
    print(f"Wrote {lines:,} sentences from {rows:,} articles -> {OUTPUT}")


if __name__ == "__main__":
    main()
