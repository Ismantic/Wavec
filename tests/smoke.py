#!/usr/bin/env python3
"""Small end-to-end test for the three command-line tools."""

import subprocess
import sys
import tempfile
from pathlib import Path


WAVEC, SIM, KMEANS = sys.argv[1:4]


def run(*args, input_text=None, expected=0):
    result = subprocess.run(
        args, input=input_text, text=True, capture_output=True, check=False
    )
    if result.returncode != expected:
        raise AssertionError(
            f"{' '.join(args)} returned {result.returncode}, expected {expected}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    corpus = root / "tiny.cut.txt"
    model = root / "tiny.vec"
    clusters = root / "clusters.txt"
    mapping = root / "clusters.map"
    corpus.write_text(
        ("北京 上海 城市\n上海 深圳 城市\n北京 深圳 中国\n") * 4,
        encoding="utf-8",
    )

    run(
        WAVEC,
        "-dim", "8",
        "-window", "2",
        "-mincount", "1",
        "-threads", "1",
        "-iter", "1",
        "-sample", "0",
        str(corpus),
        str(model),
    )
    assert model.read_text(encoding="utf-8").splitlines()[0] == "5 8"

    result = run(SIM, str(model), "100", input_text="北京\nquit\n")
    assert "Not in vocabulary." not in result.stdout

    run(
        KMEANS,
        str(model),
        "2",
        "3",
        "2",
        "--output",
        str(clusters),
        "--export",
        str(mapping),
    )
    assert clusters.is_file()
    assert len(mapping.read_text(encoding="utf-8").splitlines()) == 5

    run(SIM, str(root / "missing.vec"), expected=1)
    run(KMEANS, str(model), "6", expected=1)
