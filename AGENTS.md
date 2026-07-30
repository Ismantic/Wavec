# Repository Guidelines

## Project Structure & Module Organization

Wavec is a C++17 implementation of CBOW training with hierarchical softmax for Chinese word vectors.

- `src/ft_wav.cc` and `src/ft_wav.h`: core training implementation and `wavec` CLI.
- `src/sim.cc`: interactive cosine-similarity lookup tool.
- `src/kmeans.cc`: spherical K-means clustering and mapping export.
- `scripts/`: corpus segmentation, model filtering, training helpers, and the pipeline `Makefile`.
- `data/`: Hugging Face download/conversion code plus checked-in sample outputs; generated corpora are ignored.
- `prepare/`: generated segmented corpus used by the trainer.

Keep new executable entry points in `src/` and add them explicitly to `CMakeLists.txt`. Put reusable workflow automation in `scripts/`.

## Build, Test, and Development Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces `build/wavec`, `build/sim`, and `build/kmeans`.

```bash
make -C scripts cut NPROC=4
make -C scripts fit THREADS=8
make -C scripts filter MINFREQ=10 MINLEN=2
make -C scripts kmeans K=100 MAX_ITER=50
```

These commands run the pipeline described in `README.md`. Raw input comes only from `SirlyDreamer/THUCNews` on Hugging Face; do not commit downloaded corpora or generated `*.vec` files.

## Coding Style & Naming Conventions

Match the existing style: four-space indentation, braces on the same line, and standard-library types with explicit `std::` qualification. Use `snake_case` for variables and functions, `PascalCase` for structs/classes, and uppercase names for shell constants. Prefer RAII containers (`std::vector`, streams) over manual allocation. Python code should follow PEP 8 and use `argparse` for CLI options. Shell scripts should retain `set -euo pipefail` and quote paths and expansions.

No formatter or linter is configured, so keep changes focused and stylistically consistent with neighboring code.

## Testing Guidelines

Run `ctest --test-dir build --output-on-failure` for the small end-to-end CLI test. At minimum, also smoke-test the command affected. For example, run `./build/sim <small-model.vec> 5` or execute K-means on a small fixture and verify both `--output` and `--export` files. Add deterministic tests when introducing logic that can be isolated; K-means currently uses a fixed seed.

## Commit & Pull Request Guidelines

History favors short, imperative subjects such as `Add --export flag...` and `Fix thread safety...`; avoid vague `Update` messages. Keep each commit scoped to one behavior. Pull requests should explain the motivation, list commands run, note input/data assumptions, and include representative CLI output when behavior changes. Link relevant issues and document any new flags in `README.md`.
