# Wavec

English | [中文](README.md)

Wavec is an educational Chinese word-embedding project that implements the
following components from scratch in C++17:

- CBOW (Continuous Bag of Words)
- Huffman trees
- Hierarchical softmax
- Frequent-word subsampling
- Multithreaded Hogwild training
- Cosine-similarity queries
- Spherical K-means clustering

The project favors readable algorithms and a complete, reproducible workflow
over production-scale abstractions. It uses
[THUCNews](https://huggingface.co/datasets/SirlyDreamer/THUCNews) as its default
corpus and [Wapic](https://github.com/Ismantic/Wapic) for Chinese word
segmentation.

## Project Layout

```text
src/                 C++ training, query, and clustering tools
scripts/             Segmentation, filtering, and pipeline commands
data/                THUCNews download and conversion code
tests/               Small end-to-end test
prepare/             Generated segmented corpus (not committed)
CMakeLists.txt        C++ build configuration
requirements.txt      Python data-processing dependencies
```

The complete data flow is:

```text
Hugging Face THUCNews
        ↓
one sentence per line
        ↓ Wapic
space-separated segmented corpus
        ↓ CBOW + hierarchical softmax
model.vec
        ↓ frequency and length filtering
model.filter.vec
        ↓ spherical K-means
clusters.txt + clusters.map
```

## Requirements and Build

Wavec requires CMake 3.14+, a C++17 compiler, and Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build creates three executables:

- `build/wavec`: train word embeddings
- `build/sim`: query similar words
- `build/kmeans`: cluster word embeddings

## Training from Scratch

Run the complete download, conversion, segmentation, training, filtering, and
clustering pipeline with:

```bash
make -C scripts all NPROC=16 THREADS=16
```

Each stage can also be run separately:

```bash
make -C scripts data       # Download and convert THUCNews
make -C scripts cut        # Segment the corpus with Wapic
make -C scripts fit        # Train word embeddings
make -C scripts filter     # Remove rare and single-character words
make -C scripts kmeans     # Cluster words and export assignments
```

Common configuration examples:

```bash
make -C scripts fit \
  DIM=100 WINDOW=8 MINCOUNT=5 ITER=5 THREADS=16

make -C scripts kmeans \
  MINFREQ=10 MINLEN=2 K=100 MAX_ITER=50 TOPN=20
```

Generated training artifacts are written to the repository root:

| File | Contents |
|---|---|
| `model.vec` | Complete word embeddings |
| `model.filter.vec` | Filtered word embeddings |
| `clusters.txt` | Words nearest to each cluster centroid |
| `clusters.map` | Full `word<TAB>cluster_id` mapping |

Downloaded data, segmented corpora, and generated models are ignored by Git.

## Command-Line Tools

### Training

```bash
./build/wavec [options] <segmented.txt> <model.vec>
```

The main options are `-dim`, `-window`, `-mincount`, `-threads`, `-iter`, and
`-sample`. Each input line represents one sentence, with words separated by
spaces.

### Similarity Queries

```bash
./build/sim model.filter.vec 10
> 北京
成都    0.765
武汉    0.760
南京    0.758
```

Enter `quit` or an empty line to exit.

### Word Clustering

```bash
./build/kmeans model.filter.vec 100 50 20 \
  --output clusters.txt \
  --export clusters.map
```

Clustering uses normalized word vectors and cosine similarity. A fixed random
seed makes cluster initialization reproducible.

## Data Management

The data layer depends only on `SirlyDreamer/THUCNews` hosted on Hugging Face:

```bash
make -C data status
make -C data download
make -C data process
```

Raw Parquet files are stored under `data/downloads/`. The converted corpus is
written to `data/derived/THUCNews.sentences.txt`. Conversion writes to a
temporary file first and replaces the final artifact only after success.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The test trains on a temporary miniature corpus and covers similarity queries,
cluster exports, and common invalid arguments. It does not download THUCNews.

## Implementation Notes

For clarity, the trainer converts the segmented corpus to integer IDs and keeps
the indexed documents in memory. Multithreaded training uses classic
word2vec-style lock-free Hogwild updates. This keeps the implementation compact
and fast, but separate runs with identical arguments are not guaranteed to be
bit-for-bit reproducible. Wavec is intended for learning the algorithms rather
than serving as a large production training platform.

A detailed Chinese explanation of CBOW, Huffman trees, hierarchical softmax,
and gradient updates is available in
[Word Vectors: W2V](https://ismantic.github.io/text/wavec.html).

## License

MIT. Refer to the upstream THUCNews and Wapic projects for their respective
license terms.
