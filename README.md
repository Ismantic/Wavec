# Wavec

CBOW + Hierarchical Softmax 中文词向量训练。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成三个可执行文件：`wavec`（训练）、`sim`（相似词查询）、`kmeans`（聚类）。

## 数据准备

将以下文件放入 `prepare/` 目录：

- `iscut` — 分词工具（来自 [Iscut](https://github.com/Ismantic/Iscut)）
- `dict.txt` — 分词词典
- `News.documents.txt` — 语料文件，每行一个句子

## 使用

通过 `scripts/Makefile` 驱动完整流程：

```bash
make -C scripts cut      # 分词
make -C scripts fit      # 训练（若未分词则自动分词）
make -C scripts filter   # 按词频和字数过滤模型
make -C scripts kmeans   # 聚类（若未过滤则自动过滤）
```

```bash
make -C scripts fit NPROC=8 THREADS=16
make -C scripts kmeans MINFREQ=20 MINLEN=2 K=100
```

## 工具

### sim — 相似词查询

```bash
./build/sim <model.vec> [topk]
> 北京
深圳    0.699
成都    0.692
广东    0.670
...
```

### kmeans — 词聚类

```bash
./build/kmeans <model.vec> <k> [max_iter] [topn]
```

使用球面 K-means（cosine similarity + round-robin 初始化）对词向量聚类。

通过 `make -C scripts kmeans` 运行时会产出两个文件：

- `clusters.txt` — 每簇 top N 词及其相似度
- `clusters.map` — 全部词的聚类映射（`word\tclusterID`）

聚类前可通过 `filter_vec.py` 过滤低频词和单字词，避免低质量向量干扰聚类效果。

## License

MIT
