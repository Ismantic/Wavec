# Wavec

CBOW + Hierarchical Softmax 中文词向量训练。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成三个可执行文件：`wavec`（训练）、`sim`（近义词查询）、`kmeans`（聚类）。

## 数据准备

将以下文件放入 `prepare/` 目录：

- `iscut` — 分词工具（来自 [Iscut](https://github.com/Ismantic/Iscut)）
- `dict.txt` — 分词词典
- `News.sentences.txt` — 语料文件，每行一个句子

## 训练

通过 `scripts/Makefile` 驱动分词和训练：

```bash
# 分词
make -C scripts cut

# 分词 + 训练（若已分词则跳过）
make -C scripts fit
```

可选参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| NPROC | 4 | 分词并行进程数 |
| THREADS | 6 | 训练线程数 |
| OUTPUT | model.vec | 输出模型路径 |

```bash
make -C scripts fit NPROC=8 THREADS=16 OUTPUT=my.vec
```

### wavec 参数

```bash
./build/wavec [options] <input> <output>
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| -dim | 100 | 向量维度 |
| -window | 5 | 上下文窗口 |
| -mincount | 5 | 最低词频 |
| -threads | 4 | 线程数 |
| -iter | 5 | 训练轮数 |
| -sample | 1e-3 | 高频词下采样阈值 |

输入文件每行一个句子，词语空格分隔。输出为 word2vec 文本格式。

## 工具

### sim — 近义词查询

```bash
./build/sim <model.vec> [topk]
> 中国
韩国    0.598
日本    0.596
...
```

### kmeans — 词聚类

```bash
./build/kmeans <model.vec> <k> [max_iter] [topn]
```

使用球面 K-means（cosine similarity + round-robin 初始化）对词向量聚类。

## License

MIT
