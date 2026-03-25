# Wavec

CBOW + Hierarchical Softmax 中文词向量训练。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

生成三个可执行文件：`wavec`（训练）、`sim`（近义词查询）、`kmeans`（聚类）。

## 训练

```bash
./build/wavec [options] <input> <output>
```

选项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| -dim | 100 | 向量维度 |
| -window | 5 | 上下文窗口 |
| -mincount | 5 | 最低词频 |
| -threads | 4 | 线程数 |
| -iter | 5 | 训练轮数 |
| -sample | 1e-3 | 高频词下采样阈值 |

输入文件每行一个文档，词语空格分隔。输出为 word2vec 文本格式。

### 全流程脚本

`scripts/train.sh` 提供从 THUCNews 语料到词向量的完整 pipeline：

```bash
bash scripts/train.sh <thucnews_dir> <output_model> [threads]
```

1. 提取文本（`prepare_thuc.py`）
2. 并行分词（`segment.sh`，依赖 [IsmaCut](https://github.com/Ismantic/IsmaCut)）
3. 训练词向量

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

## 算法

- **CBOW**：上下文词的平均向量预测中心词
- **Hierarchical Softmax**：霍夫曼树将 softmax 复杂度从 O(V) 降至 O(log V)
- **两遍数据加载**：第一遍统计词频建词典，第二遍转为整数索引，避免内存溢出

详细推导见 [W2V.md](W2V.md)。

## License

MIT
