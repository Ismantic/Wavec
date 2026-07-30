# Wavec

[English](README_EN.md) | 中文

Wavec 是一个面向教学的中文词向量项目，使用 C++17 从零实现：

- CBOW（Continuous Bag of Words）
- Huffman Tree
- Hierarchical Softmax
- 高频词下采样
- 多线程 Hogwild 训练
- Cosine 相似词查询
- Spherical K-means 词聚类

项目强调算法清晰和流程完整，不依赖现成的词向量训练框架。默认使用
[THUCNews](https://huggingface.co/datasets/SirlyDreamer/THUCNews) 作为语料，
通过 [Wapic](https://github.com/Ismantic/Wapic) 完成中文分词。

## 项目结构

```text
src/                 C++ 训练、查询和聚类实现
scripts/             分词、过滤和完整训练流程
data/                THUCNews 下载与转换代码
tests/               小语料端到端测试
prepare/             生成的分词语料（不提交）
CMakeLists.txt        C++ 构建配置
requirements.txt      Python 数据处理依赖
```

完整数据流：

```text
Hugging Face THUCNews
        ↓
一行一句的原始语料
        ↓ Wapic
空格分隔的分词语料
        ↓ CBOW + Hierarchical Softmax
model.vec
        ↓ 词频与词长过滤
model.filter.vec
        ↓ Spherical K-means
clusters.txt + clusters.map
```

## 运行环境

需要 CMake 3.14+、支持 C++17 的编译器和 Python 3.9+。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建后生成：

- `build/wavec`：训练词向量
- `build/sim`：查询相似词
- `build/kmeans`：聚类词向量

## 训练过程

一条命令完成下载、转换、分词、训练、过滤和聚类：

```bash
make -C scripts all NPROC=16 THREADS=16
```

各阶段也可以单独执行：

```bash
make -C scripts data       # 下载并转换 THUCNews
make -C scripts cut        # 使用 Wapic 分词
make -C scripts fit        # 训练词向量
make -C scripts filter     # 过滤低频词和单字词
make -C scripts kmeans     # 聚类并导出映射
```

常用参数：

```bash
make -C scripts fit \
  DIM=100 WINDOW=8 MINCOUNT=5 ITER=5 THREADS=16

make -C scripts kmeans \
  MINFREQ=10 MINLEN=2 K=100 MAX_ITER=50 TOPN=20
```

默认训练产物均位于仓库根目录：

| 文件 | 内容 |
|---|---|
| `model.vec` | 完整词向量 |
| `model.filter.vec` | 过滤后的词向量 |
| `clusters.txt` | 每个簇最接近中心的词 |
| `clusters.map` | `word<TAB>cluster_id` 全量映射 |

下载数据、分词语料和模型文件已加入 `.gitignore`。

## 命令行工具

### 训练

```bash
./build/wavec [options] <segmented.txt> <model.vec>
```

主要选项包括 `-dim`、`-window`、`-mincount`、`-threads`、`-iter` 和
`-sample`。输入语料每行表示一个句子，词之间以空格分隔。

### 相似词查询

```bash
./build/sim model.filter.vec 10
> 北京
成都    0.765
武汉    0.760
南京    0.758
```

输入 `quit` 或空行退出。

### 词聚类

```bash
./build/kmeans model.filter.vec 100 50 20 \
  --output clusters.txt \
  --export clusters.map
```

聚类使用归一化词向量和余弦相似度。固定随机种子保证聚类初始化可复现。

## 数据管理

数据层只依赖 Hugging Face 上的 `SirlyDreamer/THUCNews`：

```bash
make -C data status
make -C data download
make -C data process
```

原始 Parquet 文件保存在 `data/downloads/`，转换后的语料保存在
`data/derived/THUCNews.sentences.txt`。生成过程采用临时文件，成功后才替换正式产物。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

测试使用临时小语料，覆盖训练、相似词查询、聚类导出和常见非法参数，不需要下载
THUCNews。

## 说明

训练器为便于理解，会将分词语料转换为整数索引后保存在内存中。多线程训练采用经典
word2vec 风格的无锁 Hogwild 更新，因此速度快、代码直接，但相同参数的不同运行不保证
逐位一致。该取舍适合学习算法，不以大型生产训练平台为目标。

CBOW、Huffman Tree、Hierarchical Softmax 和梯度更新的原理讲解见
[番外篇：词向量 W2V](https://ismantic.github.io/text/wavec.html)。

## License

MIT。THUCNews 和 Wapic 的许可条件请分别参考其上游项目。
