# Wavec

CBOW + Hierarchical Softmax 词向量训练，兼具 FastText 风格的文档分类能力。

## 特性

- **CBOW 架构**：通过上下文词预测中心词，学习词向量
- **霍夫曼树 Softmax**：将输出层计算复杂度从 O(V) 降至 O(log V)
- **文档分类**：支持 `__label__` 前缀的标签，联合训练词向量与分类器
- **多线程训练**：按文档分块并行
- **高频词下采样**：减少常见词的训练频次，提升稀有词的向量质量

详细的算法推导见 [W2V.md](W2V.md)。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 使用

直接运行内置测试：

```bash
./build/wavec
```

自定义训练需修改 `src/ft_wav.cc`，通过 API 配置参数：

```cpp
wavec::FastText model;
model.SetVecSize(100);      // 向量维度
model.SetWindow(5);         // 上下文窗口
model.SetMinCount(5);       // 词频阈值
model.SetMinLabelCount(1);  // 标签频次阈值
model.SetCores(8);          // 线程数
model.SetIter(10);          // 训练轮数
model.SetSample(1e-3);      // 下采样阈值

model.Fit("train.txt", "model.vec");
```

### 输入文件格式

每行一个文档，标签以 `__label__` 前缀标识，词语以空格分隔：

```
__label__sports __label__nba Lakers win championship
__label__tech Apple releases new iPhone
```

### 输出文件

- `model.vec` — 词向量文件（首行为 `词数 维度`，后续每行 `词 向量值...`）
- `model.vec.syn1` — 霍夫曼树节点参数

## License

MIT
