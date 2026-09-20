## Lecture 1

### 一些概念

#### FLOPS

指每秒浮点运算次数（Floating Point Operations Per Second），通常一次加法和一次乘法被视为1次浮点运算。

#### 扩展定律（Scaling Laws）

- 模型规模N，即超参数（hyperparameters）的数量
- 数据规模D，即训练数据的token量
- 计算量C，即算力

transformer的训练计算量通常为：
$C = 6 * N * D$

通常，如果算力额定，当$D = 20N$时，模型的性能会达到最优。

#### 困惑度（Perplexity）

衡量语言模型预测下一个 token 能力的指标，困惑度越低，说明模型预测能力越强。

计算公式为：

$$
\mathrm{PPL}
=
\exp\left(
-\frac{1}{T}
\sum_{t=1}^{T}
\log p(x_t\mid x_{<t})
\right)
$$

其中括号里面为交叉熵损失，因此：

$$
\mathrm{PPL}=e^{\mathcal{loss}}
$$

#### 模型训练的全监督与弱监督

- 全监督（Fully Supervised）：训练数据中每个样本都有明确的标签，模型通过学习这些标签来进行预测。

- 弱监督（Weakly Supervised）：训练信号并非完整、精确、可靠的人工标签，而是来自成本较低但可能存在噪声或模糊性的来源。

### tokenizer

tokenizer 是将文本转换为模型可以处理的数字序列的工具。

#### BPE（Byte Pair Encoding）

基本思想：通过原生文本训练分词器，来构建一个基于数据的词表。
如果遇到一个词不在词表(即UNK)中，则将其拆分为更小的子词，直到所有子词都在词表中。
