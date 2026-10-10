---
outline: [2, 3]
---

# 第 2 章 · Transformer 架构

<span id="guide-ch-2"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>完成[分词器与 BPE](/part-1/chapter-1#guide-ch-1)；掌握矩阵乘法、张量形状和广播，参见[PyTorch 基础](/appendix#app-tensors)。

<strong>准备工作：</strong>准备 PyTorch 环境与小批量 token ID；先用小尺寸张量测试模块，不必先加载完整语料。

<strong>本章任务：</strong>实现线性层、嵌入层、RMSNorm、RoPE、因果多头注意力、SwiGLU，并组装 Transformer 语言模型。

</div>

首先要明确，语言模型是一种<strong class="key-term">自回归</strong>的<strong class="key-term">预测</strong>模型。

- <strong class="list-label">输入</strong>：

  - 语言模型不直接处理文字，而是处理数字。每个词、标点符号或更小的语言单位（如“ing”、“un-”）都会被赋予一个唯一的整数 ID(也就是上一章提到的<strong class="key-term">token</strong>) 。

  - 利用<strong class="key-term">并行计算</strong>，模型可以按照 batch 同时处理多段文本。

- <strong class="list-label">输出</strong>：

  - 对输入序列的每一个词，模型都会预测下一个词是什么，即给出概率分布。

  - 所有可能词的概率<strong class="key-term">必须标准化</strong>，即和为1。

- <strong class="list-label">训练</strong>：

  - 现代语言模型（尤其是像GPT这样的自回归模型）的预训练过程，本质上是一种基于大规模文本语料库的<strong class="key-term">自监督学习（Self-Supervised Learning）</strong>。其核心目标是最大化给定上文序列（Context）条件下，预测下一个词元（Token）的条件概率。

    - <strong class="list-label">目标任务：</strong>下一个词元预测。 模型学习的是一个概率分布 P(wₙ \| w₁, w₂, ..., wₙ₋₁)，即在已知前面所有词的条件下，下一个词 wₙ 出现的概率。

    - <strong class="list-label">学习范式：</strong>自监督学习 (Self-Supervised Learning)。由于训练的标签是从输入数据本身中<strong class="key-term">自动获取的</strong>，而非人工标注，因此被称为“自监督”。这种范式使得模型可以利用海量的、无标签的原始文本进行学习，是大型语言模型能够成功训练的关键。

    - <strong class="list-label">优化过程：</strong>最小化损失函数 (Loss Function Minimization)。当模型做出预测后，会通过一个名为<strong class="key-term">交叉熵损失（Cross-Entropy Loss）</strong>的函数，来量化其预测的概率分布与“真实标签”（即下一个词的<strong class="key-term">独热编码 one-hot vector</strong>）之间的差距。然后，模型使用<strong class="key-term">反向传播（Backpropagation）算法</strong>来调整内部数以亿计的参数，其目标就是让这个损失值尽可能小。这个不断调整参数以减少误差的过程，就是模型的<strong class="key-term">“学习”</strong>过程。

## 2.1 基础模块:线性层和嵌入层

<span id="sec-2-2"></span>

[参考资料 2.1](/part-1/chapter-2#read-2-1)

### 2.1.1 参数初始化

<span id="sec-2-2-1"></span> 我们训练神经网络的目的是为了找到一组参数，使得模型在训练数据上的表现最好。但是，如果初始参数设置不当，可能会导致模型无法收敛或者收敛到局部最优解甚至出现<strong class="key-term">梯度爆炸、梯度消失</strong>等问题。因此，参数初始化也是一门需要研究的学问。

最常见的两种参数初始化方法：<strong class="key-term">Xavier初始化和He初始化</strong>。

- <strong class="list-label">Xavier 初始化 (Glorot Initialization)</strong>

  Xavier初始化由 Xavier Glorot和Yoshua Bengio在2010年提出。它的核心思想是保持每一层激活值的<strong class="key-term">方差</strong>和反向传播时梯度的方差在前向和反向传播中保持不变。Xavier初始化适用于<strong class="key-term">Sigmoid, Tanh</strong> 等对称函数。

  Xavier初始化通常有两种分布形式：

  - <strong class="list-label">均匀分布 (Uniform)</strong>： 权重从均匀分布 U\[-r, r\] 中采样，其中： $r = \sqrt{\frac{6}{fan_{in} + fan_{out}}}$ 均匀分布方差为 $\frac{(b-a)^2}{12} = \frac{r^2}{3}$。 这里的 $fan_{in}$ 是一层网络输入神经元数量，$fan_{out}$ 是输出的神经元数量。

  - <strong class="list-label">正态分布 (Normal)</strong>： 权重从均值为0，标准差为 $\sigma$ 的正态分布中采样，其中： $\sigma = \sqrt{\frac{2}{fan_{in} + fan_{out}}}$ 这是由于每经过一层，权重的方差就会变成原来的 $\frac{1}{fan}$,其中$fan$表示这一层的神经元的数量。

  工作原理简述： 通过同时考虑输入和输出神经元的数量，Xavier初始化试图在层与层之间找到一个平衡点，使得信号的方差既不会在传播中衰减，也不会无限放大。

- <strong class="list-label">He 初始化 (Kaiming Initialization)</strong>

  He初始化由Kaiming He(何凯明，残差网络也是他提出的)在2015年提出。针对Xavier初始化在<strong class="key-term">ReLU激活函数</strong>上效果不佳的问题，He初始化就适配于<strong class="key-term">ReLU激活函数</strong>。

  ReLU函数 $f(x) = \max(0, x)$ 的特性是，它会将所有负输入都变为0。 这破坏了Xavier初始化所依赖的“激活函数关于原点对称”的假设，并导致大约一半的神经元输出为0，从而改变了输出的方差。

  He初始化考虑到ReLU会将一半的输入置为零，这会使得输出方差减半。为了补偿这一点，He初始化在计算方差时引入了一个因子2。其他和Xavier初始思路一致。

  - <strong class="list-label">均匀分布 (Uniform)</strong>： 权重从均匀分布 U\[-r, r\] 中采样，其中： $r = \sqrt{\frac{6}{fan}}$

  - <strong class="list-label">正态分布 (Normal)</strong>： 权重从均值为0，标准差为 $\sigma$ 的正态分布中采样，其中： $\sigma = \sqrt{\frac{2}{fan_{in}}}$

  之所以Kaiming初始化只考虑$fan_{in}$，是因为这个更注重<strong class="key-term">前向传播</strong>，不是很在意反向传播。

### 2.1.2 实验中参数初始化标准

- <strong class="list-label">线性层权重:</strong>$\mathcal{N}(\mu = 0, \sigma^2 = \frac{din+dout}{2})$,范围限制在$\pm 3\sigma$以内。

- <strong class="list-label">词嵌入权重:</strong>$\mathcal{N}(\mu = 0, \sigma^2 = 1)$,范围限制在$\pm 3$以内。

- <strong class="list-label">RMSNorm权重:</strong>1

## 2.2 词嵌入层(embedding layer)

<span id="sec-2-3"></span>

- <strong class="list-label">输入</strong>：整数 token ID 序列（这里已经是由之前的 BPE tokenizer 处理文本之后的结果了）

- <strong class="list-label">词嵌入（embedding）</strong>：目的是为了将 token ID 转换为<strong class="key-term">稠密向量</strong>；一个词嵌入层（embedding layer）的作用就是把这些离散的、没有语义的整数ID，转换为连续的、包含语义信息的<strong class="key-term">向量</strong>。

  比如，“猫”的向量可能和“狗”的向量在某种“动物”维度上比较接近，而和“汽车”的向量相距较远。每个嵌入层接收形状为 (batch_size, sequence_length) 的整数张量，并生成形状为 (batch_size, sequence_length, d_model) 的向量序列。d_model 代表输出维度，越大代表语义信息越丰富，模型越强大。

## 2.3 两大基础模块:线性模块和嵌入模块

<span id="sec-2-4"></span>

### 2.3.1 线性模块(Linear Module)

线性模块是神经网络里面最最基础的模块了，它就是一个线性变换，将输入的向量映射到另一个向量。

$$
y = Wx + b
$$

其中，$W$是权重矩阵，$b$是偏置向量，$x$是输入向量，$y$是输出向量。

实现的过程中<strong class="key-term">别忘了初始化</strong>就好。

```python
    class LinearModule(nn.Module):
        def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.device = device
            self.dtype = dtype
            self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
            # self.b = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            # 对权重进行Xavier初始化
            std = 2 / (self.in_features + self.out_features) ** 0.5
            torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.W.T
    
```

### 2.3.2 嵌入模块(Embedding Module)

嵌入模块就是我们之前所说的嵌入层所利用的模块，它的功能是将代表文本的整数“词元ID”（token ID）转换为高维的、模型能够理解的向量表示。

嵌入层用最高效的“<strong class="key-term">查字典</strong>”方式，实现了在数学上等价于“<strong class="key-term">一个输入为独热编码的线性层</strong>”的功能。输入为(batch_size, sequence_length)代表一个批次每个句子有sequence_length个token，输出为(batch_size, sequence_length, embedding_dim)代表每个token的embedding向量。

```python
    class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings # 词表vocab_size大小
        self.embedding_dim = embedding_dim # 词向量维度d_model
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids] # 从词表到词向量的映射的神经网络里读取token_ids输出词向量 
```

## 2.4 Pre-norm Transformer Block

<span id="sec-2-5"></span>

- <strong class="list-label">Transformer 块的基本结构</strong>：

  - <strong class="list-label">多头自注意力机制（Multi-Head Self-Attention mechanism）</strong>：这是 Transformer 的核心，允许模型在处理序列时对输入的不同部分赋予不同的权重。

  - <strong class="list-label">位置级前馈网络（Position-wise Feed-Forward Network）</strong>：一个简单的全连接层，独立地应用于序列中每个位置的向量。

- <strong class="list-label">残差连接（Residual Connection）</strong>：

  - 原始 Transformer 论文指出，每个子层都使用了残差连接。残差连接的目的是将子层的输入（x）直接加到子层的输出（Sublayer(x)）上，即 x + Sublayer(x)。

  - 这种设计有助于解决深度网络中的<strong class="key-term">梯度消失问题</strong>，使得信息可以直接通过多层网络传递。

- <strong class="list-label">层归一化（Layer Normalization）的不同应用方式</strong>：

  - <strong class="list-label">后归一化</strong>（Post-norm）Transformer：这是原始 Transformer 论文中使用的架构。层归一化是在每个子层的输出之后，残差连接之后应用的。即：LayerNorm(x + Sublayer(x))。

  - <strong class="list-label">前归一化</strong>（Pre-norm）Transformer：这是后续研究（Nguyen and Salazar, 2019; Xiong et al., 2020）发现的一种改进架构。层归一化是在每个子层的输入之前应用的。即：x + Sublayer(LayerNorm(x))。此外，在整个 Transformer 堆栈的最后一个 Transformer 块之后，还会有一个额外的<strong class="key-term">层归一化</strong>。

    - <strong class="note-label">优点</strong>：多项工作发现，这种“前归一化”的方式改善了 Transformer 的训练稳定性。

    - <strong class="note-label">直观解释（Intuition）</strong>：前归一化会创建一个“干净的残差流（residual stream）”，即从输入嵌入到 Transformer 最终输出的路径上，没有任何归一化操作直接作用于残差连接本身。这种设计被认为能改善梯度流动（gradient flow），使得梯度更容易有效地反向传播通过多层网络。

    - <strong class="note-label">当前标准</strong>：由于其训练稳定性的优势，“前归一化”已经成为现代大型语言模型（如 GPT-3, LLaMA, PaLM 等）的标准实践。

### 2.4.1 RMSNorm的原理

RMSnorm 是一种<strong class="key-term">简化版的层归一化（Layer Normalization）。</strong>

- <strong class="list-label">LN 层归一化的公式是：</strong>

  $$
  LN(x)=\gamma \odot \frac{x - \mu}{\sigma}+\beta
  $$

  - $x$ 是输入特征向量（对于 NLP 通常是 (batch_size, sequence_length, hidden_size) 中的一个 hidden_size 维度上的向量），这里的 hidden_size 通常就是 d_model。

  - $\mu$ 是<strong class="key-term">平均值（mean）</strong>，计算的是 $x$ 在其最后一个维度上的均值。

  - $\sigma$ 是<strong class="key-term">标准差（standard deviation）</strong>，计算的是 $x$ 在其最后一个维度上的标准差。

  - $\gamma$ 和 $\beta$ 是<strong class="key-term">可学习的缩放（scale）和偏移（shift）参数</strong>，它们的维度与 $x$ 的最后一个维度相同。它们允许归一化后的数据进行仿射变换，从而恢复模型的表达能力。

- <strong class="list-label">RMSnorm 的公式是：</strong>

  <div class="key-formula">

  $$
  RMSnorm(x)=\gamma⊙\frac{x}{RMS(x)}
  $$

  </div>

  - $x$ 是输入特征向量。

  - $RMS(x)$ 是 $x$ 的<strong class="key-term">均方根（Root Mean Square）</strong>。

    $$
    RMS(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2}=\sqrt{mean(x^2)}
    $$

  - 这里的 $D$ 是 $x$ 的特征维度大小（即 hidden_size）。

  - $\gamma$ 是<strong class="key-term">可学习的缩放（scale）参数</strong>，其维度与 $x$ 的最后一个维度相同。

  - <strong class="note-label">注意：</strong> RMSnorm <strong class="key-term">没有</strong> （偏移）参数 $\beta$。

  移除均值计算可以带来以下好处：

  - <strong class="list-label">计算效率更高</strong>：计算均值需要对所有元素求和再除以维度，这本身是一个 O(D) 的操作。移除这一步可以减少计算量，尤其是在硬件层面上可能更高效。

  - <strong class="list-label">内存占用更少</strong>：不计算和存储均值，也不需要 $\beta$ 参数。

  - <strong class="list-label">简化模型</strong>：参数更少，模型更简洁。

让我们再详细解释一下 (batch_size, sequence_length, hidden_size) 这种形状的张量在 Layer Normalization (LN) 或 RMSnorm 中是如何被归一化的。

1.  <strong class="note-label">(batch_size, sequence_length, hidden_size) 张量的含义</strong>

    - <strong class="list-label">batch_size</strong>: 批次大小，表示一次性处理多少个独立的序列（或句子）。

    - <strong class="list-label">sequence_length</strong>: 序列长度，表示每个序列中有多少个 token（词或子词）。

    - <strong class="list-label">hidden_size</strong>: 隐藏层维度，也常被称为 <strong class="key-term">d_model</strong>。这是每个 token 经过词嵌入层或其他层后，所对应的<strong class="key-term">稠密向量的维度</strong>。

    可以把这个三维张量想象成：batch_size 个矩阵，每个矩阵的形状是 (sequence_length, hidden_size)。而每个 (hidden_size) 向量，就是对应于序列中某个位置的某个 token 的表示。

2.  <strong class="note-label">Layer Normalization 和 RMSnorm 的归一化范围</strong>

    当你看到 $x$ 在 LN/RMSnorm 的公式中时，这个 $x$ 指的是：

    针对 (batch_size, sequence_length, hidden_size) 张量中的每一个 (batch_idx, sequence_idx) 对应的 hidden_size 维度上的向量。

    也就是说：对于批次中的每个样本 (batch_idx),对于序列中的每个位置 (sequence_idx),都会独立地提取出一个形状为 (hidden_size,) 的向量。Layer Normalization 或 RMSnorm 的均值/RMS 和标准差，就是在这个 (hidden_size,) 维度的向量上计算的。

    <div class="custom-block tip">

    <p class="custom-block-title">例子</p>

    具体地说：如果你的输入张量是 input_tensor，其形状为 (B, S, D) (B=batch_size, S=sequence_length, D=hidden_size)。当你应用 Layer Normalization 或 RMSnorm 时，它们会遍历：b 从 0 到 B-1,s 从 0 到 S-1,然后，对于每一个 (b, s) 对，它会取 input_tensor\[b, s, :\] 这个子向量（其形状为 (D,)），并对这个 (D,) 维度的向量进行归一化。所以，归一化操作是独立地应用于 B \* S 个这样的 D 维向量。

    </div>

3.  <strong class="note-label">词嵌入层之后的稠密向量</strong> batch_size 中一个，然后这个对应映射后的 hidden_size 也就是经过词嵌入层之后的稠密向量，对这个向量在进入 Transformer 块之前归一化。

    - <strong class="list-label">batch_size 中一个</strong>：对应于 input_tensor\[b, :, :\]，即批次中的一个完整序列。

    - <strong class="list-label">这个对应映射后的 hidden_size</strong>：对应于 input_tensor\[b, s, :\]，即该序列中某个特定 token（在 s 位置）的 hidden_size 维度的向量。

    - <strong class="list-label">也就是经过词嵌入层之后的稠密向量</strong>：正是如此。词嵌入层的输出形状就是 (batch_size, sequence_length, embedding_dim)，其中 embedding_dim 通常就是 hidden_size (或 d_model)。

    - <strong class="list-label">对这个向量在进入 Transformer 块之前归一化</strong>：这正是“前归一化”（pre-norm）Transformer 的核心思想。在每个子层（多头自注意力或前馈网络）的输入端，都会对这个 (hidden_size,) 维度的向量进行归一化。

<strong class="list-label">总结一下：</strong>

Layer Normalization 和 RMSnorm 总是沿着<strong class="key-term">特征维度</strong>（通常是最后一个维度，即 hidden_size 或 embedding_dim）进行归一化，并且这种归一化是<strong class="key-term">独立地</strong>应用于每个样本的每个序列位置的。它不涉及批次维度或序列长度维度上的统计计算。

这个操作确保了进入 Transformer 子层的每个 token 的向量表示，其数值范围都得到了稳定，从而有助于训练的<strong class="key-term">稳定性和效率</strong>。

```python
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #weight对应缩放参数 gamma
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 题目要求对于不同的精度要先转换为float32再进行归一化，最后再转换回原来的精度
            origin_dtype = x.dtype
            x_fp32 = x.to(torch.float32)
    
            norm_x = x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
            x_norm = norm_x.to(origin_dtype)
            return x_norm * self.weight
```

### 2.4.2 位置级前馈网络 (Position-wise Feed-Forward Network)

<span id="sec-2-6"></span> <strong class="critical-term">原始 Transformer 的 FFN (基于 ReLU)</strong>

- <strong class="list-label">结构</strong>：包含两个线性变换层，中间夹着一个 <strong class="critical-term"> ReLU 激活函数</strong>。

- <strong class="list-label">公式</strong>：$FFN(x) = Linear_2(ReLU(Linear_1(x)))$

- <strong class="list-label">维度</strong>：中间隐藏层（Linear_1 的输出）的维度 d_ff 通常是输入维度 d_model 的 4 倍 (d_ff = 4 \* d_model)。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 常见激活函数及对应场景</p>

<strong class="list-label">Sigmoid 函数</strong>

- <strong class="list-label">公式：</strong> $f(x) = \frac{1}{1 + e^{-x}}$

- <strong class="list-label">范围：</strong>(0, 1)

- <strong class="list-label">优点：</strong>

  - 输出平滑，适合于<strong class="critical-term"> 二分类问题</strong>。

  - 可以将输出映射到(0, 1)区间。

- <strong class="list-label">缺点：</strong>

  - <strong class="list-label">梯度消失：</strong>在输入值很大或很小时，导数接近0，导致更新缓慢。

  - 输出不以0为中心，可能导致优化效率下降。

<strong class="list-label">Tanh 函数</strong>

- <strong class="list-label">公式：</strong> $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

- <strong class="list-label">范围：</strong>(-1, 1)

- <strong class="list-label">优点：</strong>

  - 比Sigmoid更为平滑，输出以0为中心。

  - 解决了<strong class="critical-term"> Sigmoid的输出不以0为中心的问题</strong>。

- <strong class="list-label">缺点：</strong>

  - 仍然存在<strong class="critical-term"> 梯度消失</strong>的问题。

<strong class="list-label">ReLU 函数</strong>

- <strong class="list-label">公式：</strong> $f(x) = max(0, x)$

- <strong class="list-label">范围：</strong>\[0, +∞)

- <strong class="list-label">优点：</strong>

  - 计算简单，收敛速度快。

  - 在正区间内，梯度恒为1，避免了梯度消失问题。

- <strong class="list-label">缺点：</strong>

  - <strong class="list-label">梯度消失问题：</strong>对于负输入，梯度为0，可能导致“死亡神经元”现象。

<strong class="list-label">Softmax 函数</strong>

- <strong class="list-label">公式：</strong> $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$ 适用于<strong class="critical-term"> 多分类</strong>任务。

- <strong class="list-label">范围：</strong>输出值在(0, 1)之间，且所有输出的和为1。

- <strong class="list-label">优点：</strong>

  - 将输出转换为概率分布，适合于多分类问题。

</div>

<strong class="critical-term">SWiGLU</strong>

<strong class="list-label">SiLU (Sigmoid Linear Unit) / Swish 激活函数</strong>

- <strong class="list-label">定义：</strong> $SiLU(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$。

- <strong class="list-label">特点：</strong> 类似于 ReLU，但在零点附近是平滑的，这有助于缓解梯度消失问题，并允许负数输入通过。

<strong class="list-label">门控线性单元 (Gated Linear Units, GLUs)</strong>

- <strong class="list-label">定义：</strong> $GLU(x, W1, W2) = \sigma(W1x) \odot W2x$。

- 其中，$\odot$ 代表<strong class="critical-term"> 元素级乘法</strong>。e.g. $[1.227, -0.162] \odot [-1.6, 1.5] = [1.227×(-1.6), -0.162×1.5] = [-1.963, -0.243]$

- <strong class="list-label">特点：</strong> 将一个线性变换的结果通过 <strong class="critical-term"> sigmoid 函数</strong>（作为门控信号），再与另一个线性变换的结果进行逐元素相乘。

- <strong class="list-label">优点：</strong> 建议通过提供一个“线性路径”来“减少深度架构中的梯度消失问题”，同时保留非线性能力。

<strong class="list-label">对于 SwiGLU 的定义：</strong>

<div class="key-formula">

$$
SwiGLU(x, W_{1}, W_{2}, W_{3})=W_{2}\left(SiLU\left(W_{1} x\right) \odot W_{3} x\right)
$$

</div>

- $x \in \mathbb{R}^{d_{model }}$ 是输入向量（在实际中，通常是 (batch_size, sequence_length, d_model) 形状张量的最后一个维度）,

- $W_{1} , W_{3} \in \mathbb{R}^{d_{f 1} ×d_{model }}$ 是线性变换的权重矩阵,

- $W_{2} \in \mathbb{R}^{d_{model } ×d_{ff }}$ 是另一个线性变换的权重矩阵,

- $d_{ff}=\frac{8}{3} d_{model }$ , $d_{ff}$ 是中间隐藏层的维度，通常是 $d_{model}$ 的 8/3 倍（并确保是 64 的倍数）

### 2.4.3 相对位置嵌入(RoPE 旋转位置编码)

[参考资料 2.2](/part-1/chapter-2#read-2-2)

Transformer 中常见的位置编码包括<strong class="key-term"> 绝对位置编码</strong>和<strong class="key-term"> 相对位置编码</strong>两大类。

Rope 旋转位置编码就是属于相对位置编码，是一种经常被用在大模型（比如 LLaMA、chatGLM）里面的方式。

#### 2.4.3.1 为什么需要位置编码？

当我们想要利用 <strong class="key-term">Transformer 架构</strong>来训练一个大语言模型时

1.  首先会输入一段文本，或者通俗地讲是一句话，比如“我喜欢你”。

2.  这句话会经过已经训练好的分词器比如 <strong class="key-term"> BPE tokenizer</strong> 进行初步分词，转换为 token 数学向量形式；

3.  之后，token 会经过<strong class="key-term"> 词嵌入层（input embedding）</strong>，其神经网络输入输出形状为（batch_size,sequence_length,d_model）转换为<strong class="key-term"> 稠密向量 X</strong>；

4.  转换后的稠密向量 X 会输入到<strong class="key-term"> 自注意力机制</strong>中，具体的运算就是

    $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$ , 其中 Q(query)、K(key)、V(value) 是 X 分别经过 $W_q$,$W_k$,$W_v$ 三个神经网络之后的结果。

5.  那么实际上 $QK^T = W_qXX^TW_k$ 的运算主要依赖于原始的稠密向量 $XX^T$。而对于一句话来说，如果不去关注每一个字的<strong class="key-term">位置信息</strong>，只是关注字词内容本身，那么每个 token 所转换成的稠密向量 X 肯定是相同的，经过注意力机制后相同文本不同位置的语句结果也不会变化，但这是不符合实际情况的。

<div class="custom-block tip">

<p class="custom-block-title">例子</p>

<strong class="note-label">比如</strong>，“我喜欢你” → \[”我”，“喜欢”，“你”\] → \[\[1,2,3\],\[4,5,6\],\[7,8,9\]\]

这句话如果改为 ”你喜欢我”→\[”你”，“喜欢”，“我”\]→\[\[7,8,9\],\[4,5,6\],\[1,2,3\]\]

可以看到，虽然”我喜欢你”和”你喜欢我”这两句话语义有很大的不同，但最后转换成的稠密向量 X 在具体的值上没有变化，都是\[1,2,3,4,5,6,7,8,9\]。

我们引入位置编码的目标就是希望额外保留原有文本的位置（索引）信息，使得”你喜欢我”变成类似

\[\[74,86,96\],\[433,53,62\],\[12,42,63\]\]这种和”我喜欢你”生成的 X 有明显区别。

</div>

这个章节主要讲解相对位置编码下的 <strong class="critical-term">Rope（旋转位置编码）</strong>

#### 2.4.3.2 什么样的位置编码是好的位置编码

1.  <strong class="list-label">相对位置</strong>

2.  <strong class="key-term"> 外推性好</strong>

外推性好代表着如果你训练时最长文本是1000，但真实推理时是5000，也能很好适应。

#### 2.4.3.3 什么是旋转矩阵？

<strong class="key-term"> 旋转矩阵</strong>是在乘以一个向量的时候改变了向量的方向但不改变大小的效果并保持了<strong class="key-term"> 手性</strong>的矩阵。

<strong class="key-term">手性</strong>指左手右手坐标系，类似物理上的左手定则，右手定则；

#### 2.4.3.4 性质

设 $M$ 是任何维的一般旋转矩阵: $M \in \mathbb{R}^{n \times n}$

- <strong class="list-label"> 两个向量的内积在它们都被一个旋转矩阵操作之后保持不变:</strong> $a^T \cdot b=(Ma)^T \cdot Mb$

- <strong class="list-label">旋转矩阵的逆矩阵是它的转置矩阵:</strong> $MM^{-1}=MM^T=I$    这里的 $I$ 是单位矩阵。

- 若用 $M(\theta)$ 表示逆时针旋转的角度为 $\theta$ ,那么有 <span class="key-formula">$M(\alpha+\beta) = M(\alpha)M(\beta)$</span>

- 旋转矩阵的转置等于原矩阵角度取负，即 <span class="key-formula">$M(\theta)^T = M(-\theta)$</span>

- 一个矩阵是旋转矩阵，当且仅当它是正交矩阵并且它的行列式是1。正交矩阵的行列式是 $\pm 1$；如果行列式是 $-1$，则它包含了一个反射而不是真旋转矩阵。

直观上理解就是乘 $M(\alpha)$ 代表先旋转 $\alpha$, 再乘 $M(\beta)$ 代表再选择 $\beta$,合计旋转了 $\alpha + \beta$,等效于乘 $M(\alpha+\beta)$。

<figure data-latex-placement="H">
<img src="/images/61b6458b48.png" style="width:80.0%" />
</figure>

#### 2.4.3.5 二维空间下的旋转矩阵

在二维空间中，旋转可以用一个单一的角 $\theta$ 定义。作为约定，<strong class="key-term"> 正角表示逆时针旋转</strong>。把笛卡尔坐标的列向量关于原点逆时针旋转 $\theta$ 的矩阵是:

$M(\theta)=\left[\begin{array}{cc} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{array}\right]=\cos \theta\left[\begin{array}{ll} 1 & 0 \\ 0 & 1 \end{array}\right]+\sin \theta\left[\begin{array}{cc} 0 & -1 \\ 1 & 0 \end{array}\right]=\exp \left(\theta\left[\begin{array}{cc} 0 & -1 \\ 1 & 0 \end{array}\right]\right)$

<strong class="critical-term"> 欧拉公式：</strong> $e^{i\theta} = cos\theta + i sin\theta$

<strong class="critical-term"> 二维向量和复数域</strong>

一般的复数表示法为 $z = a + i b$，其中 $i$ 为复数单位 ，我们就可以根据实轴和虚轴对一个负数向量化表示，比如 $z_1 = 3+4 i \rightarrow (3,4)$

#### 2.4.3.6 将位置信息注入到稠密向量中

前面我们讲到，原始不加入位置信息的经过词嵌入层之后的稠密向量 X 存在着较大缺陷，因此，我们要在原始向量的基础上添加位置信息，即 $f(x_i,i)$, 其中 i 代表该元素的位置信息（索引）。

相对位置编码希望能利用上 token 之间的相对位置信息，假定 query 向量 $q_m$ 和 key 向量 $k_n$ 之间的内积操作可以被一个函数 g 表示，该函数 g 的输入是词嵌入向量 $x_m$ ， $x_n$ 和它们之间的相对位置 m−n ：<span class="key-formula">$<f_q(x_m,m),f_k(x_n,n)>=g(x_m,x_n,m−n)$</span>

#### 2.4.3.7 二维场景

在二维场景下，定义为

<span class="key-formula">$\begin{array}{l} f_{q}\left(\boldsymbol{x}_{m}, m\right)=\left(\boldsymbol{W}_{q} \boldsymbol{x}_{m}\right) e^{i m \theta} =q_me^{i m \theta}\\ f_{k}\left(\boldsymbol{x}_{n}, n\right)=\left(\boldsymbol{W}_{k} \boldsymbol{x}_{n}\right) e^{i n \theta}=k_ne^{i n \theta} \\ g\left(\boldsymbol{x}_{m}, \boldsymbol{x}_{n}, m-n\right)=\operatorname{Re}\left[\left(\boldsymbol{W}_{q} \boldsymbol{x}_{m}\right)\left(\boldsymbol{W}_{k} \boldsymbol{x}_{n}\right)^{*} e^{i(m-n) \theta}\right] \end{array}$</span>

其实和无位置编码的 Transformer 架构比起来的区别只是增加了 $e^{im\theta}$ 项

然后对 $QK^T$ （即求内积）= g 的推导如下所示：

- <strong class="list-label"> 方法一：</strong>暴力推导

  <figure data-latex-placement="H">
  <img src="/images/06d52dfd64.png" style="width:80.0%" />
  </figure>

- <strong class="list-label">方法二：</strong>利用旋转矩阵的性质

  对于 $f_q=q_me^{i m \theta}$ 可以理解为 $q_m$ 逆时针旋转的角度为 $m \theta$ ,即 $M(m \theta)$ 而 $f_k = k_ne^{i n \theta}$ 可以理解为 $k_n$ 逆时针旋转的角度为 $n \theta$ ,即 $M(n \theta)$

  那么两者内积$<f_q,f_k> = <M(m \theta) q_m, M(n \theta) k_n> = q_m^T M(m \theta)^T M(n \theta) k_n = q_m^T M(m \theta) M(n \theta) k_n = q_m^T M((n - m) \theta) k_n$ 之后仿照证明方法1也可得到 $g = <f_q,f_k> = q_m^T M((n - m) \theta) k_n$

#### 2.4.3.8 扩展至多维场景

实际的词嵌入维度（d_model）一般都不是二维的，不过处理的方式也不复杂，是<strong class="key-term"> 两两一分组</strong>，每一组还是上面二维场景下的操作。

$$
\left(\begin{array}{ccccccc}
\cos m \theta_{0} & -\sin m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_{0} & \cos m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_{1} & -\sin m \theta_{1} & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_{1} & \cos m \theta_{1} & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right)\left(\begin{array}{c}
q_{0} \\
q_{1} \\
q_{2} \\
q_{3} \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right)
$$

对于此，因为有很多零参与计算会很浪费，优化的方法是：

$\left(\begin{array}{c} q_{0} \\ q_{1} \\ q_{2} \\ q_{3} \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{array}\right) \otimes\left(\begin{array}{c} \cos m \theta_{0} \\ \cos m \theta_{0} \\ \cos m \theta_{1} \\ \cos m \theta_{1} \\ \vdots \\ \cos m \theta_{d / 2-1} \\ \cos m \theta_{d / 2-1} \end{array}\right)+\left(\begin{array}{c} -q_{1} \\ q_{0} \\ -q_{3} \\ q_{2} \\ \vdots \\ -q_{d-1} \\ q_{d-2} \end{array}\right) \otimes\left(\begin{array}{c} \sin m \theta_{0} \\ \sin m \theta_{0} \\ \sin m \theta_{1} \\ \sin m \theta_{1} \\ \vdots \\ \sin m \theta_{d / 2-1} \\ \sin m \theta_{d / 2-1} \end{array}\right)$

即对对应的元素相乘相加，这样做在代码里面可以用 arrange 函数处理了。

此外，对 RoPE 不是对所有维度都用同一个 $θ$ 进行旋转，而是给<strong class="key-term"> 不同维度分配不同的旋转速度（频率）</strong>。 将 $d_{model}$ 维的向量两两分组，共有 $\frac{d_{model}}{2}$ 个组。第 i 组（i from 0 to $\frac{d_{model}}{2}  - 1$）的旋转角度 $θ_i$ 定义为：$θ_i = 10000 ^ {(-\frac{2i}{d_{model}})}$

这里底数设置成10000是<strong class="key-term"> 经验数值</strong>，针对4096这类维度绰绰有余，实际上如果把底数设计的更大一些，比如50000，会有更好的<strong class="key-term"> 外推性</strong>，因为表示的频率范围会扩大。

#### 2.4.3.9 直观解释

由旋转矩阵的性质可知，原始 Q K 经过内积之后并不会改变原始绝对值大小，

Rope 编码最后结果 $q_m^T M((n - m) \theta) k_n$ 里有一项是 $(n-m)\theta$ ,这就说明了每一项注意力计算结果会和两个 token 的<strong class="key-term"> 相对位置 n-m 有关系</strong>。

直观上，位置离的越近的两个 token 按理说关联应该越强，比如“锦瑟无端五十弦”和后面一句”一弦一柱思华年”关系比较紧密，但和”望帝春心托杜鹃”逻辑性可能没那么紧密，token 关联就小一点，自注意力机制结果也应该小一些。经过位置编码的结果也类似，极端情况下,n=m即n-m=0两者同一个位置的时候，$cos(n-m)\theta$ 就是1，$-sin(n-m)\theta=0$,注意力最强。

#### 2.4.3.10 旋转位置编码的远程衰减性

直观上想，距离越远的 token 之间越不相干,也就是<strong class="key-term"> 注意力分数比较低</strong>，即 $QK^T$ 值比较低

我们知道由于 $QK^T$ 内积运算的结果包括周期函数（即$cos\theta$和$sin\theta$）,其最终求和的结果也一定是周期函数，而我们这里所讨论的远程衰减性实际上指一个周期就已经很长了，可能达到上万，而在这段区域内，内积的值是<strong class="key-term"> 振荡衰减</strong>的，所以我们称为远程衰减性。

证明有些复杂，可见：<https://zhuanlan.zhihu.com/p/647109286>

<figure data-latex-placement="H">
<img src="/images/2d3f359335.png" style="width:80.0%" />
</figure>

<strong class="critical-term">Rope homework</strong>

```python
    class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta #这个是RoPE的底数超参数，不是直接的角度
        self.d_k = d_k #d_k就是d_model,即嵌入之后的稠密向量，它必须为偶数
        self.max_seq_len = max_seq_len
        self.device = device
        #计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        #记录每个token的位置信息
        positions = torch.arange(self.max_seq_len)
        #计算正弦和余弦
        sinusoids = torch.outer(positions, freqs) #outer是外积，即每个位置都与每个频率相乘
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False) #利用register_buffer表示这是固定的，不需要学习
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 这里的x是输入的稠密向量，token_positions是token的位置信息
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        x1 = x[...,0::2] # 偶数位置
        x2 = x[...,1::2] # 奇数位置

        output1 = x1 * cos - x2 * sin # 偶数位置乘以cos，奇数位置乘以sin
        output2 = x1 * sin + x2 * cos # 偶数位置乘以sin，奇数位置乘以cos
        out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2]
        out = out.flatten(-2)  # [batch, seq_len, d_k]
        return out
```

<div class="custom-block tip">

<p class="custom-block-title">薇言大义</p>

Rope旋转位置编码是非常重要，面试中也经常遇见。

</div>

### 2.4.4 缩放点积注意力 scaled dot-product attention

<div class="key-formula">

$$
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$

</div>

- <strong class="list-label"> 定义：</strong><strong class="key-term"> 查询向量 Q </strong>和所有词的<strong class="key-term"> 键向量 K </strong>进行<strong class="key-term"> 点积（Dot-Product）运算</strong>。

- <strong class="list-label"> 为什么？</strong>点积是衡量两个向量相似度的一种常用方法。如果 Q 和某个 K 的方向很接近，它们的点积就会很大，代表<strong class="key-term"> 相关性高</strong>。Kᵀ 表示对Key矩阵进行转置，是为了让矩阵乘法能够顺利进行。

- <strong class="list-label"> 得到什么？</strong>一个<strong class="key-term"> 注意力分数</strong>（Attention Score）矩阵。这个矩阵的每一行表示一个词的Query，每一列表示它对句子中其他词的原始注意力分数。

<div class="custom-block tip">

<p class="custom-block-title">例子 · <strong class="key-term"> 为什么需要缩放</strong></p>

- <strong class="critical-term"> 问题：</strong>当 $d\_k$（Key向量的维度）比较大时，$Q \cdot K$ 的点积结果的<strong class="key-term"> 方差</strong>也会变大，这意味着点积的数值可能会非常大或非常小。

- <strong class="critical-term"> 后果：</strong>如果数值进入 Softmax 函数时过大，Softmax 的输出会趋近于“硬性”的 one-hot 分布（比如 \[0, 0, 1, 0, 0\]）。这意味着梯度会变得极小（<strong class="key-term"> 梯度消失</strong>），导致模型在训练时很难学习到有效的参数。

- <strong class="critical-term"> 解决方案：</strong>将点积结果除以 $\sqrt{d\_k}$。论文作者证明，这样做可以使得点积结果的方差保持在 1 左右，从而避免了上述问题，让训练过程更加稳定。

</div>

<div class="custom-block tip">

<p class="custom-block-title">例子 · <strong class="key-term"> Mask机制</strong></p>

对于有些我们不希望产生注意力的key，可以使用mask矩阵，在对应位置标上“False”或者其他的标识用来表示这个位置<strong class="key-term"> 不要产生注意力</strong>，之后在标记“False”的位置填充替换为<strong class="key-term"> “$-\infty$”</strong>，这样做的好处就是之后做softmax归一化时可以直接$e^{-\infty}=0$，忽略掉这一部分的注意力。

e.g. $Q^TK$相乘后的形状为 $R^{n \times m},$ 那么mask矩阵的形状也应该是 $R^{n \times m}$

$\begin{pmatrix}a_{11} & a_{12} & a_{13} \\a_{21} & a_{22} & a_{23} \\a_{31} & a_{32} & a_{33}\end{pmatrix} \otimes \begin{pmatrix} True & True & False \\ True & False & True \\ True & False & True \end{pmatrix}=\begin{pmatrix}a_{11} & a_{12} & -\infty \\a_{21} & -\infty & a_{23} \\a_{31} & -\infty & a_{33}\end{pmatrix}$

</div>

### 2.4.5 多头因果注意力机制

多头注意力的核心思想：从多个不同的<strong class="key-term"> 子空间</strong>中学习信息，让模型能够共同关注来自不同位置、不同维度的信息。

主要流程：原始的输入形状是(batch_size, seq_len, d_model)

1.  对于输入，要先用W_q, W_k, W_v 线性变换得到q,k,v

2.  如果有n_heads个头，就把最后一个维度切分成n_heads份，q,k,v每一部分都切分成 d_model//n_heads 的维度，

3.  对于每个头，对于q,k,v都去做attention操作。

4.  最后把所有的头按照最后一个维度concat起来，然后做一次线性变换。

$$
\operatorname{MultiHead}(Q, K, V)  =\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \operatorname{head}_{h}\right) \\
    \text { for } \operatorname{head}_{i}  =\operatorname{Attention}\left(Q_{i}, K_{i}, V_{i}\right)
$$

$$
\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x)
$$

## 2.5 Transformer 总结篇

这一小节是第一章最重要的部分，有别于原始论文里提出的模型流程，它从整体上讲解了一些新的主流大模型的底层 Transformer 架构（主要就是 <strong class="critical-term"> Llama</strong>，推荐直接阅读 Llama 的源代码，和作业重合度很高）。

<figure data-latex-placement="H">
<img src="/images/8d7f8c5b90.png" style="width:80.0%" />
</figure>

### 2.5.1 输入部分

模型的输入是<strong class="key-term"> 大规模</strong>的文本语料库。其核心训练目标是学习文本的统计规律，从而能够<strong class="key-term"> 基于给定的上文（context），准确地预测下一个词元（token）</strong>。通过这种方式，模型可以生成连贯、流畅的文本。

### 2.5.2 BPE tokenzier(预分词)

原始文本是由字符组成的字符串，无法直接输入神经网络。Tokenizer 的作用是<strong class="key-term"> 将原始文本字符串转换为一个整数序列（Integer IDs）</strong>。我们使用的 BPE (Byte-Pair Encoding) 是一种亚词 (subword) 分词算法，它通过以下步骤工作：

1.  <strong class="critical-term"> 初始化</strong>: 词典由所有单个字节 (0-255) 组成。

2.  <strong class="critical-term"> 迭代合并</strong>: 在训练语料中，不断寻找出现<strong class="key-term"> 频率最高的相邻字节对（或已合并的 token 对）</strong>，将它们合并成一个新的 token，并加入词典。

3.  <strong class="critical-term"> 最终</strong>: 经过指定次数的合并后，形成最终的词典 (vocabulary)。Tokenizer 利用这个词典和合并规则，将任意文本切分成一系列 token，并映射为其在词典中的<strong class="key-term"> 唯一整数 ID</strong>。

### 2.5.3 Embedding （词嵌入）

预分词得到的结果维度是vocab_size,即词典的大小，一般会用<strong class="critical-term"> 独热码</strong>来表示，这样做的稀疏向量太多，太浪费计算资源。为了把稀疏向量映射为<strong class="key-term"> 稠密向量</strong>，维度一般是 d_model=512。一个词嵌入层（embedding layer）的作用就是把这些离散的、没有语义的整数ID，转换为连续的、包含语义信息的<strong class="key-term"> 向量</strong>。比如，“猫”的向量可能和“狗”的向量在某种“动物”维度上比较接近，而和“汽车”的向量相距较远。每个嵌入层接收形状为 (batch_size, sequence_length) 的整数张量，并生成形状为 (batch_size, sequence_length, d_model) 的向量序列。d_model 代表输出维度，越大代表语义信息越丰富，模型越强大。

### 2.5.4 Position encoding（位置编码）

在原始论文中，位置编码被放在 embedding 之后，但在作业和主流大模型实现里面，位置编码就直接放在多头注意力那里来只处理 query 和 key 了。

位置编码是由于仅仅只是词嵌入虽然能够保留语义，但不能很好地表示每个 token 的位置（索引）信息。位置编码包括绝对位置编码和相对位置编码两大类，一般相对位置编码的效果会更好，作业中使用的方法是 rope 旋转位置编码。

### 2.5.5 Transformer block

作业中的 Transformer block 由 RMSnorm、Causal 多头注意力机制、SwiGlu 这几部分组成。

### 2.5.6 RMSnorm

简化版的层归一化

$RMSnorm(x)=γ⊙\frac{x}{RMS(x)}$ $RMS(x)=\sqrt{\frac{1}{D}∑_{i=1}^{D}x_i^2}=\sqrt{mean(x^2)}$

值得注意的是，对于从embedding 层输入过来的 (batch_size, sequence_length,d_model(或者 hidden_dim))，归一化的目标 x 是最后一维的 d_model 。

### 2.5.7 Causal Multi-head self-attention（多头注意力机制）

1.  <strong class="critical-term"> 多头机制 (Multi-Head)</strong>: 将 d_model 维的 Q, K, V 向量按照 n_heads 分成 d_model//n_heads 多个头 (head)，让每个头关注输入序列的不同方面，增强了模型的表达能力，最后再拼接在一起。

2.  <strong class="critical-term"> 旋转位置编码 (RoPE)</strong>: 为了注入位置信息，RoPE <strong class="key-term"> 直接作用于 Q 和 K 向量</strong>，通过旋转它们来编码其绝对位置，并使注意力得分自然地依赖于它们的相对位置。V 向量不参与位置编码。

3.  <strong class="critical-term"> 因果 Mask (Causal Masking)</strong>: 在自回归的文本生成任务中，模型在预测位置 i 的词元时，<strong class="key-term"> 只能关注到位置 i 及之前的所有词元</strong>，不能“偷看”未来的信息。这是通过在注意力得分矩阵上应用一个上三角遮罩实现的。”

4.  还有在点积缩放运算里面使用mask来给e的指数赋值$-\infty$，方便运算。

### 2.5.8 SwiGLU

1.  <strong class="critical-term"> 原始 Transformer 的 FFN (基于 ReLU)</strong>:

    - <strong class="list-label"> 结构</strong>: 包含两个线性变换层，中间夹着一个 ReLU 激活函数。

    - <strong class="list-label"> 公式</strong>: FFN(x) = Linear_2(ReLU(Linear_1(x)))

    - <strong class="list-label"> 维度</strong>: 中间隐藏层（Linear_1 的输出）的维度 d_ff 通常是输入维度 d_model 的 4 倍 (d_ff = 4 \* d_model)。

2.  <strong class="critical-term"> SiLU (Sigmoid Linear Unit) / Swish 激活函数</strong>:

    - <strong class="list-label"> 定义</strong>: $SiLU(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$ 其中，$\sigma(x)=\frac{1}{1+e^{-x}}$ 代表sigmoid激活函数。

    - <strong class="list-label"> 特点</strong>: 类似于 ReLU，但在零点附近是平滑的，这有助于缓解梯度消失问题，并允许负数输入通过。

3.  <strong class="critical-term"> 门控线性单元 (Gated Linear Units, GLUs)</strong>:

    - <strong class="list-label"> 定义</strong>: $GLU(x, W1, W2) = σ(W1x) ⊙ W2x$。

    - $\sigma$ 代表 Sigmoid 激活函数。

    - $\odot$ 代表元素级乘法 (element-wise multiplication)。

    - <strong class="list-label"> 优点</strong>: 建议通过提供一个“线性路径”来“减少深度架构中的梯度消失问题”，同时保留非线性能力。

### 2.5.9 最终输出

从几个 Transformer block 输出的形状还是(batch_size, sequence_length，d_model) ，此后继续归一化，Linear，Softmax。

归一化自然就是稳定梯度

Linear是将(batch_size, sequence_length, d_model)—\>(batch_size, sequence_length, vocab_size)

之后利用softmax进行计算vocab_size里面每一个词的概率，选择最大的词作为下一个词。

<div class="custom-block tip">

<p class="custom-block-title">薇言大义</p>

这一部分内容可以复习或者一开始就看一看，有一个整体的认识。

</div>

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 2.1 · 基础模块:线性层和嵌入层

<span id="read-2-1"></span>

He初始化：<https://zhuanlan.zhihu.com/p/40175178> 《神经网络与深度学习》 邱锡鹏 相关章节(讲的最好)

### 2.2 · 相对位置嵌入(RoPE 旋转位置编码)

<span id="read-2-2"></span>

RoPE旋转位置编码相关知识:

- <https://www.bilibili.com/video/BV1CQoaY2EU2?spm_id_from=333.788.player.player_end_recommend_autoplay&vd_source=453c2363ee43bdaa84f759f243a88819>

- <https://www.bilibili.com/video/BV1Mj421R7JQ/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=453c2363ee43bdaa84f759f243a88819>

- <https://zhuanlan.zhihu.com/p/647109286>

- <https://zhuanlan.zhihu.com/p/642884818>
