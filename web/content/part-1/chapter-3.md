---
outline: [2, 3]
---

# 第 3 章 · 语言模型的训练

<span id="guide-ch-3"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[Transformer 架构](/part-1/chapter-2#guide-ch-2)的 logits 输出；掌握概率、对数及链式法则，参见[数学与符号](/appendix#app-math)；信息论基础见[信息论前置知识](/appendix#app-information)。

<strong>准备工作：</strong>准备小模型、已编码的训练样本与独立验证样本；固定初始化和批量设置。

<strong>本章任务：</strong>实现稳定的交叉熵、AdamW、学习率调度与梯度裁剪；结合信息论理解训练损失与困惑度。

</div>

前面的章节讨论的是Transformer模型的架构和理论知识,这一章节侧重讲解Transformer模型的训练方法(事实上对于绝大多数深度学习的训练都适用)。主要涉及:

- <strong class="critical-term">损失函数</strong>:需要定义交叉熵损失函数

- <strong class="critical-term">优化器</strong>:需要定义AdamW优化器来最小化损失

- <strong class="critical-term">训练循环</strong>:需要构建完整的基础设施来加载数据、保存检查点和管理训练过程

<strong class="note-label">前置知识：</strong>自信息、熵、联合熵、条件熵与互信息的定义和示例见[信息论前置知识(简易版)](/appendix#app-information)。

## 3.1 交叉熵(Cross-Entropy)损失与KL散度

<span id="sec-3-2"></span>

[参考资料 3.1](/part-1/chapter-3#read-3-1)

### 3.1.1 KL散度(相对熵)

<span id="sec-3-1-1"></span> 假设随机变量 X 的真实概率分布为 P(X),而我们在处理实际问题时使用了一个近似的分布 Q(X) 来进行建模。由于我们使用的 Q(X) 是而不是真实的,所以我们在具体化的取值时需要一些<strong class="key-term">附加的信息</strong>来抵消分布不同造成的影响。我们需要的平均附加信息量可以使用<strong class="key-term">相对熵</strong>,或者叫<strong class="critical-term">KL散度(Kullback-Leibler Divergence)</strong>来计算,KL散度可以用来<strong class="key-term">衡量两个分布的差异</strong>:

$D_{KL}(P||Q)=-\sum_{i=1}^nP(x_i)logQ(x_i)-(-\sum_{i=1}^nP(x_i)logP(x_i))=\sum_{i=1}^nP(x_i)log\frac{P(x_i)}{Q(x_i)}$

KL散度的性质:

1.  KL散度不是一个对称量 $D_{KL}(P||Q) \ne D_{KL}(Q||P)$

2.  KL散度永远大于等于0(因为最好也就是 $Q(X)=P(X)$,此时额外信息量为0)

### 3.1.2 交叉熵

<span id="sec-3-1-2"></span>

实际上就是 KL 散度的第一项值

$D_{KL}(P||Q)=-\sum_{i=1}^nP(x_i)logQ(x_i)-(-\sum_{i=1}^nP(x_i)logP(x_i)) = -H(P) + H(P,Q)$

<span class="key-formula">$H(P, Q)=H(P)+D_{K L}(P \| Q)=-\sum_{i=1}^{n} P\left(x_{i}\right) \log Q\left(x_{i}\right)$</span>

对于真实分布 $P(X)$ 来讲,它的熵 $H(P)$ 是一个固定值,<strong class="key-term">真正决定 KL 散度值的还是交叉熵</strong>。

我们训练神经网络的目标就是希望近似分布 $Q$ 逼近真实分布 $P$。

### 3.1.3 交叉熵损失函数

<span id="sec-3-1-3"></span> <strong class="key-term">二分类</strong>

只有两类,设:

- 真实标签为 $y \in \{0, 1\}$

- 模型输出的预测概率为 $\hat{y} \in [0, 1]$

则交叉熵损失函数为:

$L=-\left[y_{i} \log \left(\hat y_{i}\right)+\left(1-y_{i}\right) \log \left(1-\hat y_{i}\right)\right]$

```python
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_pred, y_true)
```

<strong class="key-term">多分类</strong>

设:

- 真实标签 $y \in \{0,1,...,K-1\}$ 是类别索引

- 模型预测的是一个概率分布向量 $\hat{y} = [p_1, ..., p_K]$

一般来讲,$p_i$ 是要经过 softmax 处理的,即 $p_{i} = \frac{e^{x_{i}}}{\sum_{k=0}^{C-1} e^{x_{k}}}$

多分类交叉熵公式为:

<span class="key-formula">$\text{CrossEntropy Loss} = -\sum_{i=1}^K y_i \log(\hat{y}_i)$</span>

```python
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)# 注意 torch 里面会自动计算softmax,所以不需要人为再去计算一遍了
```

### 3.1.4 实际代码的应用

<span id="sec-3-1-4"></span>

Exp-normalize 技巧 + LogSoftmax + NLLloss

#### 3.1.4.1 <strong>Exp-normalize 技巧</strong>

<span id="sec-3-1-5"></span>

Exp-normalize 技巧的目标是为了实现后续 softmax 计算数值的稳定性,避免出现上溢或者下溢的情况。具体来说,

先将输入减去向量最大值:$z = x - max(x)$

这样做,所有的$x_i \subseteq (-\infty,0]$,也就是 $e^x$ 的值域只会在 y 轴左半部分,这样做的好处是:

- 最大的值变为 0,避免其可能原始数值很大,导致上溢指数爆炸；

- 至少有一个 exp(0)=1,防止softmax里的分母为 0 下溢。

- 这种变换不改变最终 softmax 的输出,因为指数任意平移不改变比值。

$\operatorname{softmax}(x){i}=\frac{e^{x{i}}}{\sum_{j} e^{x_{j}}}=\frac{e^{x_{i}-m}}{\sum_{j} e^{x_{j}-m}}=\operatorname{softmax}(z)_{i}$

<figure data-latex-placement="H">
<img src="/images/dac7819b26.png" style="width:80.0%" alt="Exp-normalize 技巧" />
<figcaption>Exp-normalize 技巧</figcaption>
</figure>

#### 3.1.4.2 <strong>LogSoftmax</strong>

<span id="sec-3-1-6"></span>

实际上就是把log和softmax两种运算给组合起来了。

- <strong class="list-label">softmax运算:</strong> $softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}$

- <strong class="list-label">LogSoftmax 运算:</strong> $\log \frac{e^{x_{i}}}{\sum_{j} e^{x_{j}}}=x_{i}-\log \sum_{j} e^{x_{j}}$

直接使用Logsoftmax而不是分步使用 log 和 softmax 的好处在于:

- 减少计算开销,直接使用了 $x_i$ 等价先 $softmax(x_i)$ 再 log

- 直接使用 Logsoftmax 计算梯度会更加稳定

#### 3.1.4.3 <strong>NLLloss负对数似然损失函数</strong>

<span id="sec-3-1-7"></span>

公式:$\text{nllloss} = -\frac{1}{N} \sum_{k=1}^{N} P(y_k) \cdot (\log\_\text{softmax})$

其实就是Logsoftmax 后续的补充操作,使得结果与 Crossentropy 结果一致。

```python
logp = F.log_softmax(logits, dim=1)    # 计算 log 概率:shape = [batch, K],    
loss = F.nll_loss(logp, targets)       # targets 是 [batch] 维度的真实类索引
```

这两行代码等价于直接用crossentropy

```python
loss = F.cross_entropy(logits, targets)
```

### 3.1.5 具体在Transformer中的应用

<span id="sec-3-1-8"></span> Transformer作为一个<strong class="key-term">自回归的语言模型</strong>,其核心任务是<strong class="key-term">预测下一个词</strong>。给定一个词序列的前i个词 $x_{1:i}$,它需要预测出第i+1个词 $x_{i+1}$ 的概率分布 $p_\theta(x_{i+1} | x_{1:i})$。这里的$\theta$代表模型的所有参数(权重)。

<span class="key-formula">$\ell(\theta ; D)=\frac{1}{|D| m} \sum_{x \in D} \sum_{i=1}^{m}-\log p_{\theta}\left(x_{i+1} \mid x_{1: i}\right)$</span>

$\ell(\theta ; D)$ 就是这个损失函数。我们来把它拆解一下:

- $|D|$ 是数据集中序列的数量。

- $m$ 是我们考虑的序列长度。

- $\sum_{x \in D} \sum_{i=1}^{m}$ 这个双重求和符号代表我们要计算数据集中每一条序列里,从第1个位置到第m个位置的每一次预测的损失,然后把它们全部加起来。

- $-\log p_{\theta}(x_{i+1} | x_{1:i})$ 这是整个损失函数的核心。

- $p_{\theta}(...)$ 是模型预测的、下一个词恰好是正确答案 $x_{i+1}$ 的概率。

- <strong class="key-term">我们的目标是让这个概率尽可能高,理想情况下趋近于1。</strong>

- $\log$函数在(0, 1\]区间内是单调递增的,所以让$p$最大化等同于让$\log(p)$最大化。

- 加上一个负号 -,最大化 $\log(p)$ 就变成了最小化 $-\log(p)$。这正是机器学习中梯度下降的目标:最小化损失函数。

- $\frac{1}{|D|m}:$ 最后除以总的预测次数(序列数 × 序列长),这是为了计算平均损失,避免数据集大小对损失值产生影响。

模型在内部并不会直接输出概率,而是先输出一个叫做<strong class="critical-term">logits</strong>的原始数值向量$o_i$。这个向量的维度等于你的词汇表大小(vocab_size),其中每个元素可以看作是对应单词的<strong class="key-term">“得分”或“置信度”</strong>,这个得分可以是任意实数(可正可负)。

为了把这些原始得分$o_i$转换成一个合法的概率分布(即所有值都在0到1之间,且加起来等于1),我们使用<strong class="key-term">Softmax</strong>函数。

$p(x_{i+1} | x_{1: i})=\operatorname{softmax}(o_{i})[x_{i+1}]=\frac{\exp (o_{i}[x_{i+1}])}{\sum_{a=1}^{\text {vocabsize }} \exp(o_{i}[a])}$

这个公式展示了如何计算正确词$x_{i+1}$的概率。

- 分子 $\exp \left(o_{i}\left[x_{i+1}\right]\right)$: 对正确词的logit分数取指数,确保其为正数。

- 分母 $\sum_{a=1}^{\text {vocabsize }} \exp \left(o_{i}[a]\right)$: 对词汇表中所有词的logit分数取指数然后求和,这是一个归一化项。

- 两者相除,就得到了正确词$x_{i+1}$的最终概率。

<strong class="list-label">总结一下整个流程:</strong>

对于训练数据中的一个序列,在每一个位置i:

- Transformer模型接收前面的序列$x_{1:i}$作为输入。

- 模型输出一个大小为vocab_size的logits向量$o_i$。

- 通过Softmax函数将$o_i$转换为概率分布。

- 取出真实下一个词$x_{i+1}$对应的概率,取负对数得到该位置的损失$-\log(p)$。

- 将所有位置、所有序列的损失加起来求平均,得到最终的总损失,然后通过反向传播来更新模型参数θ以最小化这个损失。

### 3.1.6 Perplexity(困惑度)

<span id="sec-3-1-3"></span> <strong class="key-term">Perplexity(困惑度)</strong> 是<strong class="key-term">衡量语言模型预测能力的指标</strong>,它反映了模型对测试数据的不确定性。<strong class="key-term">Perplexity 越低,说明模型对数据的预测越准确；Perplexity 越高,说明模型对数据的预测越不确定</strong>。

#### 3.1.6.1 Perplexity 的数学定义

Perplexity(PP)与交叉熵(Cross-Entropy)密切相关,数学定义如下:

<span class="key-formula">$PP(W) = 2^{H(W)}$</span>

其中,<strong class="key-term">$H(W)$ 是语言模型的交叉熵</strong>,计算公式如下:

$H(W) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, w_2, ..., w_{i-1})$

- <strong class="list-label">W</strong>:一整段文本

- <strong class="list-label">N</strong>:文本中的单词总数

- <strong class="list-label">$P(w_i | w_1, ..., w_{i-1})$</strong>:模型对单词 $w_i$ 在给定前面单词的条件下的预测概率

<strong class="note-label">直观理解</strong>

1.  <strong class="list-label">如果模型完美预测每个单词的概率都是 1</strong>,那么 <strong class="key-term">PP = 1</strong>,即模型完全不困惑。

2.  <strong class="list-label">如果模型随机猜测(每个单词概率相等)</strong>,那么 <strong class="key-term">PP 会很高</strong>,表示模型的预测不确定。

#### 3.1.6.2 Perplexity 的意义

<strong class="key-term">衡量语言模型的好坏</strong>:

1.  <strong class="list-label">低 Perplexity(PP 低)</strong> → 说明模型更擅长预测下一个单词,效果更好。

2.  <strong class="list-label">高 Perplexity(PP 高)</strong> → 说明模型对语言的理解较差,需要优化。

<strong class="key-term">用于对比不同模型的性能</strong>:

- 例如,GPT-3 的 Perplexity 低于 GPT-2,说明 GPT-3 在相同数据集上的预测能力更强。

<strong class="key-term">优化语言模型的目标</strong>:

- 训练过程中,优化目标通常是<strong class="key-term">最小化交叉熵(降低 Perplexity)</strong>,使模型更擅长语言理解和生成。

## 3.2 Optimizer: 优化器

<span id="sec-3-3"></span>

[参考资料 3.2](/part-1/chapter-3#read-3-2)

Optimizer(优化器)是深度学习中的一种核心算法,其主要作用是根据模型的损失函数(Loss Function)计算出的梯度(Gradient),来指导并更新网络中的参数(如权重和偏置),目标是让损失函数的值越来越小,从而使模型的预测结果越来越准确。

常见的Optimizer包括:<strong class="key-term">SGD、Momentum、Adagrad、RMSprop、Adam等</strong>。

### 3.2.1 SGD (随机梯度下降)

<span id="sec-3-3-1"></span>

SGD 是一般的梯度下降法的近似版本,传统的梯度下降法是针对<strong class="key-term">整个训练集</strong>进行求梯度,而 SGD(mini SGD) 是对训练集的<strong class="key-term">一小批量</strong>进行求梯度并更新。

<strong class="key-term">公式(和梯度下降法一样):</strong>

<span class="key-formula">$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$</span>

其中:

- $\theta_t$:第 t 步的参数(或者权重)；

- $\alpha$:学习率；

- $\nabla_\theta J(\theta_t)$:损失函数对参数的梯度。$\nabla_\theta J(\theta_t)=\partial_{\mathbf{w}} \frac{1}{\left|\mathcal{B}_{t}\right|} \sum_{i \in \mathcal{B}_{t}} f\left(\mathbf{x}_{i}, \boldsymbol{\theta}\right)$ 其中 $\mathcal{B}_{t}$ 代表这个批次

相较于 GD ,SGD <strong class="key-term">计算损耗更少</strong>,但梯度的估计存在噪声,导致收敛过程会发生震荡,并且<strong class="key-term">不保证能收敛到全局最优点</strong>。

### 3.2.2 动量法(Momentum)

<span id="sec-3-3-2"></span>

直接利用SGD的话,可能出现相邻两个时刻梯度差距过大,来回<strong class="key-term">震荡</strong>,动量法利用了上一时刻的梯度值,在 SGD 的基础上引入“动量”,即<strong class="key-term">历史梯度的线性组合(加权平均)</strong>,以缓解震荡、加速收敛。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · <strong class="key-term"> 为什么动量法要叫动量法？</strong></p>

这是从山坡上最快降到谷底这一物理过程上类比得出来的,实际上叫“惯性法”更加合适。

对于梯度法,它没有考虑历史时刻的速度也就是惯性,而动量法通过将当前梯度与上一时刻的速度进行<strong class="key-term">加权平均</strong>的方式利用了惯性。打个比方,一个小球从山坡上降到谷底,梯度法不考虑它的速度,没到一个地方就计算这个地方的梯度,那么在接近谷底的陡峭点(梯度高),那么小球就直接冲到山谷另一侧了,而如果我们保留了历史的速度,在这个位置冲到另一侧时就会由于动量的存在冲的没那么厉害,减少震荡。

</div>

<strong class="key-term">公式:</strong>

$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta_t)$

$\theta_{t+1} = \theta_t - \alpha v_t$

其中:

- $v_t$:动量变量,表示历史梯度的指数加权平均；

- $\beta$:动量衰减系数(惯性系数),常取 0.9。

#### 3.2.2.1 动量法的两大好处

1.  <strong class="list-label">在梯度方向一致时加速</strong>:

    在山谷的一侧坡上,梯度方向基本不变(一直指向谷底)。每一时刻的梯度(加速度) $J(\theta_t)$ 都会和上一时刻的速度 $v_{t-1}$ 方向<strong class="key-term">相同</strong>,速度 $v_t$ 就会不断累积,变得越来越大,从而让参数 $θ$ 的更新步伐越来越大,实现<strong class="key-term">加速</strong>效果。

2.  <strong class="list-label">在梯度方向震荡时抑制</strong>:

    在山谷的陡峭两侧,梯度方向会来回变化 ($J(\theta_t)$ 和 $J(\theta_{t-1})$ 方向相反)。这时,$J(\theta_t)$ 会和  $v_{t-1}$ 中的震荡方向分量相互<strong class="key-term">抵消</strong>,使得 $v_t$ 在这个方向上的值变小。这就起到了<strong class="key-term">抑制震荡</strong>、平滑更新路径的作用。

但相较于 GD、SGD来说,动量法需要<strong class="key-term">额外维护一个与模型参数量等大的动量变量 $v_{t-1}$</strong>,其空间复杂度为 O(N),N为模型参数数量。

<figure data-latex-placement="H">
<img src="/images/5690d449af.png" style="width:80.0%" alt="动量法示意图" />
<figcaption>动量法示意图</figcaption>
</figure>

### 3.2.3 AdaGrad

<span id="sec-3-3-3"></span>

核心思想:为每一个参数分配<strong class="key-term">不同</strong>的<strong class="key-term">自适应学习率</strong>

像 GD、SGD 这些算法的学习率 $\alpha$ 一般都是固定的,AdaGrad 认为有些参数更新频繁,有些很少更新,不同的参数应根据其<strong class="key-term">历史梯度</strong>有不同学习率

<strong class="key-term">公式:</strong>

$G_t = G_{t-1} +∇g_t ⨀∇g_t$

$x_t=x_{t−1}−\frac{\alpha}{ϵ+G_t}⨀∇g_t$

其中:

- $g_t$:第 t 步的梯度；

- $G_t$:从第 1 步到第 t 步,梯度平方的累加；

- $⨀$ 表示<strong class="key-term">逐元素相乘</strong>

<strong class="key-term">e.g.:</strong>

假设在二维平面上的梯度g=\[4,9\],初始时 G=0,则 $G_1$ =\[16,81\],我们对第一维度进行更新时,采用 $x_t^{(1)}=x_{t-1}^{(1)}-\frac{\eta}{\epsilon+\sqrt{16}}\times4$

对第二维度进行更新时,采用 $x_t^{(2)}=x_{t-1}^{(2)}-\frac{\eta}{\epsilon+\sqrt{81}}\times9$

从上面我们就可以看出,当梯度的某一个方向上比较大时,分母会约束其变小,而当梯度的某一个方向上比较小时,分母会令其相对变大。从中其实可以体现出归一化的效果,有点类似于下图的转换结果:

<figure data-latex-placement="H">
<img src="/images/9bc2f96548.png" style="width:80.0%" alt="AdaGrad示意图" />
<figcaption>AdaGrad示意图</figcaption>
</figure>

AdaGrad 的一个重大缺点是 $G_t$ 是<strong class="key-term">无上限累计的</strong>,这样到后期学习率基本上单调递减到0了,就不会训练了。

### 3.2.4 RMSProp

<span id="sec-3-3-4"></span>

为了克服 AdaGrad 的缺点,RMSProp 不是直接累加所有历史梯度,而是使用<strong class="key-term">滑动窗口式的加权平均</strong>,只“记住<strong class="key-term">最近</strong>的梯度行为”。

设梯度为 $g_t$,均方梯度为 $E[g^2]_t$:

1.  <strong class="list-label">估计梯度平方的指数平均:</strong>

    $E[g^2]_t = \beta \cdot E[g^2]_{t-1} + (1 - \beta) \cdot g_t^2$

2.  <strong class="list-label">更新参数:</strong>

    $\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t$

其中:

- $g_t^2$:表示梯度按元素平方的值

$\begin{aligned} E\left[g^{2}\right]_{1} & =(1-\gamma) g_{1}^{2} \\ E\left[g^{2}\right]_{2} & =\gamma E\left[g^{2}\right]_{1}+(1-\gamma) g_{2}^{2}=\gamma(1-\gamma) g_{1}^{2}+(1-\gamma) g_{2}^{2} \\ E\left[g^{2}\right]_{3} & =\gamma E\left[g^{2}\right]_{2}+(1-\gamma) g_{3}^{2} \\ & =\gamma^{2}(1-\gamma) g_{1}^{2}+\gamma(1-\gamma) g_{2}^{2}+(1-\gamma) g_{3}^{2} \\ \ldots & \\ E\left[g^{2}\right]_{t} &= (1-\gamma)\sum_{i=1}^{t}\gamma^{t-i} g_{i}^{2} \end{aligned}$

最近的梯度平方权重大,<strong class="key-term">越早之前的梯度影响越小</strong>(指数级衰减).

#### 3.2.4.1 直观理解

1.  RMSProp 旨在估计一个“近似<strong class="key-term">局部</strong>梯度方差”；

2.  如果梯度波动很大(即平方值变化大),那 $E[g^2]_t$ 就会大；

3.  RMSProp 用这个值来<strong class="key-term">对当前梯度进行缩放</strong>,让不同参数维度的学习率自适应:

$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t$ 这意味着:

- 如果某个参数维度的梯度在近期波动很大(不稳定),我们就<strong class="key-term">减少它的学习率</strong>；

- 如果某个维度的梯度变化小(较稳定),就<strong class="key-term">保持或稍大更新步幅</strong>。

<strong class="key-term">超参数的取值:</strong>

- $\beta$ 取值:

  - 接近 1(如 0.99):长记忆(依赖更长的历史)

  - 接近 0(如 0.5):短记忆(几步就忘)

  - 通常设置为 0.9:平衡短期与长期趋势

### 3.2.5 Adam

<span id="sec-3-3-5"></span>

Adam 是把之前的优化器的特点都融合起来了,不仅<strong class="key-term">看当前位置的梯度</strong>(就像普通SGD那样),还<strong class="key-term">考虑过去梯度的平均值</strong>(动量)和<strong class="key-term">梯度平方的平均值</strong>(像RMSProp那样),并进行<strong class="key-term">自适应的学习率调整</strong>,从而在不同参数维度上使用不同的更新步长。

假设我们在训练中第 t 步,梯度为 $g_t$。Adam的更新公式如下:

1.  <strong class="list-label">一阶矩估计(动量)</strong>:

    $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$

2.  <strong class="list-label">二阶矩估计(RMS)</strong>:

    $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$

3.  <strong class="list-label">偏差校正</strong>:

    $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

4.  <strong class="list-label">参数更新</strong>:

    <span class="key-formula">$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$</span>

其中:

\- θ 为当前模型参数 - $g_t$ 为当前梯度(损失对参数的导数) - $m_t$ 为梯度的指数加权平均(动量) - $v_t$ 为梯度平方的指数加权平均 - $\beta_1$ 控制动量的衰减率,常设为0.9 - $\beta_2$ 控制梯度平方平均的衰减率,常设为0.999 - $\alpha$ 代表学习率,默认0.001 - $ϵ$ 是防止除以0的微小常数,通常设为 $10^{-8}$

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · <strong class="key-term">为什么要进行偏差矫正？</strong></p>

由于 $v_t$ 是从 $v_0 = 0$ 开始的,所以在前几步迭代时,$v_t$ 的值明显偏小(因为它平均值主要还在积累阶段)。 例如第1步: $v_1 = (1 - \beta) g_1$ 这个值太小了,因为历史还没积累。 于是,我们用一个“修正系数”将它还原为更合理的期望值: $\hat{v}_t = \frac{v_t}{1 - \beta^t}$

其中:

- $v_t$ 是带偏的动量估计；

- $\hat{v}_t$ 是<strong class="key-term">无偏估计</strong>(bias-corrected)；

- $1 - \beta^t$ 是第 t 步的修正系数。

</div>

随着 t 增大,$\beta^t$ 趋近于 0,修正项变为 1,所以只在初期有显著影响。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

<strong class="key-term">因为 $v_0=0$不是一个真实的估计,它肯定比现实值要小,所以$v_t$是被低估了。</strong>

而由上面的推导我们知道 $E\left[g^{2}\right]_{t} = (1-\gamma)\sum_{i=1}^{t}\gamma^{t-i} E[g_{i}^{2}]$,

假设每一步期望采样独立同分布,令$E[g_{i}^{2}] = \mu$

因此有 $E[v_t]=(1−β)\sum_{i=1}^tβ^{t−i}E[g_i^2]$ $=(1−β)\sum_{i=1}^tβ^{t−i}μ$ $=(1−β)μ\sum_{i=1}^tβ^{t−i}$ $=(1−β)μ\sum_{j=0}^{t-1}β^j(令 j=t−i)$ $=(1−β)μ⋅\frac{1−β^t}{1−β}$

所以有 $E[v_t]=E[g^2]⋅(1−β^t)$ 因此误差估计的修正项就是 $\frac{1}{1-\beta^t}$ 最终的 $\hat{v}_t$ 是无偏估计

</div>

### 3.2.6 AdamW

<span id="sec-3-3-6"></span>

在原始的 Adam 中,我们常常使用 <strong class="key-term">L2 正则项</strong> 来防止过拟合,形式是:$L(\theta) + \frac{\lambda}{2} \|\theta\|^2$ 这就会带来一个额外的梯度项 $\lambda \cdot \theta$,被加入到梯度更新中。

所以,很多框架(如 PyTorch)默认会把它<strong class="key-term">加到梯度上</strong>:

$g_t^{\text{new}} = g_t + \lambda \cdot \theta$

然后再执行 Adam 的更新逻辑,这样附加项会对原有正常结果产生影响。

AdamW 就是希望<strong class="key-term">将正则项“从梯度更新中剥离”,变成一个独立的权重衰减项。</strong>

即:

<span class="key-formula">$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t \right)$</span>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

<strong class="note-label">注意:</strong>weight decay(L2衰减)是直接对参数 $\theta_t$ 进行缩小,而不是对梯度进行修改！

</div>

### 3.2.7 Muon

<span id="sec-3-3-7"></span> Muon 优化器是近期比较流行的一种优化器，在Deepseek v4 的训练中被应用。与 Adam 等优化器是逐元素地缩放梯度的操作不同, <strong class="critical-term">Muon 把权重矩阵当作矩阵来处理</strong>:对动量更新后的矩阵做<strong class="key-term">正交化</strong>。

#### 3.2.7.1 混合 Newton-Schulz 迭代

对于给定的矩阵 $M$，假设其<strong class="key-term">奇异值分解</strong>为：

$$
M=U\Sigma V^T
$$

Newton-Schulz 迭代的目的是将 $M$ <strong class="key-term">近似正交化</strong>为 $UV^T$。通常，首先会将 $M$ 归一化为 $M_0=M/\lVert M\rVert_F$（注：$\lVert M\rVert_F$ 代表 <strong class="key-term">Frobenius 范数</strong>，即每个元素的根号平方和），以确保其最大奇异值不超过 $1$。

<div class="key-formula">

$$
\lVert A\rVert_F=\sqrt{\sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}^2}
$$

$$
M_k=aM_{k-1}+b(M_{k-1}M_{k-1}^{T})M_{k-1}
+c(M_{k-1}M_{k-1}^{T})^2M_{k-1}
$$

</div>

前 $8$ 步使用系数 $(a,b,c)=(3.4445,-4.7750,2.0315)$，快速拉近奇异值到 $1$。最后 $2$ 步使用 $(a,b,c)=(2,-1.5,0.5)$，精准稳定奇异值。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 这系数是怎么得来的？</p>

Muon 作者 Keller Jordan 提到这组数是通过一种偏经验的梯度搜索方式调出来的；前八轮激进，后两轮稳定。

</div>

#### 3.2.7.2 算法流程

每一步训练步骤 $t$ 对每个独立权重矩阵 $W$：

1.  <strong class="list-label">计算梯度</strong>：

    $$
    G_t=\nabla_W L_t(W_{t-1})
    $$

2.  <strong class="list-label">累积动量</strong>：

    $$
    M_t=\mu M_{t-1}+G_t
    $$

    类似 Nesterov 动量，保留历史梯度的趋势。

3.  <strong class="list-label">Hybrid Newton-Schulz + Nesterov 更新</strong>：

    $$
    O_t'=\mathrm{HybridNewtonSchulz}(\mu M_t+G_t)
    $$

4.  <strong class="list-label">RMS 重缩放</strong>：

    $$
    O_t=O_t'\cdot \max(n,m)\cdot \gamma
    $$

    其中 $n,m$ 表示行列数。

5.  <strong class="list-label">权重更新</strong>：

    $$
    W_t=W_{t-1}\cdot(1-\eta\lambda)-\eta O_t
    $$

    同时完成<strong class="key-term">权重衰减</strong>和<strong class="key-term">梯度更新</strong>。

## 3.3 自适应学习率调整(学习率调度器)

<span id="sec-3-4"></span> 学习率是调整梯度下降步长的超参数,学习率过大过小都会导致模型无法收敛。在实际训练中,我们常常会根据训练的轮数来调整学习率,这就是学习率调整。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · <strong class="critical-term">为什么需要学习率调度器？</strong></p>

以一个“一个盲人下山”为例子。

- <strong class="list-label">山:</strong> 代表整个复杂的损失函数(Loss Landscape)。山的海拔高度就是损失函数的值(Loss)。

- <strong class="list-label">盲人:</strong> 代表我们的模型。

- <strong class="list-label">盲人的位置:</strong> 代表模型当前的一组参数(权重和偏置)。

- <strong class="critical-term">目标:</strong> 走到山谷的最低点,也就是让损失函数的值最小。

- <strong class="list-label">拐杖:</strong> 用来感知脚下地面的坡度,这个“坡度”就是梯度(Gradient)。梯度会指向当前位置最陡峭的上升方向,那么它的反方向就是下降最快的方向。

- <strong class="list-label">如何走路(策略):</strong> 这就是Optimizer(优化器)。它决定了盲人听从拐杖的建议后,下一步该往哪里走,走多大一步。

- <strong class="list-label">步子的大小:</strong> 这就是学习率(Learning Rate)。步子太大可能会直接跨过最低点(“扯着蛋”),步子太小则下山速度太慢。

如果整个下山过程都用固定的步长(固定的学习率),会遇到两个问题:

- <strong class="list-label">步子太大:</strong> 在山坡上(训练初期)走得很快,但快到山谷(最优点)附近时,因为步子太大,很容易一步跨过最低点,然后在谷底来回<strong class="key-term">“震荡”</strong>,始终无法精确到达谷底中心。

- <strong class="list-label">步子太小:</strong> 从一开始就走得很慢,虽然最终能精确到达谷底,但整个下山过程会非常非常耗时(训练时间过长)。

一个聪明的盲人会怎么做？他会先迈大步,快速接近谷底的大致区域；感觉脚下的路越来越平坦时,就放慢脚步,小心翼翼地小步探索,找到最低的那个点。 这就是学习率调度器的核心思想。

学习率调度器就是用来调整学习率的,它通常是一个函数,输入是当前的迭代次数,输出是当前的学习率。

</div>

常见的学习率调度器包括:<strong class="key-term">StepLR、CosineAnnealingLR、ReduceLROnPlateau等</strong>。

### 3.3.1 StepLR (阶梯式下降)

<span id="sec-3-4-1"></span> StepLR是一种简单的学习率调度器,它根据训练的轮数来调整学习率,通常是<strong class="key-term">阶梯式下降</strong>。

- <strong class="list-label">公式:</strong>

  $\alpha_t = \alpha_0 \cdot \gamma^{t}$ 其中$\alpha_0$是初始学习率,$\gamma$是衰减因子,$t$是当前的迭代次数。

- <strong class="list-label">特点:</strong> 学习率的变化是跳跃式的,像下楼梯一样,每隔一段时间就下降一个台阶。<strong class="key-term">简单有效,但不够平滑</strong>。

### 3.3.2 CosineAnnealingLR (余弦退火)

<span id="sec-3-4-2"></span>

余弦退火的方式就是预先<strong class="key-term">设定一个初始学习率</strong>,然后逐步衰减。然后余弦的<strong class="key-term">周期性</strong>让学习率在降到最低后再到最高值。

公式:

(Warm-up) If $t < T_w$, then $\alpha_t = \frac{t}{T_w}\alpha_{max}$.

(Cosine annealing) If $T_w \leq t \leq T_c$, then

$$
\alpha_{t}=\alpha_{\min }+\frac{1}{2}\left(1+\cos \left(\frac{t-T_{w}}{T_{c}-T_{w}} \pi\right)\right)\left(\alpha_{\max }-\alpha_{\min }\right)
$$

.

(Post-annealing) If $t > T_c$, then $\alpha_t = \alpha_{min}$.

其中$T_w$是预热轮数,$T_c$是余弦退火轮数,$\alpha_{max}$是初始学习率,$\alpha_{min}$是最小学习率。

<figure data-latex-placement="H">
<img src="/images/0c79164a2e.png" style="width:80.0%" alt="余弦退火示意图" />
<figcaption>余弦退火示意图</figcaption>
</figure>

```python
    class CosineSchedule:
    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        if it < self.warmup_iters:
            return self.max_learning_rate * it / self.warmup_iters
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))) / 2
```

### 3.3.3 ReduceLROnPlateau (遇到平台期就减速)

<span id="sec-3-4-3"></span> ReduceLROnPlateau (Reduce Learning Rate On Plateau) 是一种“自适应”学习率调度策略。它通过监控一个指定的性能指标(通常是<strong class="key-term">验证集损失val_loss</strong>),当这个指标在一定时期内(称为<strong class="key-term">“耐心值”patience</strong>)不再出现明显改善时,它就会自动将当前学习率降低一个固定的比例(factor),帮助模型“小步慢走”,从而跳出平台期,更精细地寻找最优解。

<strong class="key-term">解释:</strong>

我们可以把这个过程想象成“一位有耐心的老师在辅导学生”。

- <strong class="list-label">学生:</strong> 我们的模型。

- <strong class="list-label">考试分数:</strong> 被监控的指标(比如验证集损失 <strong class="key-term">val_loss</strong>)。

- <strong class="list-label">学习方法/强度:</strong> 学习率。

- <strong class="list-label">老师:</strong> ReduceLROnPlateau 调度器。

这位老师的辅导策略是:

- <strong class="list-label">观察(Monitor):</strong> 每次考完试(每个epoch结束后),老师都会记录下学生的分数(val_loss)。

- <strong class="list-label">保持耐心(Patience):</strong> 老师不会因为学生一次没考好就立刻改变策略。他会设定一个<strong class="key-term">耐心值</strong>,比如 patience=5,意味着他会连续观察5次考试。

- <strong class="list-label">判断平台期(Plateau):</strong> 如果在这连续的5次考试中,学生的最高分都没有被刷新(val_loss没有降到更低),老师就认为学生遇到了“学习瓶颈”或<strong class="key-term">“平台期”</strong>。

- <strong class="list-label">调整策略(Reduce LR):</strong> 一旦确认学生进入平台期,老师就会调整辅导策略,说:“看来之前的方法太激进了,我们放慢点,把知识点弄得再细一些。”—— <strong class="key-term">这就是将学习率乘以一个衰减因子(factor),比如0.1,让学习率骤减</strong>。

- <strong class="list-label">冷静期(Cooldown):</strong> 调整策略后,老师会进入一个<strong class="key-term">“冷静期”</strong>,比如 cooldown=2。在这两次考试内,即使学生分数还没起色,老师也不会再调整策略,而是给学生时间去适应新的、更慢的学习节奏。

<strong class="critical-term">工作流程示例:</strong>

假设我们设置如下:

monitor=’val_loss’, mode=’min’, patience=3’, factor=0.5, lr=0.01

<table>
<caption>ReduceLROnPlateau 工作流程示例</caption>
<thead>
<tr>
<th style="text-align: center;">

<strong>Epoch</strong>

</th>

<th style="text-align: center;">

<strong>val_loss</strong>

</th>

<th style="text-align: center;">

<strong>说明</strong>

</th>

<th style="text-align: center;">

<strong>Patience计数</strong>

</th>

<th style="text-align: center;">

<strong>学习率(LR)</strong>

</th>

<th style="text-align: center;">



</th>

</tr>
</thead>
<tbody>
<tr>
<td style="text-align: center;">

1

</td>

<td style="text-align: center;">

1.0

</td>

<td style="text-align: center;">

初始最佳值

</td>

<td style="text-align: center;">

0

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

2

</td>

<td style="text-align: center;">

0.9

</td>

<td style="text-align: center;">

改善,更新最佳值为0.9

</td>

<td style="text-align: center;">

重置为0

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

3

</td>

<td style="text-align: center;">

0.8

</td>

<td style="text-align: center;">

改善,更新最佳值为0.8

</td>

<td style="text-align: center;">

重置为0

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

4

</td>

<td style="text-align: center;">

0.82

</td>

<td style="text-align: center;">

未改善

</td>

<td style="text-align: center;">

1

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

5

</td>

<td style="text-align: center;">

0.81

</td>

<td style="text-align: center;">

未改善

</td>

<td style="text-align: center;">

2

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

6

</td>

<td style="text-align: center;">

0.83

</td>

<td style="text-align: center;">

未改善

</td>

<td style="text-align: center;">

3

</td>

<td style="text-align: center;">

0.01

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td colspan="5" style="text-align: left;">

<strong class="critical-term">触发条件:Patience达到3,LR需要降低</strong>

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

7

</td>

<td style="text-align: center;">

0.75

</td>

<td style="text-align: center;">

改善,更新最佳值为0.75

</td>

<td style="text-align: center;">

重置为0

</td>

<td style="text-align: center;">

0.005

</td>

<td style="text-align: center;">



</td>

</tr>
<tr>
<td style="text-align: center;">

8

</td>

<td style="text-align: center;">

0.76

</td>

<td style="text-align: center;">

未改善

</td>

<td style="text-align: center;">

1

</td>

<td style="text-align: center;">

0.005

</td>

<td style="text-align: center;">



</td>

</tr>
</tbody>
</table>

<strong class="key-term">优缺点总结:</strong>

1.  <strong class="list-label">优点:</strong>

    - <strong class="critical-term">自适应和直观:</strong>它的逻辑非常符合直觉,能够根据模型的真实表现来调整学习率,而不是依赖一个固定的、需要预先猜测的时间表。

    - <strong class="critical-term">对于精调非常有效:</strong>在训练后期,当模型性能提升变得困难时,这种“自动减速”机制对于找到更深、更优的最小值点非常有帮助。

2.  <strong class="list-label">缺点:</strong>

    - <strong class="list-label">依赖高质量的验证集:</strong>如果验证集的表现有很大的随机性或噪声,可能会导致调度器过早或错误地降低学习率。

    - <strong class="list-label">可能反应较慢:</strong>如果patience设置得过高,模型可能会在平台期浪费很多个epoch,拖慢整体训练进程。

### 3.3.4 Warmup (预热)

<span id="sec-3-4-4"></span> Warmup (预热) 是一种在深度学习模型训练初期，将学习率从一个<strong class="key-term">非常小的值(甚至为0)</strong>逐步、平滑地增加到预设的初始学习率的策略。其核心目的是为了在模型训练最开始的、最不稳定的阶段，给予模型一个“热身”或“适应”的过程，防止因初始学习率过大而导致的训练不稳定甚至崩溃。它通常作为其他学习率调度器(如余弦退火)的前奏部分。

#### 3.3.4.1 <strong>为什么初期需要预热:</strong>

在训练刚开始时，由于

1.  <strong class="list-label">参数是完全随机的:</strong>模型对输入数据一无所知，其内部的权重和偏置都是随机初始化的“胡乱猜测”。

2.  <strong class="list-label">初始损失和梯度巨大:</strong>因为参数是随机的，模型第一次进行前向传播时，其预测结果会与真实标签相差十万八千里。这会导致计算出的损失函数的值（Loss）通常非常大。

3.  <strong class="list-label">梯度方向极其不确定:</strong>巨大的损失会反向传播，计算出的梯度（Gradient）也会异常巨大且方向不稳定。这个梯度仅仅反映了从一个完全随机的点该如何“逃离”，这个方向对于全局最优解来说，参考价值很低。

而在真正进入到第一轮训练的时候，每个数据点对模型来说都是新的，模型会很快地进行数据分布修正，如果这时候学习率就很大，极有可能导致开始的时候就对该数据<strong class="key-term">“过拟合”</strong>，后面要通过多轮训练才能拉回来，浪费时间。当训练了一段时间(比如两轮、三轮)后，模型已经对每个数据点看过几遍了，或者说对当前的batch而言有了一些正确的先验，较大的学习率就不那么容易会使模型学偏，所以可以适当调大学习率。这个过程就可以看做是warmup。

那么为什么之后还要decay呢？当模型训到一定阶段后(比如十个epoch)，模型的分布就已经比较固定了，或者说能学到的新东西就比较少了。如果还沿用较大的学习率，就会破坏这种稳定性，用我们通常的话说，就是已经接近loss的local optimal了，为了靠近这个point，我们就要慢慢来。

#### 3.3.4.2 <strong>Warmup 与其他调度器的关系:</strong>

非常重要的一点是：Warmup几乎总是与我们上面提到的其他学习率调度器(如CosineAnnealingLR, StepLR)配合使用。

完整的训练过程的学习率策略被分为两个阶段:

- <strong class="list-label">第一阶段(预热期):</strong>学习率从0线性增长到 initial_lr。

- <strong class="list-label">第二阶段(正常调度期):</strong>学习率从 initial_lr 开始,按照你选择的主调度器(如余弦退火)的规则进行衰减。

<figure data-latex-placement="H">
<img src="/images/b218784267.png" style="width:80.0%" alt="Warmup 与其他调度器配合使用示意图" />
<figcaption>Warmup 与其他调度器配合使用示意图</figcaption>
</figure>

## 3.4 梯度裁剪

<span id="sec-3-5"></span> 梯度裁剪是防止梯度爆炸的一种方法,通过将梯度限制在一个范围内,从而防止梯度爆炸。

公式:

$g_t$ = min($g_t$, clip_value)

其中$clip\_value$是裁剪阈值。

```python
    class GradientClip:
    def __init__(self, parameters,max_l2_norm,epslion=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epslion = epslion

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None] #我们求l2范数是对所有元素求的,所以要先把所有元素给flatten
        all_grads = torch.cat([grad.flatten() for grad in grads])
        grad_l2 = torch.norm(all_grads,2)
        if grad_l2 > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epslion)
            for grad in grads:
                grad.mul_(clip_coeff)
```

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 3.1 · 交叉熵(Cross-Entropy)损失与KL散度

<span id="read-3-1"></span>

交叉熵损失函数:<https://blog.csdn.net/SongGu1996/article/details/99056721>

建议先熟悉信息论中的基本概念,如熵、相对熵、交叉熵等。

### 3.2 · Optimizer: 优化器

<span id="read-3-2"></span>

通俗易懂理解(梯度下降)优化算法:<https://blog.csdn.net/Invokar/article/details/86768571>

优化算法 《动手学深度学习》:<http://zh.gluon.ai/chapter_optimization/index.html>

神经网络中 warmup 策略为什么有效:<https://www.zhihu.com/question/338066667>
