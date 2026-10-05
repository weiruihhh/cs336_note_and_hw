---
outline: [2, 3]
---

# 基础知识速查

## 数学与符号速查

<span id="app-math"></span> 本附录供阅读模型、损失和强化学习公式时查阅。符号可能随章节变化，出现歧义时以局部定义为准；熵及相关概念见[信息论前置知识](/appendix#app-information)，交叉熵与 KL 散度见[语言模型的训练](/part-1/chapter-3#guide-ch-3)。

### 先看对象与维度

- <strong class="list-label">标量：</strong>单个数，例如学习率 $\eta$、损失 $L$。

- <strong class="list-label">向量：</strong>一组有序数，例如 $\boldsymbol{x}\in\mathbb{R}^{d}$。

- <strong class="list-label">矩阵：</strong>二维数组，例如 $W\in\mathbb{R}^{m\times n}$；更高维数组在实现中统称张量。

- <strong class="list-label">转置与乘法：</strong>$A^\top$ 交换矩阵的行列；若 $A\in\mathbb{R}^{m\times k}$、$B\in\mathbb{R}^{k\times n}$，则 $AB\in\mathbb{R}^{m\times n}$。

- <strong class="list-label">逐元素运算：</strong>$\odot$ 表示对应元素相乘，区别于矩阵乘法；实现时还应检查广播规则。

<strong class="note-label">例：</strong>若 $X\in\mathbb{R}^{B\times T\times d}$、$W\in\mathbb{R}^{d\times h}$，则 $XW\in\mathbb{R}^{B\times T\times h}$。这里 $B$ 是批量大小，$T$ 是序列长度，线性层作用在最后一维。

### 概率、对数与平均

- <strong class="list-label">条件概率：</strong>$p(y\mid x)$ 表示给定 $x$ 后 $y$ 的概率；它与联合概率 $p(x,y)$ 不是同一对象。

- <strong class="list-label">期望：</strong>离散情形下 $\mathbb{E}_{x\sim p}[f(x)]=\sum_x p(x)f(x)$。样本均值是期望的估计，记录估计时的采样分布。

- <strong class="list-label">对数：</strong>乘积变成对数之和。自然对数以 $e$ 为底，信息量也可能采用以 2 为底的对数；比较数值前先统一底数。

- <strong class="list-label">归约：</strong>求和与求平均会改变梯度尺度。序列长度不一致时，“每条序列平均”和“所有有效 token 平均”通常不同。

<div class="key-formula">

$$
\log p(x_{1:T})=\sum_{t=1}^{T}\log p(x_t\mid x_{<t}).
$$

</div>

<strong class="note-label">阅读方法：</strong>先找条件、采样分布和求和范围，再判断每个量是标量、向量还是序列；不要只看公式外形。

### 梯度与更新方向

对标量损失 $L(\theta)$，梯度 $\nabla_\theta L$ 与参数 $\theta$ 具有对应形状。链式法则连接复合函数中各层的导数；反向传播据此计算参数梯度。

<div class="key-formula">

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t).
$$

</div>

上式是最小化损失的梯度下降；最大化奖励时方向相反，也可以把负奖励写成待最小化的损失。阅读 PPO、GRPO、DPO 时，先确认正文写的是目标还是 loss。相关推导见对齐与推理强化学习（后续篇章，本次试读未收录）。

### 参考文献与延伸阅读

<strong class="note-label">正文索引：</strong>[模型中的线性代数](/part-1/chapter-2#guide-ch-2)、[损失函数与优化器](/part-1/chapter-3#guide-ch-3)、策略梯度与优势估计（后续篇章，本次试读未收录）。

## PyTorch 张量与自动求导

<span id="app-tensors"></span> 本附录只列跨章常用规则。模型结构和注意力计算见[Transformer 架构](/part-1/chapter-2#guide-ch-2)，性能影响见FlashAttention 与 Triton（后续篇章，本次试读未收录）。

### 形状、索引与广播

<strong class="list-label">读形状：</strong>在每个操作旁写出维度含义。例如 token ID 通常是 $(B,T)$，嵌入后为 $(B,T,d)$，预测词表上的 logits 为 $(B,T,V)$；实际布局以实现为准。

<strong class="list-label">广播：</strong>从末尾维度向前比较，每一对维度需要相等，或其中一个为 1；缺失的前导维度按 1 理解。例如 $(B,T,d)+(d)$ 可逐位置加同一偏置，而 $(B,T,d)+(B)$ 通常不能表达“每个样本一个偏置”。后一需求应先把偏置整理成 $(B,1,1)$。

<strong class="list-label">归约轴：</strong>`sum` 或 `mean` 会沿指定维度聚合；`keepdim=True` 保留长度为 1 的轴，便于后续广播。写 Softmax 时先确认归一化轴确实是候选类别或注意力的 key 位置轴。

<strong class="note-label">查阅：</strong>[PyTorch 广播规则](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)。

### 形状变换与存储布局

- <strong class="list-label">转轴：</strong>`transpose`/`permute` 改变维度顺序，可能得到非连续布局；它们不是对元素值做数学转化。

- <strong class="list-label">重塑：</strong>`reshape` 改变形状且保持元素数量，可能返回视图，也可能复制数据，不应假设它总是零拷贝。

- <strong class="list-label">视图：</strong>`view` 对尺寸和步幅有要求。遇到不兼容布局时，可根据目的使用 `reshape`，或先取得连续布局再 `view`。

- <strong class="list-label">设备与精度：</strong>除形状外，还要同时检查 `device` 和 `dtype`；不要把类型转换和数据搬运的成本混入不相关的算子比较。

<strong class="note-label">查阅：</strong>[PyTorch reshape 文档](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)。

### 反向传播与训练模式

<strong class="list-label">自动求导：</strong>参与求导的运算构成计算图；反向传播把梯度累加到相应参数的 `.grad`。普通训练需要在合适位置清零梯度，梯度累积则有意延后清零和更新。

<strong class="list-label">三个不同开关：</strong>`model.eval()` 切换某些模块的训练/评估行为，并不关闭自动求导；`no_grad` 用于不记录相关计算的梯度；参数的 `requires_grad` 决定是否需要对其求导。冻结参考模型与关闭训练模型的梯度不是一回事。

<strong class="note-label">使用前检查：</strong>输出形状是否正确、loss 是否有限、需要训练的参数是否有梯度，再开始长时间训练。

### 参考文献与延伸阅读

[Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)； [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)； [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。使用时对照实际安装版本。

## 计算量、显存与单位

<span id="app-units"></span> 配合性能分析（后续篇章，本次试读未收录）、GPU 存储（后续篇章，本次试读未收录）与Scaling Law（后续篇章，本次试读未收录）查阅。

### 容量、速率与计算量

- <strong class="list-label">容量：</strong>1 byte = 8 bit；1 GB = $10^9$ byte，1 GiB = $2^{30}$ byte。

- <strong class="list-label">计算量：</strong>FLOPs 表示浮点运算次数；FLOP/s 表示每秒运算次数。缩写有歧义时结合上下文确认。

- <strong class="list-label">带宽与吞吐量：</strong>带宽常用 byte/s；吞吐量常用 token/s 或 sample/s。注明计时是否包含读取、同步和优化器更新。

<div class="key-formula">

$$
\text{吞吐量}=\frac{\text{实际处理的 token 或样本数量}}{\text{对应范围内的时间}}.
$$

</div>

### 显存估算从哪些项开始

$P$ 个元素、每个占 $s$ byte，数据本体约占 $Ps$ byte。十亿个 FP16/BF16 元素约占 2 GB，即 1.86 GiB。

<strong class="note-label">训练显存：</strong>还包括梯度、优化器状态、激活、临时工作区、通信缓冲区及分配开销。各项精度随实现而变；参数字节数不能代表训练峰值。

### 性能比较与规模估算

<strong class="list-label">性能比较：</strong>固定输入、精度、设备和正确性要求，再比较耗时与显存；分别记录理论峰值、实测吞吐量和利用率，区分硬件与算法影响。

<strong class="list-label">规模估算：</strong>稠密语言模型训练常粗估为 $C\approx6ND$：$N$ 为参数量，$D$ 为训练 token 数，$C$ 为计算量。使用前核对结构与计数假设。

### 参考文献与延伸阅读

<strong class="note-label">正文索引：</strong>测量与显存快照（后续篇章，本次试读未收录）、GPU 存储与注意力优化（后续篇章，本次试读未收录）、Scaling Law 变量与预算（后续篇章，本次试读未收录）。

## 实验复现与结果记录

<span id="app-repro"></span> 这份速查表用于第一篇训练、第二篇性能比较以及第五、六篇后训练实验。建议每次运行保存一份配置和一张结果表，避免只留下截图。

### 一次运行应记录什么

代码与环境  
仓库版本、依赖版本、设备型号与数量、驱动和运行精度。

模型与数据  
模型和 tokenizer 版本，数据来源、配置、划分、样本数、清洗或转换脚本；验证集与训练集分开管理。

训练配置  
种子、序列长度、批量口径、优化器、学习率计划、训练步数、有效 token 数和 checkpoint 路径。

评估配置  
提示模板、采样设置、最大生成长度、停止条件、解析逻辑、评分方式和评估版本。

输出与异常  
训练/验证曲线、主指标、解析失败、非有限值、耗时和显存；需要诊断时保留有代表性的输入输出。

### 批量与梯度累积的口径

在每个数据并行进程使用相同 microbatch、连续累积 $K$ 次后更新且没有额外变化的情况下：

<div class="key-formula">

$$
B_{\mathrm{global}}=B_{\mathrm{micro}}\times K\times W_{\mathrm{DP}}.
$$

</div>

这里 $W_{\mathrm{DP}}$ 是数据并行进程数；张量并行或流水线并行的卡数不能直接代入此处。变长序列实验还应记录有效 token 数。

<strong class="note-label">损失缩放：</strong>各 microbatch 大小相同、且 loss 使用相同均值口径时，可按累积次数缩放。若有效 token 数不同，应按实际计数正确加权，不能机械地平均各批的均值。具体实现见梯度累积（后续篇章，本次试读未收录）。

### 公平比较与恢复训练

<strong class="list-label">公平比较：</strong>预先固定基线、评价集合与指标；消融尽量只改变目标因素。区分模型效果、数据处理差异和硬件带来的速度差异。

<strong class="list-label">恢复训练：</strong>除模型权重外，按训练流程保存优化器、调度器、训练步和所需随机状态；数据采样位置等状态也会影响续训轨迹。恢复能力应先用短运行检查。

<strong class="list-label">随机性：</strong>设置种子有助于控制实验，但不能保证跨设备、跨版本或所有算子逐位一致。需要严格比较时，记录确定性设置并观察多次运行的波动。

### 参考文献与延伸阅读

[PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)； [脚本化训练与 checkpoint](/part-1/chapter-4#guide-ch-4)；计时与性能分析（后续篇章，本次试读未收录）；训练后评估对照（后续篇章，本次试读未收录）。

## 正则表达式与文本匹配

<span id="app-regex"></span> 本附录供阅读[分词器与 BPE](/part-1/chapter-1#guide-ch-1)中的预分词部分时查阅。

### 概念与常用符号

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 正则表达式</p>

正则表达式（Regular Expression,常缩写为regex或regexp）是一个强大的文本模式匹配工具。它本质上是一种用特殊符号编写的“规则字符串”,可以用来<strong class="key-term">查找、替换、分割或验证任何符合该规则的文本</strong>。可以理解为ctrl+f的超级加强版。

</div>

<table>
<caption>正则表达式符号表<span id="tab-regex_symbols"></span></caption>
<thead>
<tr>
<th style="text-align: left;"><strong>类型</strong></th>
<th style="text-align: left;"><strong>符号/语法</strong></th>
<th style="text-align: left;"><strong>解释说明</strong></th>
<th style="text-align: left;"><strong>示例</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">普通字符</td>
<td style="text-align: left;"><code>a</code>, <code>b</code>, <code>1</code>, <code>2</code></td>
<td style="text-align: left;">匹配它们自身。</td>
<td style="text-align: left;"><code>cat</code> 会精确匹配字符串 "cat"。</td>
</tr>
<tr>
<td style="text-align: left;">元字符（任意字符）</td>
<td style="text-align: left;"><code>.</code></td>
<td style="text-align: left;">匹配<strong>除了换行符以外</strong>的任意单个字符。</td>
<td style="text-align: left;"><code>c.t</code> 会匹配 "cat", "cot", "c_t" 等。</td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">元字符（重复次数）</td>
<td style="text-align: left;"><code>*</code></td>
<td style="text-align: left;">匹配前面的元素 <strong>0次或多次</strong>。</td>
<td style="text-align: left;"><code>ca*t</code> 会匹配 "ct", "cat", "caaat"。</td>
</tr>
<tr>
<td style="text-align: left;"><code>+</code></td>
<td style="text-align: left;">匹配前面的元素 <strong>1次或多次</strong>。</td>
<td style="text-align: left;"><code>ca+t</code> 会匹配 "cat", "caaat",但<strong>不匹配</strong> "ct"。</td>
</tr>
<tr>
<td style="text-align: left;"><code>?</code></td>
<td style="text-align: left;">匹配前面的元素 <strong>0次或1次</strong>。</td>
<td style="text-align: left;"><code>colou?r</code> 会匹配 "color" 和 "colour"。</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">字符集</td>
<td style="text-align: left;"><code>[...]</code></td>
<td style="text-align: left;">匹配方括号内的<strong>任意一个</strong>字符。</td>
<td style="text-align: left;"><code>c[ao]t</code> 只会匹配 "cat" 和 "cot"。</td>
</tr>
<tr>
<td style="text-align: left;"><code>[^...]</code></td>
<td style="text-align: left;">匹配<strong>不在</strong>方括号内的任意一个字符。</td>
<td style="text-align: left;"><code>[^0-9]</code> 会匹配任何非数字字符。</td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">分组与或</td>
<td style="text-align: left;"><code>(...)</code></td>
<td style="text-align: left;">将括号内的内容视为一个整体,可以对整体做重复。</td>
<td style="text-align: left;"><code>(ab)+</code> 会匹配 "ab", "abab", "ababab"。</td>
</tr>
<tr>
<td style="text-align: left;"><code>|</code></td>
<td style="text-align: left;">表示"或"（OR）逻辑。</td>
<td style="text-align: left;"><code>cat</code>dog| 会匹配 "cat" 或者 "dog"。</td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">预定义字符类</td>
<td style="text-align: left;"><code>\d</code></td>
<td style="text-align: left;">匹配任意一个<strong>数字</strong> (Digit),等同于 <code>[0-9]</code>。</td>
<td style="text-align: left;"><code>\d\d\d</code> 会匹配 "123", "987"。</td>
</tr>
<tr>
<td style="text-align: left;"><code>\w</code></td>
<td style="text-align: left;">匹配任意一个<strong>单词字符</strong>,包括字母、数字、下划线。</td>
<td style="text-align: left;"><code>\w+</code> 会匹配一个完整的单词或数字。</td>
</tr>
<tr>
<td style="text-align: left;"><code>\s</code></td>
<td style="text-align: left;">匹配任意一个<strong>空白字符</strong>,包括空格、制表符、换行符。</td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>

### Python 使用示例

<div class="custom-block tip">

<p class="custom-block-title">例子 · 在python代码中使用正则表达式</p>

re.findall和re.finditer的区别:

re.findall返回所有匹配的子字符串,返回一个列表。

re.finditer则是<strong class="key-term">惰性的</strong>,返回一个迭代器,每次只返回一个匹配的子字符串,需要手动调用next()方法来获取下一个匹配的子字符串。正因如此,无论文本有多大,匹配项有多少,<strong class="key-term">内存占用都极低</strong>,因为它一次只处理一个匹配项。这是处理大文件的唯一可行方法。讲义中也推荐使用re.finditer。

</div>

```python
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    re.findall(PAT, "some text that i'll pre-tokenize")
    >>> ['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenize']
```

### 参考文献与延伸阅读

正则表达式相关知识: <https://www.runoob.com/regexp/regexp-intro.html>

## 信息论前置知识(简易版)

<span id="app-information"></span><span id="sec-3-1"></span>

### 信息 (Information)

<span id="app-information-1"></span>

它用来<strong class="key-term">度量事件发生所消除的不确定性(Uncertainty)的大小</strong>。

1.  直观理解

    - “明天太阳会照常升起”:这句话包含的信息量很小。因为它几乎是必然发生的,没有消除我们什么不确定性。

    - “明天北京会下雪(假设在夏天)”:这句话包含的信息量极大。因为它是一个极小概率事件,一旦发生,就极大地消除了我们的不确定性。

2.  <strong class="list-label">量化定义:</strong>自信息 (Self-Information) 信息论使用“自信息”来量化单个事件发生时所提供的信息量。一个事件 x 的自信息量 I(x) 定义为:

    $I(x) = -\log_b(p(x))$

    - <strong class="list-label">p(x):</strong>事件 x 发生的概率。

    - $\log_b$:对数函数。底数 b 的选择决定了信息量的单位。

      - b=2:单位是 <strong class="key-term">比特 (bit)</strong>,这是最常用的单位,对应于二进制世界。

      - b=e:单位是 <strong class="key-term">奈特 (nat)</strong>。

      - b=10:单位是 <strong class="key-term">哈特利 (hartley)</strong>。

    - <strong class="list-label">负号:</strong>因为概率 p(x) 在 \[0, 1\] 之间,它的对数是小于等于0的。加上负号可以确保信息量是一个非负数,这符合我们的直觉。

3.  示例

    - <strong class="list-label">假设我们抛一枚均匀的硬币:</strong>

    - “正面朝上”的概率 p(正面) = 0.5。

      它提供的信息量 $I(正面) = -\log_2(0.5) = -\log_2(2^{-1}) = -(-1) = 1 bit$。

    - 同样,“反面朝上”提供的信息量也是 1 bit。

      这告诉我们,要确定一个硬币的正反面,我们需要 1 bit 的信息。

### 熵 (Entropy)

<span id="app-information-2"></span>

如果我们想衡量的是一个<strong class="key-term">系统(随机变量)的整体不确定性</strong>,而不是单个事件,那就要用到“熵”的概念。

1.  直观理解

    - 熵是<strong class="key-term">信息量的期望值(数学期望)</strong>。它衡量了一个随机变量所有可能结果的平均不确定性。

    - <strong class="key-term">一个系统越混乱、越不可预测,它的熵就越高</strong>。

    - <strong class="key-term">一个系统越稳定、越可预测,它的熵就越低</strong>。

2.  量化定义

    - 对于一个离散随机变量 X,它有多种可能的取值 $\{x_1, x_2, ..., x_n\}$,对应的概率为 $\{p(x_1), p(x_2), ..., p(x_n)\}$。那么,这个随机变量 X 的熵 H(X) 定义为:

    - $H(X) = E[I(X)] = \sum_{i=1}^{n} p(x_i) I(x_i) = -\sum_{i=1}^{n} p(x_i) \log_b(p(x_i))$

3.  示例

    - <strong class="list-label">比较两枚硬币的熵:</strong>

      - <strong class="list-label">均匀硬币 (Fair Coin):</strong>p(正面)=0.5, p(反面)=0.5

      - 不均匀硬币 (Biased Coin):p(正面)=0.9, p(反面)=0.1

      - 两面都是正面的硬币 (Two-headed Coin):p(正面)=1, p(反面)=0

    - <strong class="list-label">均匀硬币 (Fair Coin):</strong>p(正面)=0.5, p(反面)=0.5

      $H(X) = -[0.5 \times \log_2(0.5) + 0.5 \times \log_2(0.5)] = 1 bit$

    - 不均匀硬币 (Biased Coin):p(正面)=0.9, p(反面)=0.1

      $H(X) = -[0.9 \times \log_2(0.9) + 0.1 \times \log_2(0.1)] \approx 0.469 bit$

    - 两面都是正面的硬币 (Two-headed Coin):p(正面)=1, p(反面)=0

      $H(X) = -[1 \times \log_2(1) + 0 \times \log_2(0)] = 0 (约定 0 \log 0 = 0)$

### 熵的相关扩展概念

<span id="app-information-3"></span>

理解了熵之后,其他几个重要概念就很容易理解了,它们描述了多个随机变量之间的关系。

1.  <strong class="list-label">联合熵 (Joint Entropy)</strong>

    - 衡量<strong class="key-term">两个或多个随机变量共同</strong>的不确定性。对于两个变量 X 和 Y,其联合熵 H(X, Y) 为:

    - $H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2(p(x, y))$

    - 其中 p(x, y) 是 X=x 和 Y=y 同时发生的联合概率。

2.  <strong class="list-label">条件熵 (Conditional Entropy)</strong>

    - 在<strong class="key-term">已知一个随机变量 X 的情况下,另一个随机变量 Y 剩下的不确定性</strong>。记为 H(Y\|X)。

    - $H(Y|X) = \sum_{x \in X} p(x) H(Y|X=x)$

    - 它表示,知道了 X 之后,对 Y 的不确定性还剩下多少。

3.  <strong class="list-label">互信息 (Mutual Information)</strong>

    - <strong class="key-term">一个随机变量 X 的信息中,有多少是与另一个随机变量 Y 共享的</strong>。它衡量了两个变量之间的相关性。记为 I(X; Y)。

    - <strong class="note-label">直观理解</strong>:知道了 X 之后,Y 的不确定性减少了多少。

    - <strong class="list-label">计算公式</strong>:

      - $I(X; Y) = H(Y) - H(Y|X) (\KeyTerm{Y的总不确定性 - 知道X后Y剩下的不确定性})$

      - $I(X; Y) = H(X) - H(X|Y) (\KeyTerm{对称的})$

      - $I(X; Y) = H(X) + H(Y) - H(X, Y)$

Venn图关系 你可以把熵想象成一个集合,那么这些概念的关系就非常清晰了:

<figure data-latex-placement="H">
<img src="/images/77d0408750.png" style="width:80.0%" alt="信息论概念的Venn图关系" />
<figcaption>信息论概念的Venn图关系</figcaption>
</figure>

 

- 左边的圆圈是 H(X)

- 右边的圆圈是 H(Y)

- 两个圆圈的重叠部分是<strong class="key-term">互信息 I(X; Y)</strong>

- H(X) 中不重叠的部分是<strong class="key-term">条件熵 H(X\|Y)</strong>

- H(Y) 中不重叠的部分是<strong class="key-term">条件熵 H(Y\|X)</strong>

- 整个两个圆圈覆盖的区域是<strong class="key-term">联合熵 H(X, Y)</strong>

### 一个生动的比喻:猜数字游戏

<span id="app-information-4"></span>

这个比喻可以把所有概念串起来。

- <strong class="list-label">游戏</strong>:我从1到8之间想一个数字,你来猜。

- <strong class="list-label">熵 H(X)</strong>:在游戏开始前,这个数字是什么？你完全不知道,有8种可能性,每种可能性概率为1/8。这个系统的总不确定性是 $H(X) = -\sum_{i=1}^{8} \frac{1}{8}\log_2(\frac{1}{8}) = \log_2(8) = 3 bits$ 。这恰好是你用“二分法”猜中这个数字所需的最少问题数(例如:“比4大吗？”“比6大吗？”“是7吗？”)。

- <strong class="list-label">信息 I(x)</strong>:我告诉你答案是“5”。这个具体事件提供的信息量是 $I("5") = -\log_2(1/8) = 3 bits$ 。一旦你知道了这个信息,所有不确定性都消除了。

- <strong class="list-label">互信息 I(X;Y)</strong>:现在,我不直接告诉你答案,而是给你一个提示 <strong class="key-term">Y</strong>:“这个数字是奇数”。

  - 这个提示 <strong class="key-term">Y</strong> 本身也有不确定性(可能是奇数或偶数),它的熵是 <strong class="key-term">H(Y)=1 bit</strong>。

  - 这个提示 <strong class="key-term">Y</strong> 给你提供了多少关于 <strong class="key-term">X</strong> 的信息？这就是互信息。知道了它是奇数,可能性从1,2,3,4,5,6,7,8缩小到了1,3,5,7。不确定性大大降低。

- <strong class="list-label">条件熵 H(X\|Y)</strong>:你知道了“数字是奇数” <strong class="key-term">(Y)</strong> 之后,对数字 <strong class="key-term">(X)</strong> 还剩下多少不确定性？

  - 现在只剩下4种可能性1,3,5,7,每种概率为1/4。

  - 剩下的不确定性(条件熵)是 $H(X|Y=\text{"奇数"}) = -\sum_{i \in \{1,3,5,7\}} \frac{1}{4}\log_2(\frac{1}{4}) = \log_2(4) = 2 bits$。

  - 这意味你还需要问2个问题才能猜到。

- <strong class="list-label">关系验证</strong>:

  - $I(X;Y) = H(X) - H(X|Y) = 3 - 2 = 1 bit。$

  - 这说明,“数字是奇数”这个提示,为你提供了 1 bit 的信息。

| <strong>中文名称</strong> | <strong>描述</strong> | <strong>关键点</strong> |
|:--:|:---|:---|
| <strong>自信息</strong> | 度量<strong>单个具体事件</strong>发生所消除的不确定性。 | 概率越小,信息量越大。 |
| <strong>熵</strong> | 度量<strong>整个系统(随机变量)</strong>的平均不确定性。 | 信息量的数学期望。系统越混乱,熵越高。 |
| <strong>联合熵</strong> | 度量<strong>两个或多个系统</strong>共同的不确定性。 | 把多个变量看成一个整体。 |
| <strong>条件熵</strong> | 在<strong>已知一个变量</strong>后,另一个变量<strong>剩下</strong>的不确定性。 | $H(Y\|X)$,知道X后Y还有多乱。 |
| <strong>互信息</strong> | 两个变量<strong>共享</strong>的信息量,即相关性程度。 | $I(X;Y)$,知道X能帮我们消除Y多少不确定性。 |
