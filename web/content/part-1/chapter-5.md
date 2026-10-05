---
outline: [2, 3]
---

# 第 5 章 · 推理与采样

<span id="guide-ch-5"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[模型前向计算](/part-1/chapter-2#guide-ch-2)和[概率与损失](/part-1/chapter-3#guide-ch-3)；能够加载[训练检查点](/part-1/chapter-4#guide-ch-4)。

<strong>准备工作：</strong>准备同一模型的 checkpoint、配套 tokenizer 和一组固定提示词；无需另行下载训练数据。

<strong>本章任务：</strong>实现自回归生成、温度缩放、Top-p 采样与停止条件；对相同提示词比较采样设置。

</div>

模型训练之后就是我们最关心也是最常用的推理部分，这一部分主要是将训练好的模型加载出来，然后根据输入的一段文本，用模型来输出一段文本。 训练部分可以理解为对输入文本的各种各样的<strong class="key-term">编码</strong>，让模型能够学习。而推理部分就是将学习后的模型后对输入的文本再进行<strong class="key-term">解码</strong>，得到最终的输出。

## 5.1 Softmax层与解码

正如我们之前所说的，模型最后有一个线性层，它将输出一个维度等于模型的词汇表大小（Vocabulary Size）的向量，这个输出的、长度为词汇表大小的向量，就是 <strong class="key-term">Logits 向量</strong>。这个 Logits 向量中的每一个元素，都对应词汇表中的一个词。例如： logits\[10\] 可能是词 “pig” 的分数,logits\[25\] 可能是词 “cat” 的分数,logits\[100\] 可能是词 “dog” 的分数。

假设我们得到的 Logits 向量中，与 “pig” 对应的分数值是 10.2，与 “cat” 对应的分数值是 5.1，而与 “dog” 对应的分数值是 -3.5。这直观地表明，模型非常有信心地认为下一个词是 “pig”，对 “cat” 有一些信心，但几乎完全排除了 “dog”。

但Logits 本身是原始分数，不直观，也不方便计算损失函数。因此，我们需要将它们转换成标准的概率分布（所有项的概率在 0 到 1 之间，且总和为 1）。 这个转换步骤通过 Softmax 函数 来完成。经过 Softmax 处理后，我们就得到了一个概率分布向量。在上面的例子中： “pig” 的概率可能变成了 0.95 (95$\%$),“cat” 的概率可能变成了 0.04 (4$\%$),“dog” 的概率可能变成了 0.01 (接近 0$\%$)。 所有词汇表的词汇的概率之和为 1。

用公式表示即：

$P(x_{t+1}=i \mid x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$

$v = \mathrm{TransformerLM}(x_{1\ldots t})_t \in \mathbb{R}^{\text{vocab\_size}}$

<figure data-latex-placement="H">
<img src="/images/8d7f8c5b90.png" style="width:80.0%" alt="Transformer架构流程图" />
<figcaption>Transformer架构流程图</figcaption>
</figure>

因此总结下来，整个解码过程就是输入一段文本(或者说提示词Prompt),模型就会根据提示词生成下一个词的概率分布，模型进行采样得到下一个词，自回归特性将重复这个过程，直到生成序列结束标记 \<endoftext\>（或达到我们指定的最大生成数）。

## 5.2 温度缩放（Temperature Scaling）

<span id="sec-5-1"></span> 我们刚才提到了Transformer 解码器在每步生成时会输出 logits（未归一化的分数），通过 Softmax 转化为概率，然后决定下一个 token。

即

$$
\operatorname{softmax}(v)_{i}=\frac{\exp \left(v_{i}\right)}{\sum_{j=1}^{vocab-size} \exp \left(v_{j}\right)} .
$$

温度缩放就是先把 logits 除以一个 ‘T‘ 值：

即

<div class="key-formula">

$$
\operatorname{softmax}(v, \tau)_{i}=\frac{\exp (\frac{v_{i}}{\tau})}{\sum_{j=1}^{{vocab-size } } \exp (\frac{v_{j}}{\tau})} .
$$

</div>

- 当 <strong class="critical-term">$T < 1$</strong> 时：会 <strong class="key-term">放大高概率 token</strong> 的优势，使分布更尖锐，对概率最高的词更偏向于生成，文本更<strong class="key-term">确定</strong>和<strong class="key-term">保守</strong>。

- 当 <strong class="critical-term">$T > 1$</strong> 时：会 <strong class="key-term">平滑 logits</strong>，使高低概率 token 差异减小，低概率 token 出现的机会增多，从而生成更具<strong class="key-term">创造性</strong>但可能更<strong class="key-term">不稳定</strong>或<strong class="key-term">混乱</strong>的内容。

当 $T$ 趋于无穷大时，输出概率分布将趋于<strong class="key-term">均匀分布</strong>，概率为 $1/K$，此时<strong class="key-term">信息熵是最大的</strong>。反过来，$T$ 趋于0时，正确类别的概率接近1，输出结果就是确定的，信息熵为 0 。 如果 $T$ 太低，生成倾向高频词，输出趋于一致；$T$ 太高，输出虽多样但可能跑题、不连贯。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · </p>

<strong class="note-label">注意：</strong>温度缩放是 <strong class="key-term">离线</strong> 操作，在生成时一次性应用，不会影响模型训练。

这里的温度 temperature 和Gemini等大模型的 temperature 设置道理一致。

</div>

## 5.3 Top-p 采样

<span id="sec-5-2"></span>

Top‑p 方法是动态选取那些累计概率高于门限 $p$ 的 token 集合，然后从中随机抽样：

1.  对所有 token 按原始概率降序排序；

2.  选择前面的 token，直到累计概率 $\geq p$；

3.  丢弃剩余 token，重新归一化所选集合的概率；

4.  从中随机选取下一个 token。

<strong class="key-term">候选集大小根据实际分布变化</strong>，$p$ 较低时生成更保守、$p$ 较高时生成更保留创造力。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)。使用前核对课程年份与仓库版本。

### 5.1 · 实现与实验依据

<span id="read-5-1"></span>

采样实现与实验范围见上述 Assignment 1 讲义；本章不额外重复数学推导，概率记号可查[数学与符号](/appendix#app-math)。
