---
outline: [2, 4]
---

# 第 9 章 · Scaling Law

<span id="guide-ch-10"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[验证损失](/part-1/chapter-3#guide-ch-3)与[计算成本](/part-2/chapter-6#guide-ch-7)；会读对数坐标，单位见[计算与存储单位](/appendix#app-units)。

<strong>准备工作：</strong>准备不同规模训练的参数量、token 数、计算预算和验证损失记录；官方作业入口见[本篇资源入口](/part-3/resources)。

<strong>本章任务：</strong>本章以理论与分析为主：理解幂律、Chinchilla 与 IsoFLOPs；建议用已有实验点练习拟合和预算分配。

</div>

[参考资料 9.1](/part-3/chapter-9#read-10-1)

## 9.1 Scaling Law

<span id="sec-11-1"></span>

Scaling Law 指出，大语言模型的性能（通常以 Test Loss 衡量）与三个主要变量之间存在着严格的<strong class="key-term">幂律（Power Law）</strong>关系。这三个变量是：

1.  <strong class="list-label">计算量 (Compute, C)</strong>：训练模型所用的 FLOPs。

2.  <strong class="list-label">数据集大小 (Dataset Size, D)</strong>：训练用的 Token 数量。

3.  <strong class="list-label">参数量 (Parameters, N)</strong>：模型神经网络的权重数量。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 幂律分布</p>

幂律意味着：规模扩大时，性能按指数规律提升，但提升幅度逐渐递减。

</div>

<figure data-latex-placement="H">
<img src="/images/53c4b72578.png" style="width:80.0%" alt="幂律分布示意图" />
<figcaption>幂律分布示意图</figcaption>
</figure>

### 9.1.1 Scaling Law 相关公式

Chinchilla 被认为是目前最通用的 Scaling Law，它修正了早期过分看重模型大小的误区，强调了<strong class="key-term">数据量</strong>的重要性。

#### 9.1.1.1 Chinchilla Loss Formula

这是目前最通用的计算公式。

<span class="key-formula">$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$</span>

参数说明：

- <strong class="list-label">L(N, D)</strong>：模型在给定参数量 N 和训练数据量 D 下的预估损失值（Loss）。损失越低，性能越好。

- <strong class="list-label">E</strong>：不可约损失（或称贝叶斯误差）。这是基于自然语言本身的熵，代表了模型性能的<strong class="key-term">理论极限</strong>。即使用了无限的参数和无限的数据，Loss 也不会低于 E。

- <strong class="list-label">N</strong>：模型参数数量。

- <strong class="list-label">D</strong>：训练数据的 Token 数量。

- <strong class="list-label">A, B, $\alpha$, $\beta$</strong>：这些是根据实验数据拟合出来的常数。

  - $\frac{A}{N^\alpha}$：表示由于<strong class="key-term">模型太小（容量不足）</strong>而导致的额外损失。

  - $\frac{B}{D^\beta}$：表示由于<strong class="key-term">数据太少（训练不足）</strong>而导致的额外损失。

<figure data-latex-placement="H">
<img src="/images/c6de525702.png" style="width:80.0%" alt="C、D、N之间的关系" />
<figcaption>C、D、N之间的关系</figcaption>
</figure>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

关键发现：Chinchilla 研究得出 $\alpha \approx 0.34$ 且 $\beta \approx 0.28$。这意味着模型参数量 N 和数据量 D 对性能的贡献是<strong class="key-term">大致相当的</strong>（指数相近），因此在扩大算力时，应该同比例扩大模型和数据（即 1:1 缩放），而不是像以前那样盲目追求巨大的模型参数。

</div>

在实际训练中，我们通常受限于计算约束。计算量 C 与 N 和 D 的关系近似为：

<span class="key-formula">$C \thickapprox 6 \times N \times D$</span>

- <strong class="list-label">C (FLOPs)</strong>：总浮点运算次数。

- <strong class="list-label">6</strong>：这是一个经验系数（在 Transformer 架构中，每个 Token 的训练大约需要 6N 次浮点运算（处理D个Token自然就6ND），其中前向传播 2N，反向传播 4N）。

有了这个公式，我们的好处就是<strong class="key-term">可以通过训练小模型，精准预测大模型的表现</strong>，从而避免为了验证想法而浪费几百万美元训练一个无用的巨型模型。

具体来说，就是已知小模型的Loss后，根据幂律曲线来估算大模型的Loss。

#### 9.1.1.2 Kaplan的单变量幂律公式

他认为模型性能主要取决于规模的指数级增长。

$L(N)\approx(\frac{N_c}{N})^{α_N}$

$L(D)\approx(\frac{D_c}{D})^{α_D}$

$L(C)\approx(\frac{C_c}{C})^{α_C}$

早期的Kaplan认为<strong class="key-term">参数量 (N)</strong> 是王道。于是拼命把模型做大，哪怕数据量不够。这就使得总体模型是欠拟合的。

#### 9.1.1.3 “IsoFLOPs 方法” (等计算量分析法)

讲义里Chinchilla关于推导Scaling Law的方法

在固定计算量 C （比如设置为10e20 FLOPs）的前提下，衡量模型参数 N 和数据量大小 D 的关系。

当他们把实验结果画成图（横轴是模型大小 $N$，纵轴是最终 Loss）时，发现曲线呈现出一个 <strong class="key-term">“U型”</strong>

1.  <strong class="key-term">左端：模型(N)太小</strong>

    - <strong class="list-label">解释</strong>：虽然你给小模型喂了海量的数据（因为 N 小，D 就可以很大），但它的“脑容量”太小了，根本记不住也学不会这么复杂的知识。导致<strong class="key-term">欠拟合</strong>。

    - <strong class="list-label">结果</strong>：Loss 很高。

2.  <strong class="key-term">右端：数据(D)太小</strong>

    - <strong class="list-label">解释</strong>：你造了一个超级巨大的模型，但因为预算 C 是固定的，你剩下的钱只够让它读几页书（D 非常小）。模型还没来得及收敛，计算预算就花光了。这叫 <strong class="key-term">训练不充分</strong>。

    - <strong class="list-label">极端的例子</strong>：原文提到，如果 N 无穷大，你的预算可能连做一次梯度下降（算一步）都不够，训练直接结束。

    - <strong class="list-label">结果</strong>：Loss 也很高。

3.  <strong class="key-term">底部：甜点区 (Sweet Spot)</strong>

    - <strong class="list-label">解释</strong>：在这里，N近似等于D， Loss 最低。

<strong class="critical-term">实操步骤</strong>:

1.  <strong class="list-label">找最低点</strong>：对于每一个预算等级（比如 $C_1, C_2, C_3$），他们都画出那个 U 型曲线，并找到曲线最低点对应的<strong class="key-term">最优模型大小</strong>，记为 $N_{opt}(C)$。

2.  <strong class="list-label">连点成线</strong>：现在有了一系列的数据点对：$\langle C_1, N_{opt1} \rangle, \langle C_2, N_{opt2} \rangle, \dots$

3.  <strong class="list-label">拟合幂律</strong>：他们用数学公式去拟合这些点，得出结论：

    $N_{opt}\varpropto C_a$

    $D_{opt}\varpropto C_b$

<strong class="critical-term">实际意义与应用</strong>

- <strong class="list-label">预测未来新模型</strong>：给定目标计算预算 C，直接用公式估算最优 N_opt 和 D_opt，避免盲目超大或超小。

- <strong class="list-label">小模型外推</strong>：通过训练一系列小模型拟合 A、B、α、β、E 等参数，即可可靠预测百倍、千倍规模模型（比如 GPT-4 的预算）的性能。

- <strong class="list-label">避免浪费</strong>：过去常花数百万美元训练一个配置不佳的巨型模型；现在可用十分之一预算的小实验验证想法。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment3-scaling)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 9.1 · Scaling Law

<span id="read-10-1"></span>

幂律分布：<https://www.zhihu.com/question/312593367>

Scaling Law: <https://zhuanlan.zhihu.com/p/671327709>
