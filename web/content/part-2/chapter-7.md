---
outline: [2, 3]
---

# 第 7 章 · FlashAttention 与 Triton 优化

<span id="guide-ch-8"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>掌握[缩放点积注意力](/part-1/chapter-2#guide-ch-2)与[性能测量](/part-2/chapter-6#guide-ch-7)；复习[显存与带宽](/appendix#app-units)及[GPU 内存架构](/appendix#app-gpu-memory)。

<strong>准备工作：</strong>准备支持所用 Triton 环境的 GPU；使用可复现的随机 Q、K、V 作为正确性和性能测试输入。

<strong>本章任务：</strong>理解分块、在线 Softmax 与重计算；实现并比较 PyTorch 基线和 Triton 注意力实现。

</div>

在上一节利用Nsight Systems等工具对模型进行计算分析,可以看到限制模型内存和计算资源的主要瓶颈主要还是在Attention注意力计算上。因此，我们优化的目标也会集中在此。仔细分析过后可以发现，Attention计算的瓶颈主要在于IO感知上，FlashAttention的优化方法应运而生。

GPU 的片上与片下内存、逻辑内存层次见[GPU 内存架构](/appendix#app-gpu-memory)。

## 7.1 FlashAttention

<span id="sec-8-2"></span>

[参考资料 7.1](/part-2/chapter-7#read-8-2)

开宗明义：FlashAttention 的背景是在当前原始的Attention在GPU中，存写速度是逊色于计算速度的(memory-bound)，也就是说存写相比于计算时拖了后腿的。为了达到最佳效率，我们需要减少多次存写，哪怕以稍微降低计算速度的代价。

### 7.1.1 GPU 的 FLOPs 计算能力与内存吞吐能力

<span id="sec-8-2-1"></span>

多年来，GPU的计算能力（FLOPS）的增长速度比增加内存吞吐量（TB/s）更快。

<figure data-latex-placement="H">
<img src="/images/59485bf200.png" style="width:80.0%" alt="GPU的FLOPs计算能力与内存吞吐能力" />
<figcaption>GPU的FLOPs计算能力与内存吞吐能力</figcaption>
</figure>

如果没有数据需要处理，那么额外的 FLOPS 的计算能力是没有意义的。总而言之，只有<strong class="key-term">存算</strong>合理配合，才能达到最佳效率。

### 7.1.2 I/O 感知

<span id="sec-8-2-2"></span> IO感知（IO-awareness） 是一种在设计算法时充分考虑<strong class="key-term">硬件输入/输出（I/O）开销</strong>的优化思想，特别是在 GPU 这类<strong class="key-term">计算速度远超内存访问速度</strong>的设备上尤为重要。

在现代GPU中，性能瓶颈往往不是浮点运算（FLOPs）的速度，而是从<strong class="key-term">显存（HBM）读写数据的速度</strong> 。因此，IO感知的算法旨在通过优化内存访问模式，特别是<strong class="key-term">减少对速度较慢、但容量较大的高带宽内存（HBM）的读写次数</strong>，来减少整体的运行时钟时间（wall-clock time）。

简单来说，IO感知的核心就是：<strong class="key-term">尽可能减少与慢速内存的数据交换，让计算尽可能在快速内存中完成。</strong>

### 7.1.3 FlashAttention如何充分考虑GPU的内存层级结构？

<span id="sec-8-2-3"></span>

#### 7.1.3.1 标准注意力算法的缺点

<span id="sec-8-2-3-1"></span> 标准注意力算法没有感知到IO的成本，它会频繁地读写HBM 。例如，它需要计算并存储一个巨大的$N\times N$的中间注意力矩阵（S和P）到HBM中，然后再从HBM中读出进行下一步计算 。这些冗余的HBM读写操作（IO）占据了大量的计算时间，成为了性能瓶颈 。

Transformer的计算瓶颈不在运算能力，而在读写速度上。

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

- <strong class="list-label">FLOPS</strong>：等同于FLOP/s，表示<strong class="key-term">Floating Point Operations Per Second</strong>，即每秒执行的浮点数操作次数，用于衡量硬件计算性能。

- <strong class="list-label">FLOPs</strong>：表示<strong class="key-term">Floating Point Operations</strong>，表示某个算法的总计算量（即总浮点运算次数），用于衡量一个算法的复杂度。

</div>

#### 7.1.3.2 FlashAttention的性能提升方式

<span id="sec-8-2-3-2"></span>

1.  <strong class="list-label">利用SRAM进行分块计算（Tiling）</strong>：为了避免对HBM的反复读写，FlashAttention的核心思路是将多个操作融合成一个单一的GPU核（Kernel），并利用速度快10倍左右的SRAM进行计算 。

    <figure data-latex-placement="H">
    <img src="/images/4a125b5d94.png" style="width:80.0%" alt="GPU体系架构" />
    <figcaption>GPU体系架构</figcaption>
    </figure>

    - 它将输入的查询（Q）、键（K）、值（V）矩阵从HBM中切分成块（Blocks/Tiles）。

    - 然后，它逐块将这些数据加载到高速的SRAM中 。所有的中间计算步骤，包括矩阵乘法和Softmax，都在SRAM内部完成，完全不产生将巨大的‘N×N‘中间矩阵写回HBM的操作 。

    - 最后，只将计算完成的最终输出结果从SRAM一次性写回HBM 。

    <figure data-latex-placement="H">
    <img src="/images/7d5bb35acc.png" style="width:80.0%" alt="FlashAttention v1内外循环" />
    <figcaption>FlashAttention v1内外循环</figcaption>
    </figure>

2.  <strong class="list-label">通过重计算（Recomputation）减少内存占用</strong>：为了进一步优化内存，FlashAttention在前向传播时不会保存用于反向传播的中间注意力矩阵（S和P）。在反向传播时，它会利用保存在SRAM上的输入块以及少量统计数据，快速地重新计算出这些中间值。这种“以时间换空间”的策略极大地节省了HBM的占用 。

### 7.1.4 实现细节

<span id="sec-8-2-4"></span>

#### 7.1.4.1 原始Attention计算

<span id="sec-8-2-4-1"></span>

$Attention(Q, K, V) = softmax((Q * K^T) / √d_k) * V$

缩放部分 $\sqrt{d_k}$ 的计算不占额外存储，可忽略掉。主要考量 $Q * K^T 、softmax()、* V$ 这三个运算

#### 7.1.4.2 FlashAttention的分块计算流程

<span id="sec-8-2-4-2"></span>

<figure data-latex-placement="H">
<img src="/images/df802fe248.png" style="width:80.0%" alt="分块计算" />
<figcaption>分块计算</figcaption>
</figure>

1.  首先，将 Q 矩阵 (形状为$(N,d)$) 按照<strong class="key-term">“行”</strong>切为 $T_r$ 块（block），每块的长度为 $B_r$ 。用 $Q_i$ 来表示切完后的某块矩阵，则 $Q_i$ 的维度为 $(B_r, d)$ 。不难理解，$Q_i$ 中存储着某 $B_r$ 个token的query信息。

2.  然后，将 $K^T$ 矩阵(形状为$(d,N)$)按照<strong class="key-term">“列”</strong>切为 $T_c$ 块，每块的长度为 $B_c$ 。用 $K^T_j$ 表示切完后的某块矩阵，则 $K_j^T$ 的维度为 $(d,B_c)$ 。易知 $K_j^T$ 中存储着某 $B_c$ 个token的key信息。

3.  同样，将 V 矩阵也按照<strong class="key-term">“行”</strong>切为 $T_c$ 块，每块长度为 $B_c$ 。用 $V_j$ 表示切完后的某块矩阵，则 $V_j$ 的维度为 $(B_c,d)$ 。易知 $V_j$ 中存储着某 $B_c$ 个token的value信息。

<strong class="key-term">计算初始attention分数</strong>：

$S_{i j}=Q_{i} * K_{j}^{T}=\left(B_{r}, d\right) *\left(d, B_{c}\right)=\left(B_{r}, B_{c}\right)$

图中的 $S_{ij}$ 表示前 $B_r$ 个token和前 $B_c$ 个token间的原始相关性分数。

具体循环中的操作示意：

<figure data-latex-placement="H">
<img src="/images/7d5bb35acc.png" style="width:80.0%" alt="FlashAttention v1内外循环" />
<figcaption>FlashAttention v1内外循环</figcaption>
</figure>

#### 7.1.4.3 softmax 计算操作

<span id="sec-8-2-4-3"></span> 原始softmax计算：$softmax(x_i) = exp(x_i - m) / Σ_j exp(x_j - m)$ 其中m是某一行列的最大值，减去它是为了数值稳定性。

定义:

- $m(x)$ : 标准场景下，该行的全局最大值

- $m(x^{(1)})$ : 分块1的全局最大值

- $m(x^{(2)})$ : 分块2的全局最大值

那么易知: $m(x) = m([x^{(1)}, x^{(2)}]) = \max(m(x^{(1)}), m(x^{(2)}))$

- $f(x)$ : 标准场景下，$e^{x - m(x)}$ 的结果

- $f(x^{(1)})$ : 分块场景下，$e^{x^{(1)} - m(x^{(1)})}$ 的结果

- $f(x^{(2)})$ : 分块场景下，$e^{x^{(2)} - m(x^{(2)})}$ 的结果

那么易知: $f(x) = [e^{m(x^{(1)})-m(x)} f(x^{(1)})$, $e^{m(x^{(2)})-m(x)} f(x^{(2)})]$

- $l(x)$ : 标准场景下，$\mathrm{rowsum}[f(x)]$ 的结果

- $l(x^{(1)})$ : 分块场景下，$\mathrm{rowsum}[f(x^{(1)})]$ 的结果

- $l(x^{(2)})$ : 分块场景下，$\mathrm{rowsum}[f(x^{(2)})]$ 的结果

那么易知: $l(x) = [l(x^{(1)}), l(x^{(2)})]$。

<span class="key-formula">$\mathrm{softmax}(x) = \frac{f(x)}{l(x)} = \frac{[e^{m(x^{(1)})-m(x)} f(x^{(1)}),e^{m(x^{(2)})-m(x)} f(x^{(2)})]}{e^{m(x^{(1)})-m(x)} l(x^{(1)}) + e^{m(x^{(2)})-m(x)} l(x^{(2)})}$</span>

分块计算操作：

$\begin{aligned} m(x) & =\max _{i}\left(x_{i}\right) \\ f(x) & =\left[e^{x_{1}-m(x)}, \ldots, e^{x_{n}-m(x)}\right] \\ l(x) & =\sum_{i} f(x)_{i} \\ \operatorname{softmax}(x) & =\frac{f(x)}{l(x)} \end{aligned}$

#### 7.1.4.4 FlashAttention的在线Softmax算法

<span id="sec-8-2-4-4"></span>

FlashAttention的核心思想是：我们可以<strong class="key-term">迭代地</strong>更新Softmax的计算结果。每当一个新的数据块到来时，我们用它来<strong class="key-term">修正</strong>之前基于旧数据块计算出的（不完整的）结果。

让我们把注意力计算中的一行（例如，$Q_i$和所有$K_j$的点积）看作是我们要计算Softmax的输入向量 $x$。这个向量 $x$ 被分成了多个块 $x_1, x_2, ..., x_{T_c}$。

算法为这一行维护三个核心变量：

- $O$: 当前的输出（对V的加权和）。

- $m$: 到目前为止见过的所有$x$元素中的最大值（<strong class="key-term">running maximum</strong>）。

- $l$: 到目前为止Softmax分母的累加和（<strong class="key-term">running denominator</strong>）。

现在，我们模拟算法的迭代过程：

<strong class="key-term">假设我们已经处理了第 1 到 j-1 个块，得到了当前的统计量 $m_{old}$ 和 $l_{old}$，以及当前的输出 $O_{old}$。</strong>

<strong class="key-term">现在，我们处理第 j 个块，$x_j$。</strong>

1.  <strong class="key-term">计算当前块的局部统计量</strong>

    我们只看当前块 $x_j$，可以计算出它的：

    - <strong class="list-label">局部最大值:</strong> $m_j = max(x_j)$

    - <strong class="list-label">局部Softmax分子:</strong> $P_tilde_j = e^{x_j - m_j}$ (注意，减去的是局部最大值)

    - <strong class="list-label">局部Softmax分母:</strong> $l_j = sum(P_tilde_j)$

2.  <strong class="key-term">合并统计量，更新全局最大</strong>

    新的全局最大值，一定是旧的全局最大值和当前块最大值中的较大者：$m_{new} = max(m_{old}, m_j)$

3.  <strong class="key-term">重新缩放（Re-scaling）并更新分母 $l$</strong>

    这是最关键的数学技巧。我们之前的 $l_{old}$ 是基于 $m_{old}$ 计算的，而 $l_j$ 是基于 $m_j$ 计算的。现在我们有了新的全局最大值 $m_{new}$，我们需要把这两部分都统一到新的基准上再相加。

    $l_{new} = l_{old} * e^{m_{old} - m_{new}} + l_j * e^{m_j - m_{new}}$

    <strong class="key-term">理解这个公式</strong>:

    - $l_{old}$ 最初是 $Σ e^{x_{old} - m_{old}}$。乘以 $e^{m_{old} - m_{new}}$ 后，它就变成了 $Σ e^{x_{old} - m_{new}}$。

    - $l_j$ 最初是 $Σ e^{x_j - m_j}$。乘以 $e^{m_j - m_{new}}$ 后，它就变成了 $Σ e^{x_j - m_{new}}$。

    - 两者相加，就得到了处理完 ‘j‘ 块数据后，基于新的全局最大值 $m_{new}$ 的正确的分母总和！

4.  <strong class="key-term">重新缩放并更新输出 $O$</strong>

    输出 $O$ 是对值矩阵 $V$ 的加权和。它的更新逻辑和分母 $l$ 完全一样，也需要重新缩放。

    $O_{new} = (O_{old} * l_{old} * e^(m_{old} - m_{new}) + (P_tilde_j * V_j) * e^(m_j - m_{new})) / l_{new}$

    <strong class="key-term">理解这个公式</strong>:

    - 分子部分 ( ... ) 是计算了未归一化的加权和，并将其统一到了 $m_{new}$ 基准上。

    - $P_tilde_j * V_j$ 是当前块 $x_j$ 对 $V_j$ 的加权和。

    - 最后，除以新的、正确的总分母 $l_{new}$，就得到了更新后的、正确的输出 $O$。

在论文的算法图中，这个公式被写作： $O_i ← diag(l_i^{new})^-1 ( diag(l_i) * e^{(m_i - m_i^{new})} * O_i + e^{(m_ij - m_i^{new})} * P -tilde_{ij} * V_j )$

这只是上述逻辑的矩阵化写法。$diag(l)$乘以$O$是为了还原出未归一化的加权和。

在反向传播中，因为中间计算的结果没有存储，所以要重新计算一下。

### 7.1.5 Flash Attention的计算量分析

<span id="sec-8-2-5"></span>

主要设计两部分矩阵计算：

对于 $S_{ij} = Q_i K_j^T$，其中 $Q_i \in \mathbb{R}^{B_r \times d}, K_j^T \in \mathbb{R}^{d \times B_c}$。根据前置知识，求 $S_{ij}$ 的计算量为 $O(B_r B_c d)$。

对于 $\tilde{P}_{ij} V_j$，其中 $\tilde{P}_{ij} \in \mathbb{R}^{B_r \times B_c}, V_j \in \mathbb{R}^{B_c \times d}$。则这里的计算量同样为 $O(B_r B_c d)$。

接下来我们看一共计算了多少次（1）和（2），也就是执行了多少次内循环： $T_c T_r = \frac{N}{B_c} \frac{N}{B_r}$

综合以上三点，flash attention的forward计算量为： $O\left(\frac{N^2}{B_c B_r} B_r B_c d\right) = O(N^2 d)$

注意，因为计算量是用大O阶表示的，所以这里我们把常数项都省略了。

#### 7.1.5.1 IO复杂度分析

<span id="sec-8-2-5-1"></span>

<table>
<thead>
<tr>
<th style="text-align: left;">

<strong>操作 (Operation)</strong>

</th>

<th style="text-align: left;">

<strong>标准注意力 (Standard Attention)</strong>

</th>

<th style="text-align: left;">

<strong>FlashAttention</strong>

</th>

<th style="text-align: left;">

<strong>备注 (Remarks)</strong>

</th>

</tr>
</thead>
<tbody>
<tr>
<td colspan="4" style="text-align: left;">

<strong>HBM -&gt; SRAM (读操作)</strong>

</td>

</tr>
<tr>
<td style="text-align: left;">

读 $Q, K, V$

</td>

<td style="text-align: left;">

$O(Nd)$

</td>

<td style="text-align: left;">

$O(Nd)$

</td>

<td style="text-align: left;">

这是无法避免的初始加载。

</td>

</tr>
<tr>
<td style="text-align: left;">

读中间矩阵 $S$ 或 $P$ ($N \times N$)

</td>

<td style="text-align: left;">

$O(N^2)$

</td>

<td style="text-align: left;">

$\emptyset$ (不读取)

</td>

<td style="text-align: left;">

核心区别: 标准注意力需将 S/P写回HBM再读出, FlashAttention完全避免了这 一步。

</td>

</tr>
<tr>
<td colspan="4" style="text-align: left;">

<strong class="key-term">SRAM -&gt; HBM (写操作)</strong>

</td>

</tr>
<tr>
<td style="text-align: left;">

写中间矩阵 $S$ 或 $P$ ($N \times N$)

</td>

<td style="text-align: left;">

$O(N^2)$

</td>

<td style="text-align: left;">

$\emptyset$ (不写入)

</td>

<td style="text-align: left;">

核心区别: FlashAttention将 中间计算结果保留在SRAM 中, 用完即弃。

</td>

</tr>
<tr>
<td style="text-align: left;">

写最终输出 $O$

</td>

<td style="text-align: left;">

$O(Nd)$

</td>

<td style="text-align: left;">

$O(Nd)$

</td>

<td style="text-align: left;">

最终结果必须写回。

</td>

</tr>
<tr>
<td style="text-align: left;">

<strong>总 I/O 复杂度</strong>

</td>

<td style="text-align: left;">

<strong>$O(N^2 + Nd)$</strong>

</td>

<td style="text-align: left;">

<strong>$O(Nd)$</strong>

</td>

<td style="text-align: left;">

当 $N \gg d$ 时, $N^2$ 项成为

</td>

</tr>
<tr>
<td style="text-align: left;">



</td>

<td style="text-align: left;">



</td>

<td style="text-align: left;">



</td>

<td style="text-align: left;">

不可逾越的瓶颈。

</td>

</tr>
</tbody>
</table>

## 7.2 FlashAttention2 与 FlashAttention1 的区别

<span id="sec-8-3"></span>

v2主要提升有两点：

1.  <strong class="key-term">引入多线程提升了并行的能力</strong>

2.  <strong class="key-term">继续优化了细节，比如延迟归一化使得内部的除法运算进一步减少。</strong>

v2 最主要的改进：原v1在进行$Q、K^T、V$的分块计算时，是把$K^T$放到最外层循环的，$Q、V$放到内层循环。v2将$Q$放到最外层，$K^T、V$放到了内层。

从图里面可以看到，对于v1，$K^T$是外层循环，这时候每次和$Q$相乘的内循环得到的中间结果是红色圈中的<strong class="key-term">“列”</strong>，这个列是没办法直接和$V$相乘得到完整的结果的，它需要等到下一次$K^T$的循环得到下一个列与$V$相乘得到的结果求和才是最终的$O$。

而对于v2,Q是外层循环，Q的每一行是每次循环的变量，这时候这一行可以和所有的内循环的$K^T$列相乘得到中间结果的行，即图中画蓝、红、紫的圈，这几个行可以单独与$V$也进行相乘，得到最终结果$O_{00}$、$O_{10}$等单个元素，不需要等待下一次循环。这样做好处就是可以把$Q$按照行进行各自独立的分割，每一行之间互不影响，就可以放到多个线程里面并行优化。

<figure data-latex-placement="H">
<img src="/images/75f994157f.png" style="width:80.0%" alt="FlashAttention v2内外循环" />
<figcaption>FlashAttention v2内外循环</figcaption>
</figure>

### 7.2.1 并行策略的进一步增强

<span id="sec-8-3-1"></span>

- <strong class="note-label">FlashAttention-1</strong>: 其并行化主要在 <strong class="key-term">批次（batch size）和头（head）</strong> 这两个维度上进行 。每个注意力头分配一个线程块 。当序列很长而批次和头数较少时，可并行的线程块数量可能远少于GPU上的流多处理器（SM）数量，导致GPU利用率低下 。

- <strong class="note-label">FlashAttention-2</strong>: 在V1的基础上，增加了在<strong class="key-term">序列长度（sequence length）维度上的并行化</strong> 。

  - <strong class="list-label">前向传播</strong>：将Q矩阵按行切分，不同的<strong class="key-term">行块</strong>（row blocks）分配给不同的线程块并行计算 。

  - <strong class="list-label">反向传播</strong>：由于梯度计算的依赖关系不同（dK和dV需要按行累加），为了优化，反向传播是按<strong class="key-term">列块</strong>（column blocks）分配给不同的线程块并行计算的 。

### 7.2.2 Warp间工作分区的优化：减少内部通信

<span id="sec-8-3-2"></span>

即使在单个线程块（Thread Block）内部，V2也优化了工作分配方式。一个线程块由多个Warp（通常每个包含32个线程）组成。

- <strong class="list-label">FlashAttention-1</strong>: 采用了一种称为 <strong class="key-term">“Split-K”</strong> 的策略。它将$K$和$V$矩阵沿着序列维度切分给不同的Warp，而$Q$块对所有Warp可见 。这种方式的缺点是，每个Warp计算出的是部分结果，为了得到最终的行输出，必须将这些中间结果写入共享内存（SRAM），进行Warp间的同步和聚合（相加），这引入了不必要的通信和共享内存读写开销，成为性能瓶颈 。

- <strong class="list-label">FlashAttention-2</strong>: 转而采用 <strong class="key-term">“Split-Q”</strong> 策略。它将$Q$矩阵切分给不同的Warp，而$K$和$V$块对所有Warp可见 。因为不同$Q$的计算是独立的，每个Warp在计算完自己的 $(Q_slice * K^T) * V$ 后，直接得到最终输出 $O$ 的一个分片，<strong class="key-term">无需与其他Warp进行通信或聚合</strong> 。这极大地减少了共享内存的读写和同步开销，是V2速度提升的关键之一。

### 7.2.3 算法层面的优化

<span id="sec-8-3-3"></span>

FlashAttention-2 在算法层面进行了一些精简，以减少开销较高的非矩阵乘法浮点运算（non-matmul FLOPs）。

- <strong class="list-label">延迟归一化 (Rescaling)</strong>: 在V1中，每一次内循环迭代，都需要用当前的局部softmax分母 $l$ 来重新缩放（rescale）累积的输出 $O$，这涉及到除法运算 。FlashAttention-2 优化了这一点，在内循环中，它只更新 $O$ 的“未归一化”版本，将最终的除法操作<strong class="key-term">延迟到外循环结束时才执行一次</strong> 。这个调整减少了中间步骤的除法运算次数，更充分地利用了为矩阵乘法优化的Tensor Cores 。

- <strong class="list-label">反向传播存储优化</strong>: V1为了反向传播时的重计算，需要从HBM中读取并保存两个统计量 $m$ (最大值) 和 $l$ (exp分数总和) 。V2则将它们<strong class="key-term">合并，只存储一个 $L = m + log(l)$（即log-sum-exp）</strong>，在反向传播时由此恢复所需信息 。这进一步减少了一次HBM的I/O操作，节省了内存带宽 。

### 7.2.4 FlashAttentionv3

<span id="sec-8-4"></span>

相比于FlashAttentionv1、v2，v3没有太多算法上的创新，主要有两个性能提升的点：

1.  <strong class="key-term">矩阵计算(GEMM)和softmax计算流水线化</strong>，交错并行，不严格顺序执行，而是会有时间的重叠。

2.  <strong class="key-term">硬件加速的 FP8 低精度计算与精度保持</strong>。

## 7.3 Triton

<span id="sec-8-5"></span>

[参考资料 7.2](/part-2/chapter-7#read-8-3)

Triton 是一个由 OpenAI 开发的、基于 <strong class="key-term">Python</strong> 的编程语言的编译器，可以让我们用类似 Python 的语法为 GPU 编写高效的<strong class="key-term">计算内核（Kernel）</strong>。这样就不用学习复杂的 CUDA C++ 编程，就能写出优化 CUDA 代码性能的<strong class="key-term">自定义算子</strong>，降低了 GPU 编程的门槛。

我们以实现Triton的向量加法为例子：

假设我们想实现一个很简单的 “向量 + 向量 → 向量” 的加法计算。绝大多数情况我们都不会考虑底层优化，而是直接在PyTorch上直接 <strong class="key-term">“c = a + b”</strong> 了，这样做当然可以，而且Pytorch也会有它自己的底层优化机制。但如果你想做一个 PyTorch 没有的、非标准的、或者多个操作想融合在一起的操作（比如 <strong class="key-term">“c = a \* b + sin(a)”</strong>），PyTorch 就需要依次调用多个内核，达不到最佳性能。

这个时候如果我们还想实现最佳的性能，方式就是写一个最佳的优化算子，一套“组合拳”直接实现“a \* b + sin(a)” 计算。传统的方式就是手动写CUDA C++，直接与GPU底层交互，不过需要我们自己管理线程块（Thread Blocks）、线程（Threads）、共享内存（Shared Memory）、内存对齐和合并访问（Coalescing）等一系列复杂的底层细节。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 算子融合</p>

算子融合是将多个独立的小操作（如乘法、加法、Mask 操作）合并成一个单一的、更高效的“大”操作（称为 <strong class="critical-term">Kernel</strong>）。这大大减少了从内存中读取和写入数据的次数，也减少了启动 GPU 计算任务的开销。

</div>

这个时候 Triton 的作用就得以体现，它让我们用类似 Python 的较为简单的语法来实现底层CUDA优化的性能。

### 7.3.1 Triton的核心思想:并行化与分块(Parallelism and Tiling)

<span id="sec-8-5-1"></span>

Triton 编程的核心是<strong class="key-term">并行化</strong>。我们不会写一个程序来处理整个矩阵 <strong class="key-term">“X”</strong>，而是 会写一个<strong class="key-term">程序模板（Triton Kernel）</strong>，GPU 会同时启动成百上千个这个程序的<strong class="key-term">实例</strong>（Program Instance），每个实例只负责一小部分工作。

还是以我们之前说的向量加法为例子：

```python
    import torch
    import triton
    import triton.language as tl
    
    @triton.jit # Triton内核函数装饰器，只有装饰器内的函数才会被Triton编译
    def add_kernel(
            x_ptr,  # 指向输入向量a的指针，就和C语言的指针差不多，直接和显存地址交互
            y_ptr,  # 指向输入向量b的指针
            output_ptr,  # 指向输出向量c的指针
            n_elements,  # 向量中的元素总数
            BLOCK_SIZE: tl.constexpr,  # 块大小，一个编译时常量
    ):
            # --- 内核的执行逻辑 ---
            # GPU 通过大规模并行来获得高性能。Triton 会启动很多个 add_kernel 的程序实例来同时处理数据。pid 就是当前这个实例的唯一编号（从0开始）。
            # 计算当前这个程序实例的ID
            pid = tl.program_id(axis=0)
            #计算当前程序实例要处理的数据块的偏移量
            #tl.arange生成一个 [0, 1, 2, ..., BLOCK_SIZE-1] 的数组
            block_start = pid *BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE) #偏移量 它定义了当前这个 pid 对应的程序实例负责处理哪一段数据。
            
            # 创建一个mask，防止内存越界
            # 比如向量长度是1000，BLOCK_SIZE是1024，最后一个块会访问到不存在的元素
            mask = offsets < n_elements
            
            # 从全局内存（HBM）加载数据块
            # mask=mask保证了只加载有效的数据
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            
            # 执行计算
            output = x + y
            
            # 将计算结果写回全局内存
            tl.store(output_ptr + offsets, output, mask=mask)
```

接下来我们将在CPU上调用这段函数

```python
    size = 98432    #创建输入数据
    a = torch.randn((size,), device='cuda')
    b = torch.randn((size,), device='cuda')
    output = torch.empty_like(a) # 准备一个空的输出tensor
    # 定义执行配置
    BLOCK_SIZE = 1024
    # grid这个元组定义了我们要启动多少个 add_kernel 的程序实例。
    grid = (triton.cdiv(size, BLOCK_SIZE),) # cdiv是向上取整的除法
    # 方括号里传入grid配置  像调用普通函数一样传入参数
    add_kernel[grid](a, b, output, size, BLOCK_SIZE=BLOCK_SIZE)
    # 验证结果
    print(torch.allclose(output, a + b)) # 输出 True
```

总结下来有如下几个要点：

- Triton内核函数一定要加 <strong class="key-term">@triton.jit</strong> 修饰符

- 每个内核函数在被实际调用的时候都只是众多实例中的一个，它只负责部分计算，所以要确保它的计算在正确的地址上（对应<strong class="key-term">输入的指针+单位偏移量offset</strong>）

- 有了指针地址后才能从GPU上加载到数据，执行完我们想要的运算之后在写回去。

- 一些细节比如<strong class="key-term">mask</strong>是考虑到了数据块的大小和真实输入的大小。

### 7.3.2 Triton的”分块”特性

<span id="sec-8-5-2"></span> 其实Triton的分块性能更好，刚才向量加法的例子没能体现，接下来我们讲解讲义上矩阵乘法的例子来充分证明这一点。

对于矩阵乘法‘C = A @ B‘ ，我们不直接计算C，而是把C<strong class="key-term">分块(tiling)</strong>，我们利用Triton启动很多个实例，每个实例就去计算其中的一个小块，对于每一个小块的计算，执行以下步骤：

 a. 从输入矩阵 ‘A‘ 中加载一<strong class="key-term">行块</strong>，从矩阵 ‘B‘ 中加载一<strong class="key-term">列块</strong>。

b\. 执行 ‘tl.dot‘（点积）运算，得到一个中间结果。

c\. 将 ‘A‘ 的指针向右移动一个块的距离，‘B‘ 的指针向下移动一个块的距离，加载下一对块。

d\. 执行 ‘tl.dot‘ 并将结果<strong class="key-term">累加</strong>到上一步的结果上。

e\. 重复 c, d 步，直到遍历完 ‘A‘ 的所有列和 ‘B‘ 的所有行。

<figure data-latex-placement="H">
<img src="/images/747a63cb4a.png" style="width:80.0%" alt="Triton矩阵乘法" />
<figcaption>Triton矩阵乘法</figcaption>
</figure>

除此之外，我们会使用 <strong class="key-term">tl.make_block_ptr()</strong> 这一智能指针来替代我们之前的手写的方式

```python
    a_block_ptr = tl.make_block_ptr(    # 创建一个描述 A 矩阵上数据块的“智能指针”
    base=a_ptr,                           # 矩阵的基地址
    shape=(M, K),                         # 整个矩阵的形状
    strides=(stride_am, stride_ak),       # 矩阵的步长
    offsets=(pid_m, BLOCK_SIZE_M, 0),    # 当前块的起始偏移 (行, 列)
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), # 定义要操作的块的形状
    order=(1, 0)                          # 内存布局顺序，通常是 (1, 0)，代表行主序，即先行后列，(0,1)就反过来)
    )

    # 在循环中，使用 tl.advance 来移动指针
    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K)) 
    #语义非常清晰：“在第1个维度（K维）上前进 BLOCK_SIZE_K”

    #加载时，使用 boundary_check 自动处理边界，不用mask了
    a = tl.load(a_block_ptr, boundary_check=(0, 1)) # 检查第0和第1个维度
```

### 7.3.3 Triton和C/C++的编译方式的区别和联系

<span id="sec-8-5-3"></span>

C/C++ 编译器比较通用，负责将通用程序翻译成 CPU 或 GPU 代码，开发者需要手动处理大量底层细节。而 Triton 编译器<strong class="key-term">垂直性更强</strong>，它专门针对 GPU 高性能计算，接收高度抽象的、以<strong class="key-term">数据块</strong>为单位的指令，并<strong class="key-term">自动</strong>完成最困难的性能优化，将开发者从复杂的 GPU 硬件细节中解放出来。

<strong class="critical-term">C/C++ (含 CUDA C++)</strong>: 编程模型更底层。在为 GPU 编写 CUDA C++ 时，你需要手动管理：

- <strong class="list-label">线程层级</strong>: 精确定义线程块（Block）和网格（Grid）的维度。

- <strong class="list-label">内存管理</strong>: 手动管理共享内存的分配和同步，以实现线程间通信和数据重用。

- <strong class="list-label">内存访问</strong>: 程序员需要自己精心设计内存访问模式，以确保“合并访问”，避免性能大幅下降。

<strong class="critical-term">Triton</strong>: 编程模型抽象层次更高。你不再直接操作单个线程，而是从<strong class="key-term">程序实例的视角</strong>出发，操作<strong class="key-term">数据块。</strong>

- <strong class="list-label">块级编程</strong>: 你通过 ‘tl.program_id‘ 获取当前程序实例的ID，然后加载、计算和存储一整块数据（例如 ‘tl.load(ptr + offsets)‘）。

- <strong class="list-label">自动优化</strong>: 你无需关心共享内存、合并访问或线程同步。Triton 编译器会分析你的块级操作，<strong class="key-term">自动生成</strong>管理共享内存、保证合并访问的底层 PTX 代码。它为你处理了最复杂的部分。

## 7.4 用Triton优化FlashAttention

<span id="sec-8-6"></span>

### 7.4.1 前向传播

<span id="sec-8-6-1"></span>

<figure data-latex-placement="H">
<img src="/images/9f7b94f13a.png" style="width:80.0%" alt="FlashAttention计算示意图" />
<figcaption>FlashAttention计算示意图</figcaption>
</figure>

就像我们之前提到的，我们想要实现性能的提高，就要尽可能地<strong class="key-term">减少对HBM的读写</strong>，因为计算速度远大于通信速度。这也是我们利用Triton优化FlashAttention的核心思想。

在讲义中，我们Triton被要求设置的网格维度是 (Tq,batch_size),也就是并行的两个维度。事实上，Triton可以并行的数量不止是两个维度，并且一般FlashAttention并行的维度是三个，分别是(Tq,batch_size,head)。讲义为了简化，只要求并行两个维度，取消了多头。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么在FlashAttention中Triton除了批次和多头之外只并行了Q?</p>

还是我们之前所说的，我们的核心思想是为了尽可能减少通信，用计算代替通信。如果把K也并行起来，那么每次一个小线程就是计算一小块Q和一小块K的点积，而最后要得到softmax的结果，需要一行所有的点积和，这就需要每个线程块之间进行通信，这就会大大降低性能。只并行Q，那么一个线程就可以得到S一行的所有值，就不必再通信了。

</div>

在每一个线程的前向传播中，$K^T$的小块是它的列，$V$的小块是它的行。我们中间会得到$S = \frac{QK^T}{\sqrt{d_k}}$的小块，用$M$来维护$S$每一行的最大值，$L$来维护$S$每一行的exp分数总和(随着循环计算的进行，$M$和$L$会不断更新)。然后我们用$P = \exp(S - M)$的小块来得到softmax的结果，用$O = PV$的小块来得到输出。

<div class="custom-block tip">

<p class="custom-block-title">薇言大义</p>

这里我觉得很值得仔细想清楚的一点是后面的O矩阵的计算，它不是想前面S那样，每次一个线程就得到一块S_ij,而是每次一个线程就得到一整行O_i,然后随着循环不断叠加更新，最后得到最终的O_i。

</div>

```python
    Q_i = tl.load(Q_block_ptr) #(Q_TILE_SIZE, D)
    O_i_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) #(Q_TILE_SIZE, D)
    L_i_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32) #(Q_TILE_SIZE, 1)
    M_i_acc = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=tl.float32) #(Q_TILE_SIZE, 1)
    # 外层循环：对 K/V 的序列长度进行分块
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr) #(K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr) #(K_TILE_SIZE, D)

        S_ij = tl.dot(Q_i, K_j.T) * scale #(Q_TILE_SIZE, K_TILE_SIZE)
        M_i_new = tl.max(S_ij, axis=1, keep_dims=True) #(Q_TILE_SIZE, 1)
        P_ij = tl.exp(S_ij - M_i_new) #(Q_TILE_SIZE, K_TILE_SIZE)
        L_i_new = tl.exp(M_i_acc - M_i_new) * L_i_acc + tl.sum(P_ij, axis=1, keep_dims=True) #(Q_TILE_SIZE, 1)
        #数据对齐
        P_ij_cast = P_ij.to(V_block_ptr.type.element_ty)#这代表从指针全局内存中读取类型，用V_j.dype理论上也行。
        O_i_new = tl.exp(M_i_acc - M_i_new) * O_i_acc + tl.dot(P_ij_cast, V_j)  #(Q_TILE_SIZE, D)
        #与pytorch代码不同，这里必须显式地更新M_i_acc和O_i_acc，因为下一个循环不会记住上一次的值
        M_i_acc = M_i_new
        O_i_acc = O_i_new
        L_i_acc = L_i_new

        # 只有 K 和 V 指针在 K 维度上前进，遍历所有 K/V 分块
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    O_i = O_i_acc / L_i_acc #全局归一化
    L_i = M_i_acc + tl.log(L_i_acc) #这个保存下来给反向传播用
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)
```

### 7.4.2 反向传播

<span id="sec-8-6-2"></span>

反向传播计算公式:

$$
\begin{aligned}
S &= \frac{QK^T}{\sqrt{d}} \\
P_{ij} &= \exp\left(S_{ij} - L_i\right) \\
dV &= P^T dO \\
dP &= dO V^T \\
dS_{ij} &= P_{ii} \circ (dP_{ii} - D_i) \\
dQ &= \frac{dS K}{\sqrt{d}} \\
dK &= \frac{dS^T Q}{\sqrt{d}}
\end{aligned}
$$

<figure data-latex-placement="H">
<img src="/images/fd1cc8920b.png" style="width:80.0%" alt="反向传播数学公式推导" />
<figcaption>反向传播数学公式推导</figcaption>
</figure>

反向传播并不会直接利用前向传播的所有计算结果，而是需要<strong class="key-term">重新计算</strong>，同样还是为了尽可能减少通信，用计算代替通信。

反向传播也要像前向传播那样按照循环来写，只是会更容易，因为参数里面会给$Q, K, V, O, L$，所以总体上只需要按照公式来

<strong class="key-term">反向传播的逻辑（链式法则）</strong>：根据链式法则，一个变量的梯度等于所有流经它的路径上的梯度之和。

对于$dK$。$K$的第$j$个分块$K_j$，在前向传播中，它不仅和$Q$的第0个分块$Q_0$作用了，也和$Q_1,Q_2, ...Q_{Tq-1}$都发生了作用。

当我们的外层循环$i=0$时，我们计算了$K_j$因为和$Q_0$相互作用而产生的梯度贡献。当外层循环$i=1$时，我们又计算了$K_j$因为和$Q_1$相互作用而产生的梯度贡献。...以此类推。

所以，$K_j$的<strong class="key-term">总梯度</strong>，必须是它与<strong class="key-term">所有</strong>$Q_i$相互作用产生的梯度的<strong class="key-term">总和</strong>。这就是为什么用+=来进行累加。如果用=直接赋值，那么当外层循环$i$走到下一个值时，上一次计算出的梯度贡献就会被覆盖和丢失，最后得到的$dK$是不完整的，只反映了$K$和最后一个$Q$分块$Q_{Tq-1}$交互的梯度。

$dV$的计算也是完全相同的道理。

对于$dQ$，逻辑是类似的，只是循环的内外层反了过来。在计算$dQ$的第$i$个分块$dQ_i$时，它也需要累加来自所有$K_j$和$V_j$的梯度贡献，所以在我们的实现中，$dQ[:, i * Bq:(i+1) * Bq, :] += dQ_{ij}$ 也是在做累加。

与前向传播不同，反向传播是<strong class="key-term">把K放在了外层循环，把Q放在了内层循环</strong>。这是为了让每一个并行的计算单元（线程块）都能独立、无冲突地完成一整块梯度（dK或dV）的计算，并将累加过程完全限制在各自高速的SRAM中。不过就算反过来也可以，只是会增加时延。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 只从公式里面看，好像dQ和dK不都分别依赖了所有K和Q吗？那凭什么要把K放在外层循环，把Q放在内层循环？</p>

这是一处非常重要的细节，把K放在外层是很重要的操作，这样做是因为如果把Q放在外层，需要很多线程块之间进行通信(下面会具体展开)，而把K放在外层，就可以巧妙避免。

</div>

1.  <strong class="list-label">已知:</strong> $Q, K, V, O, L, dO$;还可先求出 $D = \text{rowsum}(dO \circ O)$(逐元素相乘)

2.  <strong class="list-label">外层循环:</strong> 遍历$K$和$V$的块$j$。

    - 加载$K$ 和 $V$ 整个矩阵

3.  <strong class="list-label">内层循环:</strong> 遍历$Q$、$O$、$dO$的块$i$。

    - 加载$Q_i$、$O_i$、$dO_i$ $dQ$整个矩阵

    - 先计算 $S_{ij} = Q_i * K_j^T / sqrt(d)$，得到$S_{ij}$的小块

    - 再计算 $P_{ij} = exp(S_{ij} - L_i)$，得到$P_{ij}$的小块

    - 之后可以累加得到$dV_j += P_{ij}^T @ dO_i$，即$dV_j$的小块

    - 然后求$dS_{ij} = P_{ij} * (dP_{ij} - D_i)$，得到$dS_{ij}$的小块($dP_{ij} = dO_i @ V_j^T$)

    - dS有两大用处，一是求$dK_j += dS_{ij}^T @ Q_i$，得到$dK_j$的小块，二是求$dQ_i += dS_{ij} @ K_j$，得到$dQ_i$的小块

那么，如果把Q放在外层会怎么样？

1.  <strong class="list-label">外层循环：</strong>遍历$Q$、$O$、$dO$的块$i$。

    - 加载$Q_i$、$O_i$、$dO_i$整个矩阵

2.  <strong class="list-label">内层循环:</strong> 遍历$K$和$V$的块$j$。

    - 加载$K$ 和 $V$ $dQ$ 整个矩阵

    - 先计算 $S_{ij} = Q_i * K_j^T / sqrt(d)$，得到$S_{ij}$的小块

    - 再计算 $P_{ij} = exp(S_{ij} - L_i)$，得到$P_{ij}$的小块

    - 之后可以累加得到$dV_j += P_{ij}^T @ dO_i$，即$dV_j$的小块

    - 然后求$dS_{ij} = P_{ij} * (dP_{ij} - D_i)$，得到$dS_{ij}$的小块($dP_{ij} = dO_i @ V_j^T$)

    - dS有两大用处，一是求$dK_j += dS_{ij}^T @ Q_i$，得到$dK_j$的小块，二是求$dQ_i += dS_{ij} @ K_j$，得到$dQ_i$的小块

可以看到关键的差别就在$Q_i$、$O_i$、$dO_i$的读取和$K$ 和 $V$的读取上，后者的大小远大于前者，经过这么多次循环后，性能就产生了较大的差距。

| <strong>对比项</strong> | <strong>K/V在外</strong> | <strong>Q在外</strong> | <strong>性能影响</strong> |
|:---|:---|:---|:---|
| <strong>HBM读取 (K, V)</strong> | 整个矩阵只被读 <strong>1</strong> 遍 | 整个矩阵被读 <strong>$T_q$</strong> 遍 | <strong>灾难性的</strong> |
| <strong>HBM读取 (Q, O, dO)</strong> | 整个矩阵被读 <strong>$T_k$</strong> 遍 | 整个矩阵只被读 <strong>1</strong> 遍 | 占优 |
| <strong>核心权衡</strong> | `T_k` 通常远小于 `T_q`（例如序列长度8k，块大小128，则`T_q=64`；如果K/V序列长度一样，`T_k`也是64。但在交叉注意力等场景下，`T_k`可能很小）。官方方案让被读取次数更多的矩阵（Q）在内层循环，被读取次数更少的矩阵（K/V）在外层循环。<strong>但更关键的是，原子写比重复读更高效。</strong> | 你的方案让 `K` 和 `V` 被反复从HBM读取，这是GPU计算中最大的性能瓶颈。 |  |
| <strong>最终评价</strong> | <strong>最优</strong>。最大化了数据在SRAM中的复用，将HBM的访问降到了最低。 | <strong>性能极差</strong>。导致了对HBM的冗余、重复访问，完全违背了IO感知的算法设计原则。 |  |

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment2-systems)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 7.1 · FlashAttention

<span id="read-8-2"></span>

- <strong class="list-label">参考:</strong>FlashAttention–猛猿–知乎

- <strong class="list-label">falshAttention动画演示-bilibili</strong>

### 7.2 · Triton

<span id="read-8-3"></span>

- <strong class="list-label">Triton官方中文文档：</strong><https://triton-lang.cn/main/index.html>
