---
outline: [2, 3]
---

# 第 14 章 · 指令微调与偏好对齐理论

<span id="guide-ch-13"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[SFT 与专家迭代](/part-5/chapter-11#guide-ch-12)与[策略优化基础：从策略梯度到 PPO](/part-5/chapter-12#guide-ch-12-policy)。

<strong>准备工作：</strong>本章以概念、数学推导和训练机制为主；实验所需模型与数据在下一章统一说明。

<strong>本章任务：</strong>梳理 SFT 的序列组织、梯度累积、奖励模型和 RLHF，推导 DPO 目标并比较相关方法。

</div>

## 14.1 指令微调与序列组织

原始的数据一般是(prompt, response)这种二元的指令回复对,在这上面训练语言模型通常称为指令微调(或监督微调； SFT)。

本章先说明训练机制与优化目标；[指令微调与 DPO 实验](/part-6/chapter-15#guide-ch-13-experiments)再按基线、SFT、DPO 和最终评估的顺序展开。

### 14.1.1 Padding 和 Packing 的区别

#### 14.1.1.1 核心定义

- <strong class="list-label">Padding</strong>:在序列的末尾或头部<strong class="key-term">补齐特殊标识符</strong>(<strong class="key-term">如 &lt;pad&gt; Token</strong>),强制使得同一个 Batch 内所有不等长序列的张量维度达到等长。

- <strong class="list-label">Packing</strong>:将多个不等长的独立数据序列<strong class="key-term">首尾拼接</strong>,组合成一个长度等于模型预设最大上下文长度的<strong class="key-term">单一长序列</strong>,从而消除无效计算位,最大化硬件计算吞吐量。

#### 14.1.1.2 技术原理

Padding 的步骤

1.  <strong class="list-label">计算最大长度</strong>:在构建 Batch 时,遍历当前批次内的所有文本序列,提取长度最大值 $L_{max}$(或采用自己设定的全局最大长度)。

2.  <strong class="list-label">张量填充</strong>:对于长度 $L < L_{max}$ 的序列,在其一端填充 &lt;pad&gt; Token到长度为$L_{max}$ 。此时 Batch 输入张量的形状严格对齐为 (Batch_Size, $L_{max}$)。

3.  <strong class="list-label">构建 Attention Mask</strong>:生成与输入维度一致的二进制矩阵。真实 Token 的位置标记为 1,被填充的 &lt;pad&gt; Token 标记为 0。

4.  <strong class="list-label">注意力屏蔽</strong>:在自注意力机制(Self-Attention)计算 $QK^T$ 时,将 Mask 中为 0 的位置在注意力得分矩阵上加上负无穷($-\infty$)。经过 Softmax 激活函数后,这些位置的权重将严格等于 0,从而确保填充的无效 Token 不会参与模型的状态更新。

Packing 的步骤

1.  <strong class="list-label">序列拼接</strong>:从数据集中顺序读取独立序列,用特定标识符(如 &lt;eos&gt; Token)作为分隔,将其直接连接成一维长数组。

2.  <strong class="list-label">截断与分块</strong>:当拼接的数组长度达到模型支持的最大上下文长度或自己设定的最大长度(如 4096)时,进行截断,形成一个形状为 (1, 4096) 的密集张量。剩余部分留作下一个张量的起点。

3.  <strong class="list-label">隔离注意力污染</strong>:为了防止拼接在同一个序列中的不同文档在注意力机制中互相读取上下文,需要构建<strong class="key-term">块对角注意力掩码</strong>。即 A 的 Token 只能与 A 计算注意力,B 的 Token 只能与 B 计算。

4.  <strong class="list-label">重置位置编码</strong>:针对拼接张量内的每一个独立子序列,其绝对位置编码必须从 0 开始重新递增,以保证模型能够正确捕捉相对位置关系。

#### 14.1.1.3 具体示例

假设我们有三条数据(Token 化后):

- <strong class="list-label">序列 A:</strong>\[A1, A2\] (长度:2)

- <strong class="list-label">序列 B:</strong>\[B1\] (长度:1)

- <strong class="list-label">序列 C:</strong>\[C1, C2, C3\] (长度:3)

假定硬件要求的序列维度(Sequence Length)为 4。

Padding 处理结果:构成 Batch Size = 3 的张量矩阵,存在大量 PAD 产生的计算浪费。

Input Tensor Shape: (3, 4)

$$
\begin{bmatrix}
A1 & A2 & PAD & PAD \\
B1 & PAD & PAD & PAD \\
C1 & C2 & C3 & PAD
\end{bmatrix}
$$

Attention Mask

$$
\begin{bmatrix}
1 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
1 & 1 & 1 & 0
\end{bmatrix}
$$

Packing 处理结果: 序列 A, B, C 被紧密拼接,中间通过特殊符隔开或靠 Mask 控制,构成 Batch Size = 1 (或更少批次)的张量,0 浪费。

\[\[A1, A2, EOS, B1, EOS, C1, C2, C3\]\] 假设以 EOS 作为分隔拼接,Input Tensor Shape: (1, 8) 假设最大长度为8

Position IDs (每个子序列位置重置) \[\[0, 1, 2, 0, 1, 0, 1, 2\]\]

二维 Block-diagonal Mask (表示 B1 无法看到 A1, A2),1 表示可见,0 表示不可见。 从上到下的行分别代表A1, A2, EOS, B1, EOS, C1, C2, C3 的注意力视野

$$
\begin{bmatrix}
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 \\  
0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 \\  
0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\
$\dots$
\end{bmatrix}
$$

#### 14.1.1.4 使用场景

- <strong class="note-label">Padding 的使用场景</strong>:

  - <strong class="list-label">推理阶段</strong>:当模型输入通常是单条文本或少量文本时,Padding 实现更为简单直接,且对推理效率影响较小。

  - <strong class="list-label">小规模微调</strong>:当训练集的数据长度高度一致,或者整体数据量较小、不需要极致压榨 GPU 算力时,Padding 实现更为简单稳妥。

- <strong class="note-label">Packing 的使用场景</strong>:

  - <strong class="list-label">大规模预训练</strong>:面对数万亿 Token 的语料,必须使用 Packing 保证 GPU 矩阵计算单元持续满载,绝对杜绝 Padding 带来的算力空转。

  - <strong class="list-label">大规模指令微调(SFT)</strong>:在 Hugging Face 的 SFTTrainer 中开启 packing=True,能显著缩短训练时间,通过增加单次 Forward/Backward pass 包含的实际 Token 数量,提高梯度更新的有效性。

## 14.2 梯度累加(Gradient Accumulation)

<span id="part6-gradient-theory"></span>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么一定要梯度累加实现Batch Size放大的效果呢？反正实质上GPU也只能跑小Batch Size，并没有提升并行速度。</p>

在深度学习中，我们希望求得的是<strong class="key-term">整个训练集（全量数据）</strong>的真实梯度，以此来更新权重。因为全量数据太大算不过来，我们只能用一个 Batch 的数据算出的梯度去<strong class="key-term">“估计”</strong>全局真实梯度。根据大数定律，Batch Size 越大，这个“估计值”就越逼近真实情况。

当 Batch Size 极小时（如 BS=2），计算出的梯度带有极高的<strong class="key-term">随机噪音（Variance）</strong>。如果仅凭这 2 个样本的极端特征就立即调用 “optimizer.step()” 更新包LLM 权重，会导致模型的优化轨迹剧烈震荡。在这种高噪音下，为了防止模型崩溃，只能被迫使用极小的学习率（Learning Rate），这反而在宏观上大幅拖慢了模型的收敛速度，甚至会导致模型陷入局部最优解。

而大 Batch Size（如 BS=32 或更大）的数学平滑作用，将 32 个样本的梯度累加并求平均后，不同样本带来的随机噪音会在数学上互相抵消。最终产生的累加梯度，指向了一个更平滑、更准确的损失函数下降方向。

</div>

GPU 显存无法一次性容纳大 Batch 的前向传播（Forward）和反向传播（Backward）激活值（如 80GB 显存也只能装下 Batch Size = 2 的 LLM 训练）。但对 32 个样本同时求损失和梯度，等价于把这 32 个样本分成 16 组（每组 2 个样本），分别求梯度后再求平均。

如果直接累加 k 次梯度，最终的梯度总量会放大 k 倍，导致更新步长过大。因此，必须在反向传播前，将每次的 Loss 除以累加步数（“gradient_accumulation_steps”），确保最终累加和正好等于全部样本的平均梯度。

## 14.3 偏好对齐：奖励模型与 RLHF 流程

### 14.3.1 奖励模型训练(RM)

奖励模型是一个<strong class="key-term">评分模型</strong>,用于评价模型输出对人类偏好的符合程度。它接受“提示+模型回答”的输入,输出一个得分,分数越高表示回答越符合人类偏好。训练一个准确的RM是RLHF的关键步骤,因为<strong class="key-term">后续强化学习将以RM提供的奖励信号来优化策略模型</strong>。

训练RM需要人工偏好数据,通常是<strong class="key-term">同一提示下多种回答的比较/排序</strong>。典型做法是:

1.  使用上一步的SFT模型,对一批提示生成若干不同回答(例如4-9个)。

2.  根据质量对这些回答进行偏好排序(从最佳到最差)。

3.  从排序结果构造<strong class="key-term">成对偏好比较</strong>的数据:(Prompt, 回答A, 回答B, 标签),其中标签指示A是否优于B。通常把人工认为更好的回答作为<strong class="key-term">“chosen”</strong>,较差的作为<strong class="key-term">“rejected”</strong>,形成二元比较对。

例如一个偏好数据可以格式化为:

```python
"prompt":"给小学生解释什么是太阳能。",
"chosen":"太阳能是来自太阳的能量...(通俗易懂正确的解释)",
"rejected":"太阳能是一种魔法,可以让灯泡亮起来。(错误或不当的回答)"
```

开源的偏好数据集通常以这样的<strong class="key-term">“chosen”</strong>/<strong class="key-term">“rejected”</strong>字段形式提供。

奖励模型通常基于预训练模型构建,以确保理解复杂语言输入的能力匹配。但它在LM顶部增加一个回归头用于输出一个标量分数。具体实现上,可以复制SFT微调后的模型权重作为初始权重,然后修改最后一层为1维输出。这样RM的输入是完整的(prompt+answer)序列,输出一个logits值作为评分。

使用前述偏好比较数据,通过<strong class="key-term">监督学习</strong>训练RM。例如采用<strong class="key-term">对比损失</strong>:对于每对(chosen, rejected),令模型输出的分数为$S_c$和$S_r$。我们希望$S_c > S_r$,且差距越大越好,这就代表我们想要的结果和干扰的噪声结果区分度更大,这相当于把偏好比较当作<strong class="key-term">二分类任务</strong>:模型判断“chosen是否优于rejected”。训练过程中可将同一Prompt下的每个正负回答对视作一个训练样本,计算上述损失并反向传播调优RM参数。

常用损失是Logistic回归损失或对数sigmoid损失: $L=−logσ(S_c−S_r)$

其中$\sigma$是sigmoid函数。

训练得到的RM需要在之后的强化学习阶段保持<strong class="key-term">权重冻结</strong>,作为固定的奖励评估器使用。

如果RM不准确,会直接导致策略模型学习到错误的优化方向。

### 14.3.2 基于人类反馈的强化学习(RLHF)

<strong class="note-label">整体流程:</strong>在拥有SFT微调模型(作为初始策略)和训练好的奖励模型之后,即可进入RLHF阶段,即用<strong class="key-term">强化学习进一步微调模型参数</strong>,使其倾向于输出高奖励的响应。常用算法是<strong class="key-term">PPO</strong>,一种策略梯度方法。RLHF通过试错式训练进一步提升模型回答质量和对齐程度,被广泛证明能显著提高模型在人类偏好、内容安全等维度的表现。

RLHF训练引入了几个重要的模型和概念:

- <strong class="note-label">策略模型(policy model)</strong>:即我们要优化的语言模型。在开始RL时通常直接使用SFT后的模型参数作为初始策略$\pi_{\theta}$。

- <strong class="note-label">参考模型(reference model)</strong>:一份策略模型初始参数的冻结副本$\pi_{\text{ref}}$。参考模型用于计算新策略与原策略之间的差异(<strong class="key-term">KL散度</strong>)以施加正则,防止策略偏离初始行为过多。

- <strong class="note-label">奖励模型(reward model)</strong>:前一步训练得到的RM,用来对策略模型生成的回复给出奖励分数R,指引策略优化方向。

- <strong class="note-label">价值函数(value function)</strong>:Critic网络,近似估计每个状态(或每个token位置)的预期累计奖励。它的作用是用于<strong class="key-term">计算优势函数</strong>A=R - V,<strong class="key-term">减少策略梯度的方差</strong>。

在工程实现中,我们会:

- 复制一份SFT模型权重作为<strong class="critical-term">“ref-model”</strong>冻结。

- 在SFT模型上添加value head得到<strong class="critical-term">“ppo-model”</strong>,此时<strong class="critical-term">“ppo-model”</strong>有两个头:原来的语言模型头用于生成文本,新的价值头用于估值。

- <strong class="note-label">训练循环:</strong>PPO训练通常以下面的方式迭代进行:

  1.  <strong class="list-label">生成数据</strong>:从训练的提示集合中抽取一批prompts,用当前策略模型(<strong class="critical-term">“ppo-model”</strong>)对每个prompt生成回复(通常采用一定随机性如<strong class="key-term">温度采样</strong>,以促进策略探索不同回答)。生成长度可限制在一定范围以控制计算成本。

  2.  <strong class="list-label">计算奖励</strong>:对每个prompt的生成回复,使用<strong class="key-term">冻结的RM</strong>打分,得到奖励分数R。此外,计算新回复相对于参考模型的<strong class="key-term">KL惩罚</strong>:即统计<strong class="critical-term">“ppo-model”</strong>在生成回复上的概率分布与<strong class="critical-term">“ref-model”</strong>分布的KL散度。通过RM得分和KL值,可构造一个<strong class="key-term">正则化后的奖励</strong>: $R' = R - \beta \cdot \text{KL}$,其中$\beta$是惩罚系数。这表示模型在得到高奖励的同时会因偏离参考而被扣分,从而平衡探索和保守。

  3.  <strong class="list-label">优势估计</strong>:使用价值函数,计算每个生成序列的值估计V,并基于最终的R’计算优势$A = R' - V$。对于序列中每个token位置,也可以计算每步的优势,但通常简化为把最终总优势赋给序列中的各token(使用<strong class="key-term">GAE</strong>等方法平滑计算)。

  4.  <strong class="list-label">PPO更新</strong>:将生成的prompt和模型输出(以及计算出的优势A和参考概率分布)作为一次<strong class="key-term">交互轨迹</strong>,基于PPO算法更新策略模型参数。PPO的损失包括:策略损失(提高高优势动作的概率,降低低优势动作概率,且用剪切机制限制更新幅度)、值函数损失(使价值头预测更贴近实际R’)以及策略与参考的KL正则项。通过反向传播调整$\pi_{\theta}$的参数。

  5.  <strong class="list-label">重复迭代</strong>:更新后,用更新后的策略继续采样新数据,不断循环。

RLHF也存在一些典型问题:

- <strong class="note-label">Reward Hacking</strong>:模型可能学到投机策略骗过了我们预期的RM,即出现了<strong class="key-term">Reward Hacking</strong>。这会导致我们模型跑偏。比如我们短视频的推荐算法奖励是用户停留时长,即看的时间越长,越会给用户推荐此类视频。但一不小心出现了色情暴力的视频,用户停留时间长满足了奖励,但这显然不是我们想要的结果。

- <strong class="note-label">不稳定性</strong>:相比监督学习,RL训练的收敛更不稳定、调参难度高。不同随机种子、奖励权重都会影响结果。需要较多试错和经验去达到理想效果。

### 14.3.3 其他后训练方法及其工程特点

除上述经典的SFT-RM-PPO流程外,业界和学术界也提出了一些<strong class="key-term">替代性的后训练方法</strong>,旨在提高训练效率、稳定性或减少对人类标注的依赖。

- <strong class="note-label">直接偏好优化(DPO)</strong>:Direct Preference Optimization 是一种<strong class="key-term">无需显式奖励模型RW和在线RL</strong>的偏好对齐方法。DPO使用离线收集的<strong class="key-term">偏好比较数据</strong>(与RM训练相同的数据),但直接通过跟偏好相关的损失函数来微调模型,使其对人类偏好更高的回答给出更高概率。

  具体而言,DPO保留一个参考模型(SFT模型参数冻结)作为基准,用损失函数鼓励模型对于<strong class="key-term">‘chosen‘</strong>回答的生成概率高于参考模型,对<strong class="key-term">‘rejected‘</strong>回答的概率低于参考模型。这样模型在不经过PPO的情况下,就隐式地优化了偏好差值,同时通过参考模型起到KL正则作用。

  $\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$

- <strong class="note-label">RRHF</strong>:Rank Responses to Align Language Models with Human Feedback,是另一种离线偏好对齐方法。RRHF利用<strong class="key-term">多个候选回复的排序信息</strong>训练模型:对于同一prompt,模型对不同响应计算长度归一化的序列概率,对比这些概率的大小与人类偏好顺序,使模型调整自身概率分布来匹配人类偏好。

  - <strong class="list-label">奖励模型(RM)评分:</strong> ‘Score(R1)=0.9‘, ‘Score(R2)=0.3‘, ‘Score(R3)=0.7‘

  - <strong class="list-label">处理流程:</strong>

    1.  <strong class="list-label">RM评分排序为:</strong>R1 \> R3 \> R2。

    2.  <strong class="list-label">计算损失:</strong>

        - 比较 (R1, R2): 模型为R1赋予的概率应高于R2,否则产生损失。

        - 比较 (R1, R3): 模型为R1赋予的概率应高于R3,否则产生损失。

        - 比较 (R3, R2): 模型为R3赋予的概率应高于R2,否则产生损失。

    3.  将所有这些成对的损失加总,形成最终的batch loss。

- <strong class="note-label">RLAIF(Reinforcement Learning from AI Feedback)</strong>:这是<strong class="key-term">用AI替代人类反馈</strong>的范式。RLAIF流程与RLHF一样,唯一区别是把人类偏好标签换成由强大的AI教师模型来自动生成。

## 14.4 DPO 原理推导与方法比较

### 14.4.1 DPO是什么，为什么我们需要它?

<strong class="critical-term">直接偏好优化（Direct Preference Optimization, DPO）</strong> 是一种用于<strong class="key-term">大语言模型对齐(Alignment)</strong>的技术，旨在让模型输出更符合人类偏好。与RLHF的方法不同，DPO <strong class="key-term">不需要单独训练奖励模型</strong>，而是直接利用人类偏好数据对模型进行微调。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

即DPO的数据形式是$(x, y^+, y^−)$这种，明确告诉你$y^+$好，$y^-$不好，调 policy目标就是为了让它更偏向$y^+$而远离$y^-$。说白了，这就已经不是强化学习了，而是<strong class="key-term">监督学习</strong>。 所以，容易理解，这个DPO适合的场景就是那种<strong class="key-term">二选一</strong>的，比如两个商品去推断哪个用户更喜欢。

</div>

RLHF 过程复杂且不稳定，需要多个阶段（<strong class="key-term">监督微调、奖励建模和强化学习</strong>）才能完成模型优化。具体来说，RLHF 方法存在以下挑战：

- <strong class="list-label">训练复杂度高</strong>：RLHF 包含训练奖励模型并在策略优化中反复采样模型输出进行强化学习。训练过程需要精心设计的奖励函数、策略梯度算法（如 <strong class="key-term">PPO</strong>），实现和调参难度大。

- <strong class="list-label">不稳定性</strong>：强化学习阶段往往不够稳定，相同的数据和超参数可能出现截然不同的结果。模型在RL过程中可能发生<strong class="key-term">Reward Hacking</strong>或策略坍塌等问题，需要引入KL惩罚项保持模型行为不要偏离初始模型太远。

- <strong class="list-label">计算成本高</strong>：由于每一步优化都依赖于奖励模型的反馈，RLHF 训练往往需要反复生成样本评估，涉及四个模型协同（策略、参考、奖励、价值模型），使单步训练耗时且资源占用大。

而DPO计算开销更低、实现更简单，但在对齐效果上却不输RLHF。DPO 从理论上将 RLHF 中的奖励优化问题转化为了一个<strong class="key-term">简单的偏好分类问题</strong>，<strong class="key-term">省略了显式的奖励模型和强化学习环节</strong>。

### 14.4.2 简要回顾 SFT、Reward Model、RLHF经典post-training 流程

- <strong class="note-label">监督微调（SFT）</strong>：指在预训练大模型的基础上，使用<strong class="key-term">高质量有标注的数据</strong>对模型进行<strong class="key-term">有监督训练</strong>，使其更好地完成下游任务。例如 ChatGPT 的训练第一步，就是用大量人工编写的<strong class="key-term">指令-回答示例</strong>对模型进行微调，得到一个顺从指令的SFT模型。这个 SFT 模型在回答形式、语气上更接近人类期望，但仍可能低质量的输出。

- <strong class="note-label">奖励模型（Reward Model）</strong>：为了进一步让模型输出符合人类偏好，研究者会让模型针对各种提示生成多个候选回答，然后请人类标注哪一个回答更好。这些人工比较结果形成了<strong class="key-term">偏好数据对</strong> 。有了这些偏好对，传统 RLHF 方法会训练一个<strong class="key-term">奖励模型 (RM)</strong>，其实质是一个<strong class="key-term">二分类模型</strong>，它输入模型输出文本，给高质量回应打高分，劣质回应打低分。

- <strong class="note-label">人类反馈强化学习（RLHF）</strong>：在这一步，使用上面的奖励模型来指导策略模型（通常是SFT模型的<strong class="key-term">副本</strong>，即<strong class="key-term">Actor模型</strong>）进行强化学习微调。典型方法是<strong class="key-term">近端策略优化 (PPO)</strong>：让 Actor 生成回答，根据奖励模型打分作为即时奖励，同时与一个<strong class="key-term">冻结的参考模型 (Reference Model)</strong>对比计算KL惩罚，最后用策略梯度提升 Actor 对高分回答的概率、降低低分回答的概率。

SFT -\> 偏好收集得到奖励函数 -\> RLHF (PPO) 这是post-training的标准传统pipeline。DPO所做的改变就是它试图<strong class="critical-term">绕过奖励模型和复杂的RL寻优，直接基于偏好数据对来优化模型参数</strong>，实现与RLHF相同的目标。

### 14.4.3 数学原理详细推导

#### 14.4.3.1 核心定义与符号体系

定义以下数学符号：

- <span class="key-formula">$x$</span>：输入，从数据集分布 $\mathcal{D}$ 中采样。

- <span class="key-formula">$y$</span>：输出。

- <span class="key-formula">$y^+, y^—$</span>：成对的偏好数据，其中 $y^+$ 是被人类标注者偏好的回答，$y^-$ 是被拒绝的回答。

- <span class="key-formula">$\pi_{\text{ref}}(y|x)$</span>：参考策略，通常是SFT后的模型。我们希望优化后模型不要偏离它太远。

- <span class="key-formula">$\pi_\theta(y|x)$</span>：我们正在训练的策略网络，参数为 $\theta$。

- <span class="key-formula">$r^*(x, y)$</span>：潜在的真实奖励函数，但在DPO中我们不需要显式建模它。

- <span class="key-formula">$\beta$</span>：KL散度惩罚系数，控制策略偏离参考策略的程度。

- <span class="key-formula">$\sigma(\cdot)$</span>：Sigmoid函数，$\sigma(z) = \frac{1}{1+e^{-z}}$。

#### 14.4.3.2 数学原理推导

DPO 利用了<strong class="key-term">变分法（Calculus of Variations）</strong>中的思想。

<strong class="note-label">第一阶段：重写RLHF的目标函数</strong>

在标准的RLHF（如PPO阶段）中，我们的目标是最大化期望奖励，同时施加KL散度惩罚以防止模型偏移旧模型太多。优化目标如下：

$\max_{\pi} \mathcal{J}(\pi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$

这实际上是一个<strong class="key-term">带约束的优化问题</strong>。虽然PPO通过迭代采样来近似求解，但在数学上，这个目标函数存在一个<strong class="key-term">解析解</strong>。

<strong class="note-label">第二阶段：最优策略的解析形式</strong>

上述目标函数本质上等价于最小化以下KL散度：

$\min_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}} \left( \pi(y|x) \; \Bigg\| \; \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) e^{\frac{1}{\beta} r(x,y)} \right) \right]$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) e^{\frac{1}{\beta} r(x,y)}$ 是配分函数（Partition Function）。

证明如下：

<figure data-latex-placement="H">
<img src="/images/5f5e903d7d.png" style="width:80.0%" alt="DPO数学原理证明" />
<figcaption>DPO数学原理证明</figcaption>
</figure>

根据<strong class="key-term">吉布斯不等式</strong>，当且仅当两个分布相等时，KL散度最小（为0）。因此，对于固定的奖励函数 $r(x,y)$，<strong class="key-term">最优策略 $\pi^*(y|x)$</strong> 必须满足：$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) e^{\frac{1}{\beta} r(x, y)}$

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 吉布斯不等式</p>

核心原理：<strong class="key-term">在所有概率分布中，真实分布能让“交叉熵”或“相对熵”达到最小。</strong>

设$P = (p_1, p_2, \dots, p_n)$ 为真实分布，$Q = (q_1, q_2, \dots, q_n)$ 为拟合的分布，都是定义在同一集合上的概率分布（$p_i \ge 0, q_i \ge 0$，且和为 1）。

- $P = (p_1, p_2, \dots, p_n)$ 为真实分布

- $Q = (q_1, q_2, \dots, q_n)$ 为拟合的分布

都是定义在同一集合上的概率分布（$p_i \ge 0, q_i \ge 0$，且和为 1）。

那么有：$\sum_{i=1}^n p_i \ln p_i \;\ge\; \sum_{i=1}^n p_i \ln q_i$ 并且等号成立当且仅当 $p_i = q_i$ 对所有 i 都成立。

把上面的式子移项，就得我们熟悉的东西： $\sum_{i=1}^n p_i \ln \frac{p_i}{q_i} \;\ge\; 0$

这就是我们一直接触的 <strong class="key-term">KL 散度（相对熵）</strong>：$D_{\mathrm{KL}}(P \| Q) \ge 0$

所以实际上<strong class="key-term">吉布斯不等式 = KL 散度非负性</strong>

用信息论的知识去直观理解：<strong class="key-term">用 Q 去“假装”数据来自真实的 P，付出的平均编码代价一定比用 P 自己编码更大或扯平。</strong>

</div>

通常在RLHF中，我们用神经网络去逼近这个 $\pi^*$。但在DPO中，我们利用这个等式进行<strong class="key-term">代数变换</strong>。

<strong class="note-label">第三阶段：奖励函数的逆向表示</strong>

现在，我们把上面的等式反过来，用最优策略 $\pi^*$ 和参考策略 $\pi_{\text{ref}}$ 来表示奖励函数 $r(x, y)$。

对等式两边取对数：

$\begin{aligned} \log \pi^*(y|x) &= \log \left( \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) e^{\frac{1}{\beta} r(x, y)} \right) \\ \log \pi^*(y|x) &= -\log Z(x) + \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) \end{aligned}$

移项解出 $r(x, y)$：

$\begin{aligned}\frac{1}{\beta} r(x, y) = \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x)\\ r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \end{aligned}$

这个公式告诉我们：<strong class="key-term">任何一个奖励函数，都可以等价地映射为一个最优策略与参考策略的对数比率，加上一个仅与 x 有关的常数项。</strong>

<strong class="note-label">第四阶段：代入Bradley-Terry模型</strong>

在人类偏好建模中，我们通常假设偏好服从 <strong class="key-term">Bradley-Terry (BT) 模型</strong>。即，回答 $y^+$ 优于 $y^-$ 的概率取决于两者奖励差值的Sigmoid： $P(y^+ \succ y^- | x) = \sigma(r(x, y^+) - r(x, y^-))$

现在，我们将第三阶段推导出的 $r(x, y)$ 代入上式。 注意，当我们计算 $r(x, y^+) - r(x, y^-)$ 时，<strong class="key-term">配分函数项 $\beta \log Z(x)$ 会因为相减而自动抵消（因为它只与 x 有关，与 y 无关）</strong>：

$$
\begin{aligned}
r(x, y^+) - r(x, y^-) &= \left( \beta \log \frac{\pi^*(y^+|x)}{\pi_{\text{ref}}(y^+|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y^-|x)}{\pi_{\text{ref}}(y^-|x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi^*(y^+|x)}{\pi_{\text{ref}}(y^+|x)} - \beta \log \frac{\pi^*(y^-|x)}{\pi_{\text{ref}}(y^-|x)}
\end{aligned}
$$

<strong class="note-label">第五阶段：最终的DPO损失函数</strong>

现在，我们将参数化的策略 $\pi_\theta$ 作为 $\pi^*$ 的近似。我们的目标是最大化观测到的偏好数据的似然度。

实际中目标函数就转换为最小化负对数似然：

<span class="key-formula">$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$</span>

这就是DPO的最终公式。 这个损失对每条偏好数据$(x, y^+, y^-)$ 都在推高输出$y^+$ 相对概率、压低输出$y^-$相对概率。同时，参考模型的对数概率差作为一个常数项嵌入其中，约束$\pi_\theta$ 不要偏离$\pi_{\text{ref}}$ 太远。

概括而言，DPO 的核心原理就是将<strong class="key-term">“最大化奖励减KL”问题等价转化为对比两段输出谁更好的分类问题</strong>。模型通过直接优化偏好数据的似然来隐式达到与RLHF相同的效果。

值得一提的是，β 超参数的选择十分重要：它决定了DPO更新的“温和”程度。较大的 β（如0.5以上）意味着模型几乎保持原状，只做细微调整；较小的 β（如0.1）则赋予模型更大自由去迎合偏好，但过小可能造成模型过度偏离原本分布。

### 14.4.4 DPO、PPO、GRPO三者比较

| <strong>维度</strong> | <strong>PPO</strong> | <strong>DPO</strong> | <strong>GRPO</strong> |
|:---|:---|:---|:---|
| 算法类型 | 强化学习（RL） | 监督学习（非传统RL） | 强化学习（PPO 变体） |
| 是否需要奖励模型 | 需要 | 不需要 | 不需要显式 |
| 训练信号来源 | 奖励模型打分 | 人类/模型偏好对（chosen vs rejected） | 同一 prompt 下多回答的相对好坏 |
| 优化目标 | 最大化期望奖励 + KL 约束 | 直接最大化偏好概率 | 最大化组内相对优势 |
| 是否在线采样 | 是 | 否（离线） | 是 |
| 训练稳定性 | 中等（需调参） | 很稳定 | 比 PPO 稳定 |
| 实现复杂度 | 高 | 低 | 中 |
| 计算/显存开销 | 高 | 低 | 中 |
| 典型应用 | ChatGPT 早期 RLHF | SFT 后对齐主流方案 | 大模型 RL 对齐（DeepSeek 系） |

### 14.4.5 其他相关前沿研究

<span id="sec-alignment-ipo-kto"></span>

前面已经把 DPO 写成了直接优化语言模型的偏好损失。沿着这条思路，还可以追问两个问题：<strong class="key-term">同一对回答之间的偏好差距，应当被拉大到什么程度？如果只有单条回答的点赞或点踩，又该怎样训练？</strong>下面分别用 IPO 和 KTO 回答这两个问题。

先统一记号。沿用前文的 $x$、$y^+$、$y^-$ 和冻结的参考策略 $\pi_{\mathrm{ref}}$，定义

$$
\begin{aligned}
u_\theta(x,y)
&=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},\\
h_\theta(x,y^+,y^-)
&=u_\theta(x,y^+)-u_\theta(x,y^-).
\end{aligned}
$$

$u_\theta$ 表示某个回答相对于参考模型的 log-probability 变化，$h_\theta$ 则表示两个回答的这种变化之差。它比较的是<strong class="key-term">相对于参考模型的偏好变化</strong>，不能直接解释为当前模型对两个回答的绝对概率差。下文简写 $h=h_\theta(x,y^+,y^-)$，DPO 的单样本损失就是

$$
\ell_{\mathrm{DPO}}(h)=-\log\sigma(\beta h).
$$

#### 14.4.5.1 IPO：让偏好差距有一个有限目标

<span id="sec-alignment-ipo"></span>

<strong class="note-label">从 DPO 的梯度出发。</strong> 对一条始终标注为 $y^+\succ y^-$ 的偏好记录，求导可得

$$
\frac{\partial\ell_{\mathrm{DPO}}}{\partial h}
=-\beta\sigma(-\beta h)<0.
$$

因此，单独看这条记录，继续增大 $h$ 总能降低损失；梯度虽然会逐渐变小，但没有有限的零点。这与 Bradley–Terry 模型中的关系一致：

$$
r(x,y^+)-r(x,y^-)=\log\frac{p}{1-p}.
$$

当经验偏好看起来接近确定性，即 $\hat p\to1$ 时，对应的奖励差趋于无穷。<strong class="key-term">有限数据里没有观察到反对票，并不等于真实偏好具有无限强度。</strong>如果模型反复拟合这样的记录，就可能把偏好差距推得过大。这里描述的是单个可分偏好对的优化倾向，并不是说所有 DPO 训练都会发散；共享参数、相互冲突的数据和训练步数都会影响实际结果。

<strong class="note-label">从偏好概率到有界的优化信号。</strong> IPO（Identity Preference Optimization）从更一般的 $\Psi$PO 框架出发。令 $\mu$ 为采集比较回答的行为策略，$\rho$ 为问题分布，该框架考虑

$$
\begin{aligned}
J_\Psi(\pi)
={}&\mathbb E_{\substack{x\sim\rho,\ y\sim\pi(\cdot\mid x)\\
                         y'\sim\mu(\cdot\mid x)}}
       [\Psi(p^*(y\succ y'\mid x))]\\
&-\tau\mathbb E_{x\sim\rho}
       D_{\mathrm{KL}}(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)).
\end{aligned}
$$

其中 $\tau>0$ 是正则系数；这里采用 IPO 笔记的记号，不要求它与前面 DPO 的 $\beta$ 取相同数值。在 Bradley–Terry 假设成立时，选择 logit 映射可与前面的奖励优化联系起来；IPO 选择<strong class="key-term">恒等映射 $\Psi(p)=p$</strong>，直接使用范围在 $[0,1]$ 内的偏好概率。

固定一个问题 $x$，定义回答对行为策略的平均胜率

$$
g_x(y)=\mathbb E_{y'\sim\mu(\cdot\mid x)}[p^*(y\succ y'\mid x)].
$$

这样，IPO 就变成了以 $g_x(y)$ 为优化信号的 KL 正则问题。复用前面 DPO 推导中的求解方式，有

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
                 \exp\!\left(\frac{g_x(y)}{\tau}\right),
\qquad
h_{\pi^*}(x,y,y')=\frac{g_x(y)-g_x(y')}{\tau}.
$$

区别在于：$g_x(y)$ 是有界的平均胜率，不会因为一个经验概率等于 1 就产生无穷大的 logit。

<strong class="note-label">落到可以训练的平方损失。</strong> 真实的 $g_x(y)$ 通常不可直接观测，手里仍然只有 $(x,y^+,y^-)$。在同一问题下独立采样两个回答、再采样偏好标签的设定中，可以把上述目标转为下面的经验损失：

<div class="key-formula">

$$
\mathcal L_{\mathrm{IPO}}(\theta)
=\mathbb E_{(x,y^+,y^-)\sim\mathcal D}
 \left[\left(h_\theta(x,y^+,y^-)-\frac{1}{2\tau}\right)^2\right].
$$

</div>

这里的 $1/(2\tau)$ 来自对偏好标签及回答顺序取期望后的转换。<strong class="key-term">不能把一次二元标注直接当作两个回答平均胜率之差的精确值</strong>；完整转换讨论的是期望目标，而非每个样本逐点相等。

与 DPO 对照，IPO 的单样本梯度为

$$
\frac{\partial\ell_{\mathrm{IPO}}}{\partial h}
=2\left(h-\frac{1}{2\tau}\right).
$$

当 $h$ 小于目标时，梯度下降推动它增大；超过目标时，则把它拉回。因此，<strong class="critical-term">IPO 不只是给 DPO 加了一个常数，而是把 logistic 偏好损失改成了具有有限目标的平方损失。</strong>

例如取 $\tau=1$，目标差距为 $0.5$。若当前 $h=0.2$，梯度为 $-0.6$，更新倾向于增大差距；若 $h=0.8$，梯度为 $0.6$，更新倾向于缩小差距。相比之下，DPO 对这两个正向标注的样本都仍有增大 $h$ 的倾向。有限目标缓解了这一方向的过度拟合压力，但不意味着 IPO 对所有数据都更好，也不能代替验证集评估。

#### 14.4.5.2 KTO：从成对比较到单条好坏反馈

<span id="sec-alignment-kto"></span>

IPO 改变了偏好差距的优化方式，但仍然需要成对数据。现实中还会遇到另一种情况：用户只对当前回答点了赞或点了踩，没有提供同一问题的另一个回答。此时数据形式是

$$
(x,y,c),\qquad c\in\{+1,-1\},
$$

其中 $c$ 是外部给定的好坏标签。KTO（Kahneman–Tversky Optimization）借鉴前景理论中的<strong class="key-term">参照点与收益、损失</strong>，用这种单条反馈来构造优化目标。

<strong class="note-label">回答与基线比较。</strong> 沿用上面的 $u_\theta$，把 KTO 的缩放系数单独记为 $\beta_K>0$，定义

$$
\begin{aligned}
z_\theta(x,y)&=\beta_K u_\theta(x,y),\\
z_{\mathrm{ref}}(x)
&=\beta_K\mathbb E_{y'\sim\pi_\theta(\cdot\mid x)}[u_\theta(x,y')]\\
&=\beta_K D_{\mathrm{KL}}
       (\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)),\\
a_\theta(x,y)&=z_\theta(x,y)-z_{\mathrm{ref}}(x).
\end{aligned}
$$

这里的 $z_{\mathrm{ref}}$ 是随当前策略变化的参照点；下标 ref 并不表示它是参考模型直接输出的质量评分。它衡量当前策略相对参考策略的整体偏移，$a_\theta$ 则比较某条回答的偏移是否高于这个基线。<strong class="key-term">回答好不好仍由标签 $c$ 决定</strong>，不能仅凭 $a_\theta>0$ 就认定回答正确或优质。

<strong class="note-label">正负样本的损失。</strong> 正样本希望 $a_\theta$ 增大，负样本希望它减小。引入两个正权重 $\lambda_+$、$\lambda_-$，可写成

<div class="key-formula">

$$
\ell_{\mathrm{KTO}}(x,y,c)=
\begin{cases}
\lambda_+\,[1-\sigma(a_\theta(x,y))],&c=+1,\\[2pt]
\lambda_-\,[1-\sigma(-a_\theta(x,y))],&c=-1,
\end{cases}
$$

</div>

$$
\mathcal L_{\mathrm{KTO}}(\theta)
=\mathbb E_{(x,y,c)\sim\mathcal D}[\ell_{\mathrm{KTO}}(x,y,c)].
$$

两个权重用来调节正负反馈的贡献；当两类数据数量不均衡时，样本比例与权重需要一起考虑。

训练时将基线作为停止梯度的量。固定基线，对 $z=z_\theta(x,y)$ 求导，有

$$
\frac{\partial\ell_{\mathrm{KTO}}}{\partial z}=
\begin{cases}
-\lambda_+\,\sigma(a)[1-\sigma(a)],&c=+1,\\[2pt]
\phantom{-}\lambda_-\,\sigma(a)[1-\sigma(a)],&c=-1.
\end{cases}
$$

因此，同样一个 $z=0.8$、基线为 $0.5$ 的回答，若被标为好，更新倾向于提高它的相对 log-probability；若被标为坏，更新方向相反。<strong class="key-term">标签决定方向，回答与基线之间的差距调节梯度强弱。</strong>Sigmoid 使单样本损失有界、两端梯度饱和，但并没有像 IPO 那样规定一个有限的目标差距，也不构成不会过度优化的保证。

<strong class="note-label">基线如何在工程上估计？</strong> 上面的定义需要对回答分布求期望，不能遍历整个生成空间。原论文的离线实现采用同一 batch 中错配的问题与回答来构造共享参照点。若有 $m$ 个样本，用一个无固定点的置换 $j(i)\ne i$ 组成 $m$ 对，可写为

$$
\widehat z_{\mathrm{ref}}
=\beta_K\max\!\left(0,\frac{1}{m}\sum_{i=1}^{m}
       u_\theta(x_i,y_{j(i)})\right).
$$

该量在损失中停止梯度。错配样本并非从 $\pi_\theta(\cdot\mid x_i)$ 抽取，因此<strong class="key-term">这是有偏的工程估计，不能称作精确 KL 或无偏蒙特卡洛估计</strong>。实现时还应区分用于计算基线的样本与带好坏标签的训练样本。

#### 14.4.5.3 把 DPO、IPO 与 KTO 放在一起看

三者都使用当前策略与参考策略的概率信息，但训练数据和损失的含义不同：

| <strong>比较项</strong> | <strong>DPO</strong> | <strong>IPO</strong> | <strong>KTO</strong> |
|:---|:---|:---|:---|
| 反馈形式 | 同一问题的偏好对 | 同一问题的偏好对 | 单条回答的好坏标签 |
| 核心比较 | 两个回答的相对变化 $h$ | $h$ 与有限目标 $1/(2\tau)$ | 单条回答与基线的差 $a$ |
| 损失特点 | 负对数 sigmoid | 目标差距的平方误差 | 按标签分支的有界 sigmoid 损失 |
| 关注问题 | 直接利用偏好对训练策略 | 控制偏好差距的优化强度 | 利用无需成对的二元反馈 |

由此看，IPO 主要追问<strong class="key-term">已有偏好对应该怎样优化</strong>，KTO 主要扩展<strong class="key-term">什么形式的反馈可以用于训练</strong>。它们不是必须依次执行的三个训练阶段。下一章仍以 DPO 实验为主；若要扩展实验，IPO 可以沿用偏好对的数据接口，而 KTO 需要好坏标签、正负样本权重以及基线估计。比较时应保持模型、数据划分与评估口径一致。

<strong class="note-label">原始论文：</strong> [IPO：A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)； [KTO：Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)。
