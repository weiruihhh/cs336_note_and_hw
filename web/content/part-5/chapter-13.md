---
outline: [2, 3]
---

# 第 13 章 · GRPO 原理与训练实验

<span id="guide-ch-12-grpo"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[策略优化基础：从策略梯度到 PPO](/part-5/chapter-12#guide-ch-12-policy)；复用[SFT 与专家迭代](/part-5/chapter-11#guide-ch-12)中的生成、评分和训练组件。

<strong>准备工作：</strong>准备 Qwen2.5-Math-1.5B、数学任务数据、评分函数与推理环境，保留基线及训练配置。

<strong>本章任务：</strong>将组内优势和策略损失对应到完整训练循环，区分 rollout、batch 与 microbatch，分析奖励稀疏等实验问题。

</div>

## 13.1 GRPO：组内优势与训练过程

一次迭代的主线是：对每个问题采样一组回答 → 评分 → 计算组内优势 → 计算回答 token 的概率与损失 → 更新策略。先说明这一过程的目标函数，再逐步对应到数据与计算。

不管是PPO还是TRPO,它们都是基于Actor-Critic结构的,想要得到优势函数$A^{\pi}(s,a) = Q^{\pi}(s,a) - V_{\phi}(s,a)$,需要训练一个Critic网络对状态价值函数$V_{\phi}(s,a)$进行估计,而GRPO和PPO最大的差别就是它<strong class="key-term">放弃了Critic网络</strong>,转而选择了<strong class="key-term">组内优化</strong>。

### 13.1.1 核心定义与符号体系

在开始推导之前,我们定义如下符号:

- <strong class="list-label">$\mathcal{Q}$</strong>:问题的分布。

- <strong class="list-label">q</strong>:从分布中采样的问题。

- <strong class="list-label">$\pi_\theta$</strong>:需要优化的策略网络,参数为 $\theta$。

- <strong class="list-label">$\pi_{ref}$</strong>:参考策略网络,通常是 SFT 之后的初始模型,用于约束 KL 散度。

- <strong class="list-label">$\pi_{\theta_{old}}$</strong>:在本次迭代更新前的旧策略网络。

- <strong class="list-label">G</strong>:Group Size,即针对同一个问题 q,我们采样的输出数量(例如 G=64)。

- <strong class="list-label">$\{\{o_1, o_2, ..., o_G\}\}$</strong>:针对问题 q,由 $\pi_{\theta_{old}}$ 采样生成的 G 个不同的输出。

- <strong class="list-label">r(o, q)</strong>:奖励函数,通常包含正确性奖励或过程奖励模型。

- <strong class="list-label">$A_i$</strong>:第 i 个输出的优势函数。

- <strong class="list-label">$\epsilon$</strong>:PPO 中的截断超参数。

- <strong class="list-label">$\beta$</strong>:KL 散度惩罚系数。

### 13.1.2 数学原理推导

#### 13.1.2.1 GRPO 的核心构建:去 Critic 化

GRPO 放弃了 $V_\phi(s)$。它利用大模型生成的多样性,对同一个问题 q 生成一组输出 $\{o_1, \dots, o_G\}$。

GRPO 假设:<strong class="key-term">这 G 个输出的奖励分布,本身就可以作为基线(Baseline)。</strong>

对于第 i 个输出 $o_i$,其对应的奖励为 $r_i = r(q, o_i)$。我们计算这组奖励的<strong class="key-term">平均值</strong>和<strong class="key-term">标准差</strong>: $\text{mean} = \bar{R} = \frac{1}{G} \sum_{i=1}^G r_i$

$\text{std} = \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \bar{R})^2 + \delta}$

<strong class="key-term">$\delta$ 为数值稳定极小值</strong>

#### 13.1.2.2 组内相对优势

GRPO 将优势函数定义为标准化后的奖励: $\hat{A}_i = \frac{r_i - \bar{R}}{\sigma_R}$

<strong class="list-label">数学性质</strong>:这里 $\bar{R}$ 充当了动态 Baseline。如果 $o_i$ 的得分高于组内平均分,则 $\hat{A}_i > 0$,模型会被鼓励增加生成该输出的概率；反之则被抑制。

#### 13.1.2.3 目标函数

GRPO 将上述优势代入 PPO 的 Clip 损失函数中,并增加了 KL 散度惩罚项以防止<strong class="key-term">模型崩溃(Reward Hacking)</strong>。

最终的 GRPO 目标函数 $J_{GRPO}(\theta)$ 为:

<div class="key-formula">

$$
J_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{Q}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \mathcal{L}_{clip}^{(i)} - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref})^{(i)} \right) \right]
$$

</div>

其中,$\mathcal{L}_{clip}^{(i)}$ 是标准的 PPO 截断损失:

$\mathcal{L}_{clip}^{(i)} = \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right)$

<strong class="key-term">这里的 KL 项通常采用近似计算:</strong>

$\mathbb{D}_{KL} \approx \frac{\pi_{\theta_{old}}(o_i|q)}{\pi_{ref}(o_i|q)} - \log \frac{\pi_{\theta_{old}}(o_i|q)}{\pi_{ref}(o_i|q)} - 1$

实际上主要区别还是在优势函数A的定义上,GRPO目标函数的形式和PPO基本上一样。

#### 13.1.2.4 Reward Hacking 的数学本质

Reward Hacking 通常表现为模型发现了一个极其生僻、反直觉但能骗取高分的动作序列 $a_{hack}$(例如在推荐系统里面我们设置奖励是用户停留时间,但发现用户在色情、暴力的界面停留时间长,助长了奖励)。

对于这种 $a_{hack}$,其特征是:

1.  <strong class="list-label">代理回报极高</strong>:$\tilde{R}(s, a_{hack}) \gg 0$

2.  <strong class="list-label">SFT概率极低</strong>:由于 $a_{hack}$ 不符合正常人类语言逻辑,参考模型 $\pi_{ref}$ 给它的概率极低,即 $\pi_{ref}(a_{hack}|s) \to 0$。

#### 13.1.2.5 KL 项的“反黑客”机制

现在,让我们看看当模型试图通过提高 $\pi_\theta(a_{hack}|s)$ 来利用这个漏洞时,总回报 $R_{total}$ 会发生什么变化。

假设模型将 $a_{hack}$ 的概率推高到显著水平(例如 $\pi_\theta(a_{hack}|s) \approx 1$)。此时 KL 惩罚项的变化如下:

$\text{Penalty} = - \beta \log \frac{\pi_\theta(a_{hack}|s)}{\pi_{ref}(a_{hack}|s)} \approx - \beta \log \frac{1}{\epsilon} \quad (\text{其中由于概率差别悬殊,使得 } \frac{1}{\epsilon} \to \infty)$

因此:$\text{Penalty} \to -\infty$

<strong class="list-label">结论推导</strong>: 即使 $\tilde{R}(s, a_{hack})$ 非常大(例如 100 分),只要该动作严重脱离了 $\pi_{ref}$ 的分布,KL 惩罚项(例如 -10000 分)就会瞬间吞噬掉所有奖励收益。

$R_{total} = \underbrace{\tilde{R}(s, a_{hack})}_{\text{Hacked Reward (High)}} - \underbrace{\beta \log \frac{\pi_\theta}{\pi_{ref}}}_{\text{KL Cost (Huge)}} < 0$

因此,在数学上,优化器会发现:<strong class="key-term">去触碰那些“非人类语言”的高分区域,得不偿失。</strong>

#### 13.1.2.6 损失函数

$\mathcal{L}_{total} = \mathcal{L}_{policy} + \beta \cdot \mathcal{L}_{KL}$

对于GRPO来说,因为本身也就没有Critic网络,所以和PPO比起来就没有$L_t^{VF}(\theta)$这一项,而且GRPO损失函数也包含了KL散度,因此也就没有再额外增加熵了。

### 13.1.3 直觉理解与物理意义

#### 13.1.3.1 物理直观理解

GRPO的组内寻找奖励的方式,本质思想是<strong class="key-term">输出的好不好,不是靠一个绝对分数,而是靠同一组里谁更好</strong>。就是<strong class="key-term">矮子里面拔大个儿</strong>,也许模型输出的这一组回答都不咋样,但还是要选择一个最好的,即便它可能绝对值不好(鸡头)。

#### 13.1.3.2 动态基线的妙处

在数学原理上,<span class="key-formula">$\hat{A}_i = \frac{r_i - \bar{R}}{\sigma_R}$</span> 利用了<strong class="key-term">蒙特卡洛采样来近似期望</strong>。

- <strong class="list-label">降低方差</strong>:对于一道极难的题,可能所有输出得分都很低。如果用绝对分数,模型收到的全是负反馈,效果就不好。但在 GRPO 中,只要你比同组的其他答案稍微好一点,你也能获得正向的梯度更新。这保证了<strong class="key-term">学习信号的稳定性</strong>。

- <strong class="list-label">无需 Critic 的本质</strong>:Critic 的本质是<strong class="key-term">拟合期望</strong>。当 G 足够大时,样本均值 $\bar{R}$ 就是奖励的无偏估计。GRPO 用<strong class="key-term">计算换空间</strong>(多采样几次输出,省去 Critic 显存)。

### 13.1.4 从公式到一次完整训练迭代

#### 13.1.4.1 采样(sample)

首先，对每个训练样本（例如每个<strong class="key-term">prompt</strong>问题）让当前的旧策略模型生成多个不同的回答（<strong class="key-term">response</strong>）。通常我们为每个prompt采样固定数量的输出，例如8个，以形成一个<strong class="key-term">候选回答组（group）</strong>。假设原始数据集中有100条prompt，那么采样后得到的回答矩阵形状就是 (100, 8)，即每个问题对应8个模型回答。这些回答都是依据<strong class="key-term">旧策略</strong>（即当前模型参数）独立采样得到的。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

在这里，“旧策略”指的是我们用于采样的模型策略参数（在训练过程中通常是前一轮优化后的策略，用作参考策略），而“新策略”指的是将要更新的模型参数。每轮采样都是基于旧策略来生成数据。

</div>

典型地，<strong class="key-term">组大小</strong> G 可以取4到8之间；研究表明在大型Transformer模型上，采用4～8个候选就足以取得良好的效果，进一步增大组的规模会继续降低梯度方差但回报递减。

#### 13.1.4.2 评分

对每个prompt获得的一组8个回答，我们需要评估它们的质量，从而得到奖励值(reward)。在GRPO框架中，可以使用<strong class="key-term">外部的评分函数</strong>或<strong class="key-term">奖励模型</strong>对回答进行评价。评分标准取决于具体任务，例如在LLM微调中可以依据回答的<strong class="key-term">正确性</strong>和<strong class="key-term">格式规范</strong>给出分数。如果有<strong class="key-term">标准答案（ground truth）</strong>，我们可以自动比较回答与标准答案的匹配程度并衡量正确性；对于格式要求明确的任务，也可以设计规则检查回答格式是否合规。这些评估得到的<strong class="key-term">原始奖励</strong>记作 <strong class="key-term">raw_rewards</strong>，其形状与采样结果一致(如100×8)，即每个问题的8个回答各有一个分数。

#### 13.1.4.3 组内归一化（计算优势 Advantage）

有了每个回答的 raw reward之后，GRPO的关键一步是在<strong class="key-term">每个prompt的回答组内对奖励进行归一化</strong>，以计算出<strong class="key-term">优势函数（advantage）</strong>值。具体做法是：<strong class="key-term">对同一prompt下的多个回答，其奖励减去该组奖励的平均值，再除以该组奖励的标准差</strong>（通常会加上一个很小的常数$\delta$防止除零）。用公式表示，每组中的第 i个回答的优势为：

$A_i \;=\; \frac{r_i \;-\; \mu_{group}}{\sigma_{group} + \delta}$

其中 $r_i$ 是第 i 个回答的原始得分，$\mu_{group}$ 是该prompt下8个回答得分的平均值，$\sigma_{group}$是它们的标准差。这样归一化后的优势值 $A_i$ 反映了每个回答<strong class="key-term">相对于同组其他回答的好坏程度</strong>：<strong class="key-term">高于0表示该回答比组内平均水平好，低于0表示比平均差。</strong>

采用<strong class="key-term">组内优势归一化</strong>有两个好处：

- <strong class="list-label">消除分数尺度的影响：</strong>归一化使我们关注的是相对排名而非绝对分值，不管原始打分函数的取值范围如何，归一化后每组内奖励的均值为0，方差为1。这样能够确保不同prompt之间的奖励分布具有可比性，也不受评分偏置的影响。模型只需判断“一组候选中哪些回答更优”即可。

- <strong class="list-label">降低梯度方差，稳定训练：</strong>这实际上提供了一种无需值函数基线的优势估计，相当于对策略梯度做了<strong class="key-term">减去baseline</strong>的处理，从而减少了策略梯度的方差。传统PPO依赖训练一个价值网络 V(s) 来估计baseline，而GRPO通过组内比较<strong class="key-term">省去了训练价值函数的繁琐</strong>。这不仅简化了算法（降低内存和算力开销），也提高了采样效率和稳定性。直观来说，<strong class="key-term">每次策略更新都会让模型更倾向于产生组内相对更好的回答，降低生成较差回答的概率</strong>。

#### 13.1.4.4 模型训练准备（数据序列化与概率计算）

经过以上步骤，我们已经得到了用于训练的数据：每个prompt的问题文本、模型生成的一个回答（将对8个回答分别进行训练）、以及对应的归一化优势值$A$。接下来，需要将文本和优势结合起来用于模型参数的更新。以下是这一阶段的具体步骤：

<strong class="critical-term">文本序列化与 Tokenization</strong>

<strong class="key-term">将prompt和对应的模型回答拼接成完整输入序列</strong>。例如，可以采用格式：“‘\<\|prompt\|\> 问题内容 ... \<\|response\|\> 回答内容 ...‘”。然后对这个序列进行<strong class="key-term">分词</strong>，转换为模型可处理的token ID表示。为了构造训练样本，我们通常将<strong class="key-term">prompt+response作为一个整体序列</strong>，并利用因果语言模型的方式来设置输入和标签：

- <strong class="list-label">input_ids</strong>：取上述序列的所有token，但去掉最后一个token（序列从开头到倒数第二个token）。这些将作为模型的输入序列，使模型在看到该token序列的前缀下去预测下一个token。

- <strong class="list-label">labels</strong>：取同一序列的所有token，但去掉第一个token（序列从第二个token到最后一个token）。这样labels\[i\]实际上就是原始序列中input_ids\[i\]之后应该预测的下一个token。模型在训练时会尝试用第 i 个输入预测出第 i+1 个token。

这个“一位移”的设置确保模型的<strong class="key-term">预测目标</strong>对齐为“给定前面的文本，预测下一个词”，符合因果语言模型训练的要求。例如，如果序列是“A B C”，那么input_ids = \[A, B\]，而labels = \[B, C\]，模型接受“A”预测出B，接受“A B”预测出C。

此外，我们还需要一个<strong class="key-term">响应的掩码（response mask）</strong>。因为prompt部分是给定的上下文，我们不希望在计算损失时将prompt部分也算进去。通过构造一个mask，将prompt部分给mask掉而<strong class="key-term">只计算回答部分的预测损失</strong>。这样可以确保优化的梯度仅来自模型生成的内容部分，而不会影响对提示的处理。这个mask会与labels对齐，mask掉序列中属于prompt提示的部分，只保留response部分参与策略梯度计算。

<strong class="critical-term">计算对数概率</strong>

有了input_ids和labels后，我们通过模型的前向传播计算<strong class="key-term">对数概率</strong>，为后续的策略梯度公式做准备。具体步骤如下：

1.  <strong class="list-label">获取原始Logits：</strong>将input_ids喂入当前策略模型，得到每个位置上的logits输出。Logits是模型未归一化的原始得分，一般维度是 (B, L, V)，其中 B 是批量大小，L 是序列长度，V 是词表大小。每个logit $l_{t,v}$表示模型在位置t上输出词汇表中第v个词的原始分值。

2.  <strong class="list-label">对数归一化（Log Softmax）：</strong>对logits沿着词表维度应用log_softmax函数，将其转换为对数概率。也就是先通过softmax将每个位置的logit变成概率分布，再取对数得到 log p。经过这一步，我们得到张量 log_probs，其形状仍是 (B, L, V)，但每个位置 t 上的向量变成了该位置输出每个词的对数概率 $\log p_{t,v}$。

3.  <strong class="list-label">标签对齐：</strong>由于自回归语言模型在位置 t 的输出是用于预测序列中第 t+1 个token（如前述的labels设置），我们需要将计算得到的对数概率与正确的目标标签对应起来。具体地，<strong class="key-term">对于序列长度为 L 的输入，位置 0 的logits用于预测位置1的标签，位置1的logits用于预测位置2的标签，... ，位置 L-2 的logits用于预测位置 L-1 的标签</strong>。因此我们通常会丢弃序列最后一位的logits（没有下一个token可预测），或者在代码实现上，通过对labels移位来对齐。这一步确保log_probs中的第 t 行对应labels中的第 t 项的对数概率。

4.  <strong class="list-label">提取目标token的对数概率：</strong>现在，对每个序列位置，我们从log_probs中取出模型对正确标签的对数概率值。这可以使用诸如torch.gather的操作：根据labels提供的索引，从每个位置的概率分布向量中选出对应标签token的概率。最终得到一个形状为 (B, L-1) 的矩阵，每个元素是模型在该位置输出正确下一个词的log概率 $\log \pi_\theta(y_t|x_{<t})$。

经过上述步骤，我们就拿到了模型对每个prompt对应回答的概率信息。其中关键是：<strong class="key-term">新策略 $\pi_\theta$ 输出当前回答序列的概率</strong>，以及<strong class="key-term">旧策略 $\pi_{\text{old}}$ 输出同一回答的概率</strong>。由于在实际实现中，我们通常会保留旧策略模型以便同时计算 $\pi_{\text{old}}(o_i|q)$，从而后续构造比率。对于训练第一轮来说，旧策略和新策略参数是相同的（模型还未更新），所以两者概率相同。

<strong class="critical-term">策略损失计算（PPO剪辑目标）</strong>

最后，我们使用前面得到的概率值和优势 A 来计算策略梯度的<strong class="key-term">损失函数</strong>，并通过反向传播更新模型参数。GRPO采用了与PPO类似的<strong class="key-term">剪辑策略梯度目标</strong>。具体来说，对于每个样本（prompt）下的第 i 个回答，我们计算<strong class="key-term">概率比率</strong> $\rho_i$ 和剪辑后的目标：

- <strong class="list-label">概率比率</strong>：定义 $\displaystyle \rho_i = \frac{\pi_{\theta}(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}$，也就是新策略模型在prompt q 上生成回答 $o_i$ 的概率与旧策略模型生成该回答概率的比值。这个比率衡量新策略相对于旧策略在该回答上的概率放大或缩小的程度。如果 $\rho_i > 1$，说明新模型比旧模型更倾向于产生这个回答；反之若 $\rho_i < 1$ 则倾向降低其概率。

- <strong class="list-label">剪辑策略目标</strong>：PPO引入剪辑的思想来避免单次更新幅度过大。我们计算一个未剪辑的目标项和一个剪辑后的目标项：$L_{PG,i}=\min(ρ_i A^i,clip(ρ_i,1−ϵ,1+ϵ)A^i)$.

- <strong class="list-label">损失与梯度：</strong>我们希望最大化上述目标期望，也就是让策略更偏向优势为正的动作，远离优势为负的动作。通常将上述目标取负号变为损失函数供优化器最小化。也即，策略梯度的损失定义为：$\mathcal{L}_{\text{policy}} \;=\; - \mathbb{E}_{i}\Big[\,L_{\text{PG},i}\,\Big]\,$

其中对每个样本的各个候选i取平均（或者求和）再取负号。在实现中，这相当于：<strong class="key-term">如果优势 $A_i$ 为正，模型将受到负梯度推动去增大对应回答的概率（因为此时$\rho A$取负后是负的梯度，梯度下降会提高该概率）；如果优势为负，模型则被推动降低该回答概率。</strong>这与策略梯度(REINFORCE)的直观作用是一致的，只不过借助了旧策略来衡量概率比率并进行了剪裁调制。

值得注意的是，在训练开始的第一次迭代，我们通常设置旧策略$\pi_{\text{old}}$与新策略$\pi_{\theta}$权重相同（因为还没更新模型）。此时对所有i都有$\rho_i \approx 1$，剪辑也不起作用，于是损失近似为 $-A_i$ 的形式。也就是说，第一步基本就是根据优势的符号来调整策略：正优势让策略概率增大，负优势让其减小。随着若干步更新后，我们会定期更新旧策略（例如每个epoch或一定步数后将当前策略保存为新的参考策略），继续上述过程，直到收敛。

#### 13.1.4.5 总结

总结来说，GRPO训练过程包括：对每个prompt用旧模型采样多条回答，计算每条回答的奖励分数，然后组内归一化得到优势，再将(prompt + 回答)序列输入模型计算对数概率，最后依据剪辑的策略梯度公式来调整模型参数。通过组相对优势的引入，GRPO避免了训练价值函数，以更低的开销实现了与PPO类似的效果。

## 13.2 GRPO 训练循环与实验

### 13.2.1 GRPO训练中的易混淆的参数

- <strong class="list-label">采样层（rollout）：</strong>先拿问题去生成回答数据

- <strong class="list-label">训练层（epoch/batch）：</strong>对这批回答做几轮训练

- <strong class="list-label">计算层（microbatch）：</strong>为了省显存做梯度累积

| 变量 | 含义 | 所在层级 |
|:---|:---|:---|
| n_grpo_steps | 外层训练总步数（重复“采样+训练”多少次） | 最外层 |
| group_size | 每个 prompt 采样多少个回答 | 采样层 |
| rollout_batch_size | 每个 GRPO step 一共采样多少条回答（responses） | 采样层 |
| n_prompts_per_rollout_batch | 每步采样多少个 prompt，公式：rollout_batch_size // group_size | 采样层 |
| epochs_per_rollout_batch | 对同一批 rollout 数据重复训练几遍 | 训练层 |
| train_batch_size | 每次优化器更新前，处理的“大 batch”大小 | 训练层 |
| gradient_accumulation_steps | 一个大 batch 内分成多少个 microbatch 来累积梯度 | 计算层 |
| micro_train_batch_size | 每个 microbatch 大小，公式：train_batch_size // gradient_accumulation_steps | 计算层 |

举个例子，假设：

- n_grpo_steps = 3

- rollout_batch_size = 24

- group_size = 4

- epochs_per_rollout_batch = 2

- train_batch_size = 12

- gradient_accumulation_steps = 3

那么：

1.  每个 GRPO step 先<strong class="key-term">采样</strong>

    - n_prompts_per_rollout_batch = 24/4 = 6

    - <strong class="list-label">即：</strong>抽 6 个题目，每题生成 4 个回答，共 24 条 rollout

2.  然后<strong class="key-term">训练</strong>（同一批 24 条数据）

    - epochs_per_rollout_batch=2，所以这 24 条会训练 2 遍

    - 每遍按 train_batch_size=12 切成 2 个大 batch

3.  每个大 batch 再<strong class="key-term">做梯度累积</strong>

    - micro_train_batch_size = 12/3 = 4

    - 每个大 batch 分成 3 个 microbatch（每个 4 条）

4.  所以每个 GRPO step 的<strong class="key-term">优化器更新数</strong>

    - <strong class="list-label">每个 epoch:</strong> 24/12 = 2 次更新

    - <strong class="list-label">每个 step:</strong> 2(epoch) \* 2 = 4 次更新

5.  全训练总计

    - <strong class="list-label">总 rollout 数：</strong>3 \* 24 = 72

    - <strong class="list-label">总 optimizer 更新：</strong>3 \* 4 = 12

### 13.2.2 完整训练思路

完整 GRPO train loop 按下面链路实现：

1.  每个训练 step 先从 train.jsonl 采样 n_prompts_per_rollout_batch = rollout_batch_size / group_size 个题目。

2.  用 vLLM 对每题生成 group_size 个 rollout（stop 到 &lt;/answer&gt;）。

3.  计算每条 rollout 的 raw reward，再按组做<strong class="key-term">中心化/标准差归一化</strong>得到 advantage。

4.  把 prompt + response 转成训练张量（input_ids/labels/response_mask）。

5.  用 policy 算 response token 的 log-prob。

6.  按 gradient_accumulation_steps 做微批训练，支持：

7.  做梯度裁剪和 optimizer step。

8.  每 val_every_steps 做一次验证奖励评估，记录曲线。

### 13.2.3 实验中真实遇到的问题

1.  我是在V100上跑的，要更改代码使得其兼容vllm架构；

2.  Qwen-1.5B-Math模型还是能力不够，如果要正经按照要求严格限制输出规范和答案的话，<strong class="key-term">奖励非常稀疏</strong>，约等于0，即每个批次一个正确答案都得不到，模型失去了训练的意义。因此，我放宽了约束，侧重答案正确，哪怕格式不规范也不要紧。

3.  我实验最终输出的结果依旧不咋地，即使放宽了约束，也只是从一个问题都回答不上来变成了回答上来一两个。。

## 13.3 本篇小结与方法衔接

本篇从零样本基线出发，先建立 SFT 的数据与损失计算组件，再通过专家迭代把生成、筛选和训练连成循环。策略梯度、优势估计与 PPO 为理解 GRPO 的更新目标提供基础；GRPO 的组内评分与优势计算则与后面的训练循环对应。

<strong class="note-label">复习线索：</strong>比较这些方法时，依次检查训练样本从哪里来、评分如何参与训练、哪些概率需要保存，以及一次采样后进行了多少次参数更新。实验结果应与开篇建立的评估口径一起阅读；评分规则发生变化时，也要记录变化。

以下补充节保留经典拒绝采样、TRPO 与 GAE 的详细证明和拓展解释，首次阅读可按需查阅。第六篇则转向通用指令与偏好数据，先集中介绍指令微调、奖励模型、RLHF 与 DPO 理论，再独立展开基线评估、SFT 与 DPO 实验。

## 13.4 补充背景：经典拒绝采样

<span id="supp-rejection"></span> 本节保留经典蒙特卡洛拒绝采样的定义与证明，供理解采样背景时查阅。专家迭代中的按评分筛选回答是一项训练数据构造操作，不能仅凭“拒绝”一词便将它与这里满足包络条件的概率采样算法等同。

<strong class="note-label">延伸阅读：</strong>[见本篇末资料 \[1\]。](/part-5/chapter-13#read-12-1)

拒绝采样(Rejection Sampling)是蒙特卡洛采样法中的一个重要方法,它的核心目的是<strong class="key-term">在无法直接对目标分布进行采样时,通过对一个易于采样的辅助分布进行采样,并依据特定概率“拒绝”掉一部分样本,从而间接获得服从目标分布的样本</strong>。

### 13.4.1 核心定义与符号体系

- <strong class="list-label">x</strong>:随机变量,可以是标量也可以是高维向量,$x \in \mathbb{R}^d$。

- <strong class="list-label">p(x)</strong>:<strong class="key-term">目标分布</strong>的概率密度函数(PDF)。

  - 在实际工程中,p(x) 通常难以直接归一化,我们往往只能计算出近似归一化的 $\tilde{p}(x)$,即 $p(x) = \frac{1}{Z}\tilde{p}(x)$,其中 Z 是未知的归一化常数。

- <strong class="list-label">q(x)</strong>:<strong class="key-term">提议分布</strong>。这是一个易于采样的分布(如高斯分布、均匀分布)。

- <strong class="list-label">k</strong>:<strong class="key-term">比例常数</strong>。它是一个标量,且必须满足<strong class="key-term">包络条件</strong>:对于定义域内所有的 x,都有$k \cdot q(x) \geq \tilde{p}(x)$。

- <strong class="list-label">u</strong>:<strong class="key-term">辅助随机变量</strong>,服从均匀分布 $u \sim \text{Uniform}[0, 1]$。

### 13.4.2 直觉上理解

通过下面的概率分布图可以很好理解拒绝采样的想法。

- <strong class="critical-term">红色曲线</strong>:表示归一化目标分布的概率密度函数 $\tilde{p}(z)$,它比较复杂,难以直接采样得到。

- <strong class="list-label">蓝色曲线</strong>:表示乘上比例系数的提议分布函数$k \cdot q(x)$,它可以被我们表示、采样出来,图中以高斯分布为例。

首先可以看到,蓝色曲线是完全包络红色曲线的,即$k \cdot q(x) \geq \tilde{p}(x)$,这表示<strong class="key-term">目标分布的概率密度函数是完全可以在我们的提议分布这一小范围内得到</strong>。

接下来,我们采样的时候,比如选择$z_0$, 那么对应的$k \cdot q(x)$结果就是$k \cdot q(z_0)$;我们知道在这一条纵轴上 $\tilde{p}(z_0)$的结果一定是在$[0,k \cdot q(z_0)]$之内的,因为目标分布被包络了。所以拒绝采样的核心想法就是在$[0,k \cdot q(z_0)]$之上<strong class="key-term">均匀分布</strong>,得到的结果$u_0$如果小于 $\tilde{p}(z_0)$,证明在红色曲线目标分布面积之内,采样有效,否则表示在红蓝之间的灰色面积里,采样无效,拒绝。

<figure data-latex-placement="H">
<img src="/images/bc8c5158a5.png" style="width:80.0%" alt="拒绝采样示意图" />
<figcaption>拒绝采样示意图</figcaption>
</figure>

### 13.4.3 严谨数学证明

我们的目标是证明:<strong class="key-term">通过拒绝采样算法生成的样本,其真实的概率分布确实是 p(x)</strong>。

#### 13.4.3.1 算法流程

1.  从<strong class="key-term">提议分布</strong>中采样一个候选样本:$X \sim q(x)$。

2.  从<strong class="key-term">均匀分布</strong>中采样一个辅助变量:$U \sim \text{Uniform}[0, 1]$。

3.  <strong class="list-label">接受判据:</strong>如果 $U \leq \frac{\tilde{p}(X)}{k \cdot q(X)}$,则接受 X；否则拒绝(丢弃并重试)。

#### 13.4.3.2 证明过程

我们需要计算在“接受”(记为事件 A)的条件下,样本 X 的<strong class="key-term">边缘概率密度</strong>。根据贝叶斯定理:

$p(x | A) = \frac{p(A | x) \cdot p_{candidate}(x)}{p(A)}$

这里有三个关键项需要展开:

1.  <strong class="list-label">步骤 1:</strong>分析 $p_{candidate}(x)$ 由于候选样本 X 是直接从 q(x) 中采样的,所以其先验概率密度就是:$p_{candidate}(x) = q(x)$

2.  <strong class="list-label">步骤 2:</strong>分析条件接受概率 p(A \| x) 给定一个具体的 x,我们接受它的条件是 $U \leq \frac{\tilde{p}(x)}{k \cdot q(x)}$。因为 U 是 \[0,1\] 上的均匀分布,所以该事件发生的概率就是区间的长度:

    $p(A | x) = \mathbb{P}\left(U \leq \frac{\tilde{p}(x)}{k \cdot q(x)}\right) = \frac{\tilde{p}(x)}{k \cdot q(x)}$

3.  <strong class="list-label">步骤 3:</strong>计算总接受率 p(A) 这是所有可能得 x 被接受的概率积分:

    $\begin{aligned} p(A) &= \int p(A | x) \cdot q(x) \, dx \\ &= \int \left( \frac{\tilde{p}(x)}{k \cdot q(x)} \right) \cdot q(x) \, dx \\ &= \int \frac{\tilde{p}(x)}{k} \, dx \\ &= \frac{1}{k} \int \tilde{p}(x) \, dx \end{aligned}$

    由于 $p(x) = \frac{\tilde{p}(x)}{Z}$,故 $\tilde{p}(x) = Z \cdot p(x)$。

    $p(A) = \frac{1}{k} \int Z \cdot p(x) \, dx = \frac{Z}{k} \underbrace{\int p(x) \, dx}_{1} = \frac{Z}{k}$

4.  <strong class="list-label">步骤 4:</strong>整合最终结果 将上述三项代入贝叶斯公式:

    $\begin{aligned} p(x | A) &= \frac{\left( \frac{\tilde{p}(x)}{k \cdot q(x)} \right) \cdot q(x)}{\frac{Z}{k}} \\ &= \frac{\frac{\tilde{p}(x)}{k}}{\frac{Z}{k}} \\ &= \frac{\tilde{p}(x)}{Z} \\ &= p(x) \end{aligned}$

## 13.5 补充推导：TRPO 的局部近似与自然梯度

<span id="supp-trpo"></span> 本节展开正文中信任域优化的近似求解过程，可在读完 PPO 与 GRPO 后回看。

### 13.5.1 证明一阶泰勒展开

1.  <strong class="list-label">新策略比旧策略额外的奖励回报</strong>:

    $J(\theta) - J(\theta_{\text{old}}) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_{\text{old}}}(s_t, a_t) \right]$

    把奖励展开为状态和策略 $J(\theta) - J(\theta_{\text{old}}) = \sum_{s} \rho_{\pi_{\theta}}(s) \sum_{a} \pi_{\theta}(a \mid s) A^{\pi_{\text{old}}}(s, a)$

2.  <strong class="list-label">定义修改</strong>:

    TRPO 做了一个近似:<strong class="key-term">假设在局部更新时,状态分布的变化忽略不计</strong>,即 $\rho_{\pi_\theta}(s) \approx \rho_{\pi_{old}}(s)$

    于是我们定义替代目标函数$L(\theta)$:

    $L(\theta) = J(\theta_{old}) + \sum_{s} \rho_{\pi_{old}}(s) \sum_{a} \pi_\theta(a|s) A^{\pi_{old}}(s, a)$

    利用<strong class="key-term">重要性采样</strong>

    $\sum_{a} \pi_\theta(a|s) A(s, a) = \sum_{a} {\frac{\pi_{old}(a|s)}{\pi_{old}(a|s)}} \cdot \pi_\theta(a|s) A(s, a)$$= \sum_{a} {\pi_{old}(a|s)} \cdot \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s, a) \right]$$= \mathbb{E}_{a \sim \pi_{old}} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s, a) \right]$

    可得 $L(\theta) = J(\theta_{\text{old}}) + \mathbb{E}_{s \sim \rho_{\text{old}}, a \sim \pi_{\text{old}}} \left[ \frac{\pi_{\theta}(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\text{old}}}(s, a) \right]$

3.  <strong class="list-label">一阶泰勒展开</strong>:

    现在,我们要对 $L(\theta)$ 在 $\theta_{old}$ 处进行一阶泰勒展开。 $L(\theta) \approx L(\theta_{\text{old}}) + \nabla_{\theta} L(\theta) \Big|_{\theta= \theta_{\text{old}}}(\theta - \theta_{\text{old}})$

    - <strong class="list-label">计算常数项 $L(\theta_{old})$,</strong>当 $\theta = \theta_{old}$ 时,概率比率为 1。 $L(\theta_{old}) = J(\theta_{old}) + \mathbb{E}_{s,a \sim \pi_{old}} [ 1 \cdot A^{\pi_{old}}(s, a) ]$

      根据优势函数的定义,其在自身策略下的期望为 0(即 $\mathbb{E}[Q-V] = V-V=0$)。 $\mathbb{E}_{a \sim \pi_{old}} [A^{\pi_{old}}(s, a)] = 0$

      所以:$\boxed{L(\theta_{old}) = J(\theta_{old})}$

    - <strong class="list-label">计算梯度项 $\nabla_\theta L(\theta) \big|_{\theta=\theta_{old}}$</strong>: 这正是我们在上一个问题中证明过的结论。 $\nabla_\theta L(\theta) = \mathbb{E} \left[ \frac{\nabla_\theta \pi_\theta}{\pi_{\theta_{old}}} A^{\pi_{old}} \right]$

      在 $\theta = \theta_{old}$ 处: $\nabla_\theta L(\theta) \big|_{\theta=\theta_{old}} = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) |_{\theta_{old}} \cdot A^{\pi_{old}} \right]$ (计算方式就是先用对数导数技巧换成导数乘$\pi$,然后消掉分母)

      这正是 <strong class="key-term">Vanilla PG 的梯度</strong>,我们记为 $\textbf{g}$。

4.  <strong class="list-label">最终展开结果</strong>:

    将上述两项代回泰勒公式:

    $L(\theta) \approx J(\theta_{old}) + g^T (\theta - \theta_{old})$

### 13.5.2 二阶泰勒展开与费雪信息矩阵

<figure>
<img src="/images/d06ffd4dbd.png" style="width:100.0%" alt="KL散度的二阶泰勒展开" />
<figcaption>KL散度的二阶泰勒展开</figcaption>
</figure>

<strong class="list-label">符号定义</strong>

- <strong class="list-label">$\theta \in \mathbb{R}^n$</strong>: 模型的参数向量。

- <strong class="list-label">x</strong>: 随机变量。

- <strong class="list-label">$p(x|\theta)$ 或 $\pi_\theta(a|s)$</strong>: 由参数 $θ$ 决定的概率分布(即策略)。

- <strong class="list-label">$\mathcal{L}(\theta) = \log p(x|\theta)$</strong>: 对数似然函数。

- <strong class="list-label">$g = \nabla_\theta \mathcal{L}(\theta)$</strong>: 分数函数,即对数似然的梯度,实际上就是在欧几里何空间下的梯度。

- <strong class="list-label">$F(\theta)$</strong>: 费雪信息矩阵。

<strong class="note-label">直观理解</strong>

用来<strong class="key-term">衡量观测数据中包含了多少关于模型参数的信息</strong>

假设你在估计一个参数 θ:

- <strong class="list-label">如果参数稍微变一点,模型的概率分布就变化很大</strong>: 数据对参数很敏感 - \> 费雪信息大

- <strong class="list-label">如果参数变了,模型几乎没变化</strong>: 数据难以区分不同参数 - \> 费雪信息小

- <strong class="list-label">所以:</strong>费雪信息越大,参数越容易被准确估计

<strong class="list-label">费雪信息矩阵的定义</strong>

$\mathbf{F}(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log p(X \mid \theta) \, \nabla_{\theta} \log p(X \mid \theta)^\top \right]$

其中:

- <strong class="list-label">$\nabla_\theta \log p(X \mid \theta)$</strong>: 对数似然的梯度(也被称为得分函数)

- <strong class="list-label">期望是对 $X \sim p(x|\theta)$ 取的</strong>

<strong class="list-label">等价形式(常用)</strong>

在满足一定正则条件下:$\mathbf{F}(\theta) = -\mathbb{E} \left[ \nabla^2_{\theta} \log p(X \mid \theta) \right]$

也就是说:

<strong class="critical-term">费雪信息 = 对数似然 Hessian 的负期望</strong>

具体推导过程:

<figure>
<img src="/images/ed38649471.png" style="width:100.0%" alt="费雪信息矩阵与海斯矩阵的关系" />
<figcaption>费雪信息矩阵与海斯矩阵的关系</figcaption>
</figure>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么费雪信息矩阵要这样定义？</p>

对数似然梯度反映了一种<strong class="key-term">瞬时敏感度</strong>,$\nabla_\theta \log p(x|\theta)$ 表示:<strong class="critical-term">在观测到样本 x 的情况下,参数$\theta$的微小变化会让概率变化多快</strong>。结果大 → 这个样本对参数很敏感,结果小 → 这个样本几乎没提供区分能力

</div>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么是外积形式(而不是 Hessian)？</p>

一方面在正则条件下,两者<strong class="key-term">严格相等</strong>:

$\mathbf{F}(\theta)=-\mathbb{E}[\nabla^2_\theta \log p(X|\theta)]$

但外积形式有几个深层优势:

1\. <strong class="key-term">必然正半定(信息不能为负)</strong> 2. 与 KL / 信息几何自然一致

</div>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么对约束的KL散度是二阶展开,对目标函数就只是一阶泰勒展开？</p>

1\. <strong class="key-term">KL 在最优点的一阶项是 0(这是本质差异)</strong>

KL 约束项:

$\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta)=\mathbb{E}_s\big[ D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}\|\pi_\theta) \big]$

在 $\theta=\theta_{\text{old}}$ 处:

\- KL = 0

\- 一阶导数 = 0(说明这是<strong class="key-term">局部最小值</strong>)

因此:

$\bar D_{\mathrm{KL}}(\theta_{\text{old}},\theta)=\frac12\Delta\theta^\top H_{\mathrm{KL}}\Delta\theta+o(\|\Delta\theta\|^2)$

<strong class="key-term">如果你只做一阶近似,KL ≈ 0,约束直接消失,没有意义</strong>。

所以:

\- <strong class="key-term">目标函数</strong>:一阶就有非零信息(梯度)

\- <strong class="key-term">KL 约束</strong>:一阶项恒为 0,必须保留二阶

2\. <strong class="key-term">为什么不对目标也做二阶泰勒展开？</strong>

数学上虽然能做到,但是:

二阶泰勒展开后 $H_L$ 是目标函数的 Hessian 矩阵:

$H_L = \nabla^2_\theta L(\theta) = \mathbb{E} \left[ \nabla^2_\theta \log \pi_\theta(a|s) \cdot A + \nabla_\theta \log \pi_\theta \nabla_\theta \log \pi_\theta^T \cdot A \right]$

\- 目标 Hessian 矩阵: - <strong class="key-term">不一定正定</strong>,这受到优势函数A的影响。我们希望<strong class="key-term">最大化 $L(\theta)$</strong>,如果二阶项是负的,就处于局部最大值,优化器会认为"离得越远越好",这在数值上非常不稳定。 - KL散度的Hessian矩阵只和策略函数$\pi$有关,而$H_L$还和优势函数A有关,每个样本的A都各不相同,<strong class="key-term">方差太大导致噪声极大</strong>。

<strong class="key-term">所以 TRPO 的设计选择是:</strong>

<strong class="key-term">用 KL 的二阶结构控制“走多远”(<strong class="critical-term">步长</strong>),用目标的一阶信息决定“往哪走”(<strong class="critical-term">梯度方向</strong>)</strong>。

</div>

### 13.5.3 二次规划近似形式

经过两次近似后,TRPO 可以近似成一个<strong class="key-term">标准二次规划</strong>(忽略常数项):

$\max_{\Delta\theta}\quad g^\top \Delta\theta$

$\text{s.t.}\quad \frac12 \Delta\theta^\top F \Delta\theta \le \delta$

<strong class="list-label">解这个二次规划:</strong>得到自然梯度方向

这是一个经典结果:在线性目标、椭球约束下,最优解方向是 $F^{-1}g$。

推导:

构造<strong class="key-term">拉格朗日函数</strong>:

$\mathcal{L}(\Delta\theta,\lambda)=g^\top\Delta\theta-\lambda\left(\frac12\Delta\theta^\top F\Delta\theta-\delta\right)$

对 $\Delta\theta$ 求导并置零:

$\nabla_{\Delta\theta}\mathcal{L}=g-\lambda F\Delta\theta=0$ $\Rightarrow \Delta\theta=\frac{1}{\lambda}F^{-1}g$

再用约束饱和确定缩放因子:

$\frac12 \Delta\theta^\top F\Delta\theta =\frac12 \frac{1}{\lambda^2} g^\top F^{-1} g = \delta \Rightarrow \lambda = \sqrt{\frac{g^\top F^{-1} g}{2\delta}}$

所以 $\Delta\theta^\star =\sqrt{\frac{2\delta}{g^\top F^{-1} g}}\;F^{-1}g$

<div class="custom-block tip">

<p class="custom-block-title">理解与提示</p>

实际 TRPO 算法在计算出这个更新方向后,会进行<strong class="key-term">Line Search(线性搜索)</strong>。即,先尝试走完这步,检查:KL 散度是否真的≤δ？目标函数L(θ)是否真的提升了？如果不满足,就按比例缩小步长(如×0.5,×0.25... )直到满 足为止。这是“Trust Region”保证单调不下降的关键工程实现。

</div>

<strong class="key-term">方向 $F^{-1}g$ 就是自然梯度方向。</strong>这导出了<strong class="key-term">自然梯度</strong>的更新方向: $\theta_{new} = \theta_{old} + \beta F^{-1} g$

其中 $F^{-1}$ 校正了欧几里得空间的梯度方向,使其符合黎曼几何(概率流形)的曲率。

### 13.5.4 黎曼几何空间与自然梯度的简要说明

在普通梯度下降中,我们更新参数的规则是:$\theta_{new} = \theta_{old} + \alpha \nabla_\theta J(\theta)$

这隐含了一个假设:我们在参数空间 $\Theta$ 中移动一个固定的欧几里得距离$||\Delta \theta||^2$,会导致策略表现发生可预期的变化。但实际上,参数空间的小微扰 $\Delta \theta$ 可能导致输出概率分布 $\pi_\theta$ 发生<strong class="key-term">剧烈变化</strong>,也可能几乎不变。参数空间是平坦的,但策略空间是<strong class="key-term">弯曲</strong>的。 所以,我们之所以引入黎曼几何是因为<strong class="key-term">距离的定义和位置相关</strong>。在概率单纯形上,衡量两个策略 $\pi_\theta$ 和 $\pi_{\theta+\delta\theta}$ 差异的最自然度量不是 $||\delta\theta||^2$,而是二阶 KL 散度,也就是我们之前推导的费雪信息矩阵$F(\theta)$。

$F(\theta)$ 扮演了<strong class="key-term">度量张量</strong> 的角色。它定义了流形上的<strong class="key-term">局部曲率</strong>。$F^{-1}$ 就是对欧几里何空间的梯度g进行了“<strong class="key-term">矫正</strong>”,使其适应流形的曲率。 $F^{-1}\nabla_\theta J$ 或 $F^{-1}g$就是<strong class="key-term">黎曼几何空间下的梯度</strong>,也被称为自然梯度。

## 13.6 补充理解：GAE 的信号处理视角

<span id="supp-gae"></span>

### 13.6.1 信号处理视角的“滤波器”

假设我们有一个真实的累积回报信号 $G_t$。我们可以将其分解为两部分: $G_t = \underbrace{V_\pi(s_t)}_{\text{均值/趋势}} + \underbrace{A_\pi(s_t, a_t)}_{\text{波动/细节}} + \underbrace{\epsilon}_{\text{噪声}}$

1.  <strong class="list-label">低频信息 ≈ 价值函数 V(s) 的偏差</strong>

    - <strong class="list-label">定义</strong>:这是信号中的“直流分量”或“缓慢变化的趋势”。

    - <strong class="list-label">RL 对应</strong>:由 Critic 网络 $V_\phi(s)$ 提供的预测值。

    - <strong class="list-label">特性</strong>:因为它是一个神经网络的输出,它是对无数条历史轨迹的平均。因此它是<strong class="key-term">平滑的</strong>、<strong class="key-term">稳定的</strong>,但在训练初期通常是不准确的(存在系统性偏差)。

2.  <strong class="list-label">高频信息≈ 蒙特卡洛采样的方差</strong>

    - <strong class="list-label">定义</strong>:这是信号中快速跳变、剧烈震荡的部分。

    - <strong class="list-label">RL 对应</strong>:单次采样轨迹中具体的奖励 $r_t$ 和状态跳转。

    - <strong class="list-label">特性</strong>:如果你把同一个策略运行 100 次,每一次的轨迹(高频细节)都不同。这不仅包含你的动作带来的真实反馈(有效高频),也包含环境掷骰子的随机性(无效高频噪声)。

回顾 GAE 的定义: $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$

这就好比我们在时间轴上对 TD Error 序列 δ 做了一个<strong class="key-term">卷积</strong>。 $\hat{A} = \delta * \text{Kernel}_{\lambda}$

其中卷积核是指数衰减函数$f(k) = (\gamma \lambda)^kf$

### 13.6.2 低频与高频

让我们看看 λ 如何控制这个“滤波器”的通带:

1.  <strong class="list-label">当 λ→0 (低通滤波器 / 截止频率极低)</strong>

    - 卷积核迅速衰减,只保留第一项 $\delta_t$。

    - <strong class="list-label">公式退化为:</strong>$r_t + \gamma V(s_{t+1}) - V(s_t)$。

    - <strong class="list-label">频域解释:</strong>我们切断了未来的所有高频波动。我们只看这一步的 $r_t$,剩下的全部用 V(s) 这个“低频均值”来代替。

    - <strong class="list-label">结果:</strong>滤除了蒙特卡洛带来的高频噪声(方差极低),但引入了 V(s) 自身的低频系统性错误(偏差大)。<strong class="key-term">我们过于信任“趋势”。</strong>

2.  <strong class="list-label">当 λ→1 (全通滤波器 )</strong>

    - 卷积核衰减很慢,累加了长远的未来。

    - <strong class="list-label">公式退化为:</strong>$R_t - V(s_t)$。

    - <strong class="list-label">频域解释:</strong>我们允许所有的频率通过。哪怕 100 步之后发生了一个随机事件导致奖励剧变,这个震荡(高频信号)也会完整地传导回 t 时刻。

    - <strong class="list-label">结果:</strong>消除了 Critic 的系统性偏差(因为最终用的是真实回报),但引入了巨大的高频环境噪声(方差极大)。<strong class="key-term">我们过于信任“细节”。</strong>

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment5-alignment)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下集中列出第五篇各章的教程、文档与阅读说明；编号与各章中的阅读入口对应。

### 1 · 拒绝采样

<span id="read-12-1"></span>

<https://blog.csdn.net/jteng/article/details/54344766>

### 2 · 重要性采样

<span id="read-12-2"></span>

<https://zhuanlan.zhihu.com/p/41217212>

<https://zhuanlan.zhihu.com/p/342936969>

### 3 · 策略梯度与 TRPO 补充推导

<span id="read-12-3"></span>

<https://www.cnblogs.com/xingzheai/p/16565686.html>
