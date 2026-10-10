---
outline: [2, 3]
---

# 第 11 章 · 数学任务、SFT 与专家迭代

<span id="guide-ch-12"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[概率与梯度](/part-1/chapter-3#guide-ch-3)、[采样](/part-1/chapter-5#guide-ch-5)及[训练流程](/part-1/chapter-4#guide-ch-4)；先建立数学任务基线，再逐步连接训练组件与策略优化。

<strong>准备工作：</strong>准备 Qwen2.5-Math-1.5B、数学任务 JSONL、评分函数和推理环境；数据版本说明见[本篇资源入口](/part-5/resources)。

<strong>本章任务：</strong>建立零样本基线，实现 SFT 辅助组件，再把生成、评分、筛选与训练连接成专家迭代流程。

</div>

## 11.1 后训练总览与本篇任务

### 11.1.1 后训练与预训练的区别

<strong class="key-term">预训练(Pre-training)</strong>指模型构建初期在海量无标注数据上训练,以学习通用特征和语言知识。预训练通常使用上百亿甚至数万亿规模的token,通过<strong class="key-term">自监督任务(如下一个词预测)</strong>让模型掌握语言的基本规律。这一阶段产生的基础模型具备广泛的语言能力,但不一定懂得遵循人类指令。

<strong class="key-term">后训练(Post-training)</strong>则发生在预训练之后、模型部署之前,用特定任务或目标数据对模型进行额外训练。后训练包含<strong class="key-term">微调(Fine-tuning)和对齐(Alignment)</strong>过程,通常涉及<strong class="key-term">有监督微调</strong>和<strong class="key-term">基于人类反馈的强化学习(RLHF)</strong>等方法。与预训练侧重通用性不同,后训练聚焦在定制模型行为,让模型适应特定应用场景或对齐人类偏好。简单来说,预训练教会模型“能回答”,而后训练教会模型“如何更好地回答”。两者相比:

- <strong class="list-label">数据</strong>:预训练用海量通用语料,后训练用更窄的、高质量的指令或对话数据。

- <strong class="list-label">目标</strong>:预训练追求广泛知识获取,后训练追求在特定任务上优化模型性能。

- <strong class="list-label">阶段</strong>:预训练耗费最大算力资源,后训练相对开销小但关键,往往决定模型最终实用效果。

### 11.1.2 后训练的主要目标

后训练阶段的核心目标是<strong class="key-term">将通用大模型对齐到人类期望的行为</strong>,主要体现在以下几方面:

- <strong class="list-label">对齐人类偏好</strong>:通过后训练,让模型不仅会回答问题,还能回答得<strong class="key-term">有用、满足特定格式</strong>或<strong class="key-term">符合人类的某些其他偏好</strong>。例如,让模型回答的内容满足latex格式。

- <strong class="list-label">增强特定能力</strong>:<strong class="key-term">针对特定任务或领域提升模型性能</strong>。例如,通过指令微调让模型学会遵循复杂指令、多轮对话逻辑；或者通过专项数据(如代码、数学推理数据)微调来增强模型的编程、推理等专项能力。后训练可以让模型在特定领域的回答更准确、更专业。

- <strong class="list-label">提高交互体验</strong>:让模型回答更符合用户预期的风格和长度,减少无关或冗长内容。比如通过偏好对齐数据,让模型的回答更加简洁明了或风趣友好,提升人机交互体验。

- <strong class="list-label">安全性与可靠性</strong>:利用后训练减少<strong class="key-term">模型幻觉</strong>和有害输出。通过奖励惩罚机制,引导模型避免编造事实或使用不当言辞,使输出更加可信且安全。

总之,后训练的目的在于<strong class="key-term">在不改变模型底层知识的前提下,塑造模型的输出行为</strong>,让其更贴近人类期望的回答方式和质量要求。

### 11.1.3 本篇的学习路线

本篇围绕数学任务展开：先测量基座模型的零样本表现，再实现 SFT 所需的训练组件，连接生成、评分与筛选，完成专家迭代；随后理解策略梯度、优势估计和 PPO，最后进入 GRPO 的原理、实现与实验。

<strong class="note-label">阅读顺序：</strong>任务与基线 → SFT 与专家迭代 → 强化学习基础 → PPO → GRPO 实验。后两章分别是[策略优化基础：从策略梯度到 PPO](/part-5/chapter-12#guide-ch-12-policy)与[GRPO 原理与训练实验](/part-5/chapter-13#guide-ch-12-grpo)。经典拒绝采样、TRPO 长推导及 GAE 的信号处理视角集中在[本篇末的补充材料](/part-5/chapter-13#supp-rejection)中。偏好数据、奖励模型与 DPO 的详细讨论见[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)。

## 11.2 零样本数学能力评估

本节先检测基座模型的零样本数学能力，为后面的 SFT、专家迭代与 GRPO 实验建立比较基线。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 什么是zero shot 和 few shot?</p>

- <strong class="list-label">Zero-shot</strong>:<strong class="key-term">不给例子,直接让模型做</strong>

- <strong class="list-label">Few-shot</strong>:<strong class="key-term">给少量例子,让模型照着学再做</strong>

模型在没有任何提示的情况下直接完成任务就是零样本,比如:

Prompt:‘把下面这句话翻译成英文:今天天气很好‘

而先给模型几个示例,让它<strong class="key-term">学你的格式 / 风格 / 规则</strong>,再完成新任务,这就是少样本,比如:

Prompt:

“已知 中文:你好,英文:Hello; 中文:谢谢,英文:Thank you”

“请翻译:中文:今天天气很好,英文:”

</div>

作业要求就用vLLM框架实现批量推理

1.  <strong class="list-label">构建 Prompt 模版</strong>: 作业使用了deepseek的 “r1_zero” 模版。模版强制模型在输出答案前先生成 “&lt;think&gt;” 标签进行推理,最后在 “&lt;answer&gt;” 标签中输出最终结果,格式化为了方便后续去评估(不过模型不一定就完全按照prompt的要求来)。

2.  <strong class="list-label">初始化 vLLM 推理引擎</strong>: 作业使用了Qwen2.5-MATH-1.5B,直接在huggingface上下载即可。

3.  <strong class="list-label">批量生成</strong>: 读取 “validation.jsonl” 数据集(参考github上疑似斯坦福的某位同学的repo),设置采样参数(Temperature=1.0, stop_token=“&lt;/answer&gt;”),调用 “llm.generate()” 获取模型输出。

4.  <strong class="list-label">答案解析与评分</strong>: 用评分函数将模型生成的文本与标准答案(Ground Truth)进行比对。该函数会检查<strong class="key-term">格式是否正确</strong>以及<strong class="key-term">答案数值是否正确</strong>。

5.  <strong class="list-label">结果序列化与分析</strong>: 将输入问题、模型生成的完整文本、评分结果保存到磁盘,并统计各类错误(格式错误 vs 答案错误)的比例。(我的结果里几乎没有几个答案正确。。)

### 11.2.1 实现细节

首先给出的模板,比如说r1_zero_shot:

```text
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

我们要把question用实际的问题去替换,在讲义里也就是用validation.jsonl的问题。

然后利用vllm框架去实现加载模型,进而批量推理。

vllm加载模型是在huggingface上下载,我个人的一个加速下载的小trick是

1.  <strong class="list-label">直接拥抱镜像站</strong>:

2.  <strong class="list-label">确保 ‘pip install -U pysocks‘ python可以利用socket5来下载东西</strong>: ‘pip install -U pysocks‘

    记得要配置 export ALL_PROXY="socks5h://127.0.0.1:20170" 使得hf走socks5h协议,这样才可以利用基于python下的hf去下载东西,比如hf download Qwen/Qwen3-4B-Base

## 11.3 SFT：训练目标与基础组件

### 11.3.1 训练目标与数据形式

有监督微调(Supervised Fine-Tuning)也称“指令微调”。在这一步,我们使用人工标注的 <strong class="key-term">\[指令,回答\]</strong> 示例数据集,对预训练模型进行继续训练,以培养模型的基础对话和指令遵循能力。SFT的<strong class="key-term">训练目标仍是让模型预测下一个token</strong>,但训练数据是成对的指令和期望回答。通过模仿高质量人类演示,模型学会遵循指令进行响应。

SFT所需的数据包含一个用户指令和对应的高质量回答。例如:展示了一条中文指令微调数据:

```python
"input":"写一首关于猪的作文",
"output":"从前有一只猪很懒,睡到中午才起来..."
```

之后我们要选择一个预训练好的基座模型作为初始模型,比如Qwen、Deepseek等,由于完整微调大模型显存开销巨大,工程上常采用<strong class="key-term">LoRA</strong>等<strong class="key-term">参数高效微调 (PEFT)</strong>技术。LoRA通过在模型某些权重上添加低秩矩阵,只训练这些小规模参数,从而显著降低显存和计算需求。

常用工具链有:

- <strong class="list-label">Hugging Face Transformers</strong>:提供模型加载、Tokenizer和Trainer等便捷工具。可结合Datasets读取和处理指令数据集。

- <strong class="list-label">PEFT库</strong>:与Transformers集成,用于插入LoRA等适配器,封装了训练这些新增参数的逻辑。

- <strong class="list-label">DeepSpeed/FSDP</strong>:在全参数微调大模型时,可借助DeepSpeed ZeRO或PyTorch FSDP进行模型并行和内存优化,实现多GPU协同训练。DeepSpeed也支持混合精度、梯度检查点等加速和省显存技巧。

通过SFT阶段,模型将具备基本的指令遵循和对话能力,为后续的奖励对齐和强化学习奠定基础。

这一章主要关于SFT微调的一些实现

这一章的作业主要是实现SFT的一些组件。

### 11.3.2 作业1“tokenize_prompt_and_output”要求:

<strong class="critical-term">构建符合“下一个Token预测”任务格式的训练数据,并确保模型只学习如何生成“回答”,而不去学习如何生成“提问”。</strong> 

为此,我们需要将自然语言形式的 <strong class="key-term">\[提问, 回答\]</strong> 对,转换为模型可输入的 <strong class="key-term">Token ID 序列</strong>,并构造一个 <strong class="key-term">掩码(Mask)</strong>。这个掩码用于指示损失函数:<strong>在计算梯度时,忽略提问部分(Prompt)和填充部分(Padding),仅计算回答部分(Response)的损失</strong>。

#### 11.3.2.1 实现思路

1.  <strong class="list-label">独立分词</strong>:

    - 分别对 “prompt_str”(问题)和 “output_str”(答案)循环调用 Tokenizer 转换成 ids。

    - 获取两者各自的每个元素的长度,从而找到“答案”在拼接序列中的起始索引。方便后续mask操作。

2.  <strong class="list-label">序列拼接与填充</strong>:

    - 我们最终要得到的Tensor形状是(batchsize,padding(prompt_ids+output_ids)后的长度),batchsize就是输入输出列表的长度,也就是样本数,对于每一个样本我们都要先拼接在一起,再填充tokenzier.pad_token到最长的样本长度；

    - 填充的时候,我的做法是再新建一个(batchsize,padding(prompt_ids+output_ids)后的长度)的张量,再把之前求得的拼接好的列表重新输一遍。

    <div class="custom-block info">

    <p class="custom-block-title">延伸阅读</p>

    这里不能够分开各自填充prompt_ids,output_ids再拼接到一起,举个例子,我们想要的效果是\[input_1,input_2,input_3,output_1,output_2,pad_token,pad_token\],如果分开padding,再拼接,结果就是\[input_1,input_2,input_3,pad_token,output_1,output_2,pad_token\],顺序出了问题。

    </div>

3.  <strong class="list-label">构建mask</strong>:

    - 创建一个与 “full_sequence” 等长的bool张量(或者0,1也行)。

    - <strong class="list-label">Prompt部分</strong>:设为 “False”。

    - <strong class="list-label">Output部分</strong>:设为 “True” 。

    - <strong class="list-label">Padding部分</strong>:设为 “False” 。

4.  <strong class="list-label">构建自回归训练数据</strong>:

    - <strong class="list-label">input_ids</strong>:取序列的 <strong class="key-term">第0个 到 倒数第2个</strong> Token(切去最后一个)。

    - <strong class="list-label">labels</strong>:取序列的 <strong class="key-term">第1个 到 最后一个</strong> Token(切去第一个)。

    - <strong class="list-label">response_mask</strong>:从mask里进行相应的切片,与 “labels” 对齐。

#### 11.3.2.2 具体示例

假设 Tokenizer 词表映射为:“"A": 1”, “"B": 2”, “"C": 3”, “"D": 4”。

- <strong class="list-label">Prompt</strong>:"A B" -\> ids: “\[1, 2\]”

- <strong class="list-label">Output</strong>:"C D" -\> ids: “\[3, 4\]”

<strong class="key-term">数据流转如下:</strong>

1.  <strong class="list-label">完整序列</strong>: “\[1, 2, 3, 4\]” (对应 "A B C D")

2.  <strong class="list-label">原始掩码</strong>: “\[0, 0, 1, 1\]” (0代表Prompt "AB",1代表Output "CD")

3.  <strong class="list-label">切片处理</strong>:

    - input_ids:“seq\[:-1\]” -\> “\[1, 2, 3\]” (输入 A, B, C)

    - <strong class="list-label">labels:</strong>“seq\[1:\]” -\> “\[2, 3, 4\]” (预测目标 B, C, D)

    - response_mask:“mask\[1:\]” -\> “\[0, 1, 1\]”

      - 第一个 “0” 对应 label “2(B)”:这是Prompt的一部分,我们不希望计算loss,不强迫模型学习Prompt。

      - 第一个 “1” 对应 label “3(C)”:这是Output的第一个词,要计算loss。

      - 第二个 “1” 对应 label “4(D)”:这是Output的第二个词,要计算loss。

### 11.3.3 作业2“compute_entropy” 要求:

计算 LLM 在生成每个 Token 时的<strong class="key-term">预测熵；逐 Token 熵(Per-token Entropy)</strong> 是对模型在特定位置生成的概率分布<strong class="key-term">不确定性的定量度量</strong>(在信息论里熵本身就代表了不确定性)；熵值越高表示模型对下一个 Token 的选择越困惑,熵值越低则表示模型越趋向于输出某些特定 Token(即预测越自信)。

#### 11.3.3.1 实现思路

输入是<strong class="key-term">Logits</strong>,即经过深度学习模型最后一个线性层输出的<strong class="key-term">未经归一化的原始分值向量</strong>。

1.  <strong class="list-label">Logits 归一化</strong>: 输入 “logits” 的维度为 (B,L,V)。首先需将其通过 Softmax 函数转换为概率分布 p(x),公式为:

    $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

2.  <strong class="list-label">对数概率计算</strong>: 计算 logp(x)。为避免数值溢出,不直接对 Softmax 结果取对数,而是使用 “F.log_softmax” 或通过公式 $\log p_i = z_i - \text{logsumexp}(\mathbf{z})$ 直接从 Logits 计算。(<strong class="key-term">具体原理细节可详见第一章笔记</strong>)

3.  <strong class="list-label">应用熵公式</strong>: 根据定义 $H(p) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$,将每个 Token 位置对应的词表维度(Vocab)进行加权求和。

4.  <strong class="list-label">维度压缩</strong>: 在最后一个维度(“vocab_size”)上进行求和,最终输出维度为 (B,L) 的张量,其中每个元素代表该位置的预测不确定性。

### 11.3.4 作业3“tokenize_prompt_and_output” 要求:

给定一个因果语言模型(Causal LM)和一段文本序列,计算模型生成该序列中<strong class="key-term">特定 Token(即标签 Labels)的条件对数概率</strong>,并可以选择性地返回我们在上一步计算的<strong class="key-term">熵(Entropy)</strong>。

#### 11.3.4.1 实现思路

1.  <strong class="list-label">获取原始 Logits</strong>: 将 “input_ids” 输入模型,得到 “logits”。其维度为 (B,L,V),其中 B 是 Batch Size,L 是序列长度,V 是词表大小。

2.  <strong class="list-label">对数归一化</strong>: 对 “logits” 使用 “log_softmax” 操作,将原始分值转换为对数概率 logp。这一步必须在词表维度(“dim=-1”)上进行。

3.  <strong class="list-label">对齐标签</strong>: 在因果语言模型中,位置 t 的 Logits 是用来预测位置 t+1 的 Token 的。因此,为了获取 “labels”\[每个位置的对数概率(labels指明了哪些位置的token是和训练对应的): - "需要将 "log_probs" \[前 L−1\]个位置与 "labels" \[后 L−1\]个位置对应。 - "或者根据具体框架实现,确保取出的概率值 $p(y_t)$ \[是基于序列 x\<t\]计算得出的。"

4.  <strong class="list-label">提取目标 Token 的概率</strong>:使用 “torch.gather” 函数,根据 “labels” 提供的索引,从 (B,L,V) 维度的对数概率张量中提取出对应的标量值。

5.  <strong class="list-label">计算熵(optional)</strong>:如果 “return_token_entropy” 为 “True”,调用之前实现的 “compute_entropy” 函数,并将其与对数概率一起存入字典。

#### 11.3.4.2 具体示例

假设输入 “input_ids” 为 “\[I, love, AI\]”,对应的 Token ID 为 “\[10, 25, 40\]”:

1.  模型在位置 “I” 输出的分布中,我们提取 Token “love” (ID 25) 的对数概率。

2.  模型在位置 “love” 输出的分布中,我们提取 Token “AI” (ID 40) 的对数概率。

3.  最终返回的结果将是 “\[log_p(love\|I), log_p(AI\|I love)\]”。

### 11.3.5 作业4“mask_normalize” 要求:

根据一个布尔掩码对输入张量的指定元素进行求和,然后将该和值除以一个常数进行归一化。 非常easy。

### 11.3.6 作业5“get_response_log_probs” 要求:

对一个<strong class="key-term">微批次(micro-batch)</strong>数据执行一次完整的<strong class="key-term">前向计算(计算损失)</strong>和<strong class="key-term">反向传播(计算梯度)</strong>。

1.  <strong class="list-label">计算掩码损失</strong>:

    - 输入“policy_log_probs”是模型对于每个token给出的对数概率,其形状为 “(batch_size, sequence_length)”。

    - 输入“response_mask”是一个与“policy_log_probs”形状相同的张量,其中 <strong class="key-term">1</strong> 代表这是模型需要学习和生成的“回答”部分,<strong class="key-term">0</strong> 代表这是输入的“提示”或填充部分。

    - 我们的目标是让模型只学习“回答”部分。因此,需要将“policy_log_probs”与“response_mask”进行<strong class="key-term">逐元素相乘</strong>,这样“提示”部分的对数概率就全部变为0,不会对总损失产生贡献。

2.  <strong class="list-label">标准化损失</strong>:

    - 将上一步得到的总和除以 “normalize_constant”。

3.  <strong class="list-label">梯度累积缩放</strong>:

    - 因为我们最终要累积“gradient_accumulation_steps”次梯度,为了使最终累积的梯度在数值上等价于单个大批次的梯度,每次计算出的微批次损失需要<strong class="key-term">除以</strong> “gradient_accumulation_steps”。

4.  <strong class="list-label">反向传播</strong>:

    - 对缩放后的损失“scaled_loss”调用“.backward()”方法。PyTorch会自动计算出模型参数关于这个“scaled_loss”的梯度,并将其存储在各个参数的“.grad”属性中。由于我们不会在每次微批次后清空梯度,这个新计算的梯度会自动累加到已有的梯度上。

### 11.3.7 梯度累积

<strong class="key-term">梯度累积(Gradient Accumulation)是一种优化显存的技术</strong>,通过将一个大的全局批次拆分为多个微批次分次执行前向与反向传播,并将计算得到的梯度在内存中进行线性叠加,直到达到预设的累积步数后才执行一次参数更新,从而在数学上等效于在大批量数据下进行训练。<strong class="key-term">本质就是一种以时间换空间的做法。</strong>

#### 11.3.7.1 技术原理

梯度累积利用了深度学习框架中梯度默认累加的特性。其执行流程如下:

1.  <strong class="list-label">批次拆分</strong>: 设定目标全局批次大小和物理微批次大小。计算累积步数: $\text{Accumulation Steps} = \frac{\text{Global Batch Size}}{\text{Micro-batch Size}}$

2.  <strong class="list-label">标准化损失计算</strong>: 在每个微批次的前向传播计算出 Loss 后,必须将其除以累积步数。 $\text{Scaled Loss} = \frac{\text{Loss}}{\text{Accumulation Steps}}$

3.  <strong class="list-label">反向传播与累积</strong>: 执行反向传播。此时框架计算当前微批次的梯度,并将其直接加到模型参数的 ‘.grad‘ 属性上,而不是替换原有的梯度。在此过程中,<strong class="key-term">不执行优化器的更新和清零操作</strong>。

4.  <strong class="list-label">参数更新</strong>: 重复2和3,直到执行完指定次数的微批次。此时,模型参数中存储的梯度即为整个全局批次的梯度总和。调用优化器更新模型权重。

5.  <strong class="list-label">梯度清零</strong>: 权重更新完成后,清空所有参数的梯度,准备开始下一个全局批次的循环。

#### 11.3.7.2 具体示例

假设显存仅能支持 Batch Size = 4,但为了保证模型收敛稳定性,需要使用 Batch Size = 32 进行训练。

<strong class="note-label">参数设定:</strong>

- <strong class="list-label">物理限制(Micro-batch):</strong>4

- <strong class="list-label">目标批次(Global Batch):</strong>32

- <strong class="list-label">累积步数:</strong>32/4=8

<strong class="note-label">数据流转:</strong>

1.  <strong class="list-label">循环执行</strong>:

    - 加载 4 条数据。

    - 前向传播计算 Loss。

    - Loss = Loss / 8。

    - 反向传播,计算梯度 $g_i$,累加到显存:$G_{total} = G_{total} + g_i$

    - <strong class="note-label">注意</strong>:此时权重 W 保持不变。

2.  <strong class="list-label">更新步骤</strong>:

    - 加载最后 4 条数据。

    - 前向传播,Loss / 8,反向传播。

    - 此时 $G_{total}$ 包含了 32 条数据的梯度信息。

    - 执行 Optimizer Update:$W_{new} = W_{old} - \eta \cdot G_{total}$

    - 执行 Optimizer Zero Grad:$G_{total} = 0$

## 11.4 专家迭代实验(Expert Iteration)

前面已经介绍了生成、评分和 SFT 组件，本节将它们连接成“生成候选 → 筛选 → 监督训练 → 再评估”的循环。经典拒绝采样的独立背景见[对应补充节](/part-5/chapter-13#supp-rejection)。

### 11.4.1 从传统 SFT 到专家迭代

- 直接使用标准答案（ground truth）进行<strong class="key-term">监督学习</strong>。

- 模型只学习“正确答案”，不参与探索过程。

- <strong class="list-label">优点：</strong>简单直接。

- <strong class="list-label">缺点：</strong>无法利用模型自身探索能力，数据质量完全依赖人工标注。

### 11.4.2 专家迭代（Expert Iteration）

专家迭代的核心思想是：

<strong class="key-term">模型自己探索 → 筛选优质答案 → 用优质答案正反馈训练 → 模型变强 → 继续探索</strong>

具体流程：

1.  先自己尝试解题（<strong class="key-term">生成多个答案</strong>）

2.  找出做对的题（<strong class="key-term">过滤</strong>）

3.  只练习做对的题（SFT训练）

4.  下一轮更可能做对更多题。

### 11.4.3 详细工程步骤

1.  <strong class="note-label">初始化和基准评估</strong>

    - 初始化 vLLM 框架

    - <strong class="list-label">在一切训练开始前：</strong>

      - 先跑完整验证集

      - 记录 baseline 性能

      - 作为后续对比基准

2.  <strong class="note-label">数据抽样</strong>

    - 从 ‘train.jsonl‘ 抽样问题

    - 控制批量规模（避免 rollout 成本过高）

3.  <strong class="note-label">Rollout 生成</strong>

    - 使用当前专家模型

    - 对每个问题生成多个回答

    - 形成候选答案集合

4.  <strong class="note-label">奖励函数过滤</strong> 使用ground truth和评分函数进行打分和筛选。

5.  <strong class="note-label">SFT 训练</strong>

    - 使用 transformers 加载模型到训练卡

    - 使用筛选后的高质量数据集

    - 执行 SFT 微调

    - 产出新权重

6.  <strong class="note-label">更新 vLLM 权重（关键优化）</strong>

    - 避免每轮重复初始化 vLLM

    - <strong class="list-label">每轮训练后：</strong>

      - 直接更新权重

      - 不重新构建推理引擎

7.  <strong class="note-label">验证集评估</strong>

    - <strong class="list-label">每轮迭代后：</strong>跑验证集,记录性能变化

    - <strong class="list-label">最终：</strong>完整评估验证集,输出最终结果

实验日志：

1.  模型性能不佳，导致每一轮输出的多个答案都被过滤掉了，即输出质量差，奖励低。还是要至少确保每一组rollout都起码有一个被保留。以前是“只收 strict reward\>0”，太容易全空。现在改为“严格+宽松双层”,提取后结果再判断。

2.  模型与推理引擎初始化。训练模型用 transformers 加载到训练卡。 生成模型用 vLLM 初始化到评估卡，并做了 xformers 相关 patch，目的是让 V100 上更稳定运行。
