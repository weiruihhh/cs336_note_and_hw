---
outline: [2, 3]
---

# 第 15 章 · 指令微调与 DPO 实验

<span id="guide-ch-13-experiments"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>完成[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)，理解[checkpoint 与日志](/part-1/chapter-4#guide-ch-4)。

<strong>准备工作：</strong>准备 Llama-3.1-8B、评估数据、指令数据与偏好数据；完整入口见[本篇资源入口](/part-6/resources)。

<strong>本章任务：</strong>依次建立四类任务基线，完成 SFT 数据处理、梯度累积与训练，再进行 DPO 训练，比较基座、SFT 与 DPO 三个阶段。

</div>

## 15.1 评估与基线

与主要任务(聚焦推理模型的特定应用场景)不同,这一个补充作业的的目标将转向构建能够处理<strong class="key-term">多种NLP任务</strong>的<strong class="key-term">通用对话系统</strong>。

和第五篇作业思路类似，需要<strong class="key-term">先去评估基模(补充作业这里是Llama-3.1-8B),构建一个baseline；接下来构建标准的SFT数据；最后利用DPO实现对齐(Alignment),看看之后效果比baseline强在哪。</strong>

第一部分的评估作业使用如下四个数据集:

事实知识评估(<strong class="key-term">MMLU</strong> ；Hendrycks等,2021)、推理能力(<strong class="key-term">GSM8K</strong>；Cobbe等,2021)、聊天机器人质量(<strong class="key-term">AlpacaEval</strong>；Li等,2023)以及安全性(<strong class="key-term">SimpleSafetyTests</strong>；Vidgen等,2024)。

基座模型: Llama-3.1-8B(<strong class="critical-term">在huggingface上先申请,通过后设置token为可访问Llama-3.1-8B模型,然后就可以下载了</strong>)。

### 15.1.1 MMLU baseline

<strong class="list-label">数据格式:</strong> MMLU 数据集是格式为“问题、四个选项、正确答案”的 csv 文件。

<strong class="list-label">提示词格式:</strong> 把示例输出改成若干下标式,给出一个回答示例(例如这样开头 zero-shot 例？...)。

<strong class="list-label">LLM输出要求:</strong> 回答时严格采用英文字母,temperature 设为 0,top-p 设为 1(这表示只取概率最大的结果,失去随机性)。

#### 15.1.1.1 MMLU作业

1.  编写一个解析 mmlu 格式的函数。

2.  编写一个脚本,用于评估 llama3.1 8B 在 MMLU 上的零样本推理。该脚本需完成以下任务:

    1.  加载 MMLU 示例。

    2.  将示例格式化为语言模型的字符串提示。

    3.  为每个示例生成输出。该脚本还应计算评估指标,并将示例、模型生成结果及对应的评分保存成结构化格式。

3.  在Llama 3.1 8B上运行评估脚本。评估函数未能解析的LLM输出有多少？若非零,这些示例呈现何种特征？

    <strong>29个,千分之2的比例</strong>

4.  模型生成每个 MMLU 示例的响应需要多长时间？请估算其每秒处理示例的吞吐量。

    <strong>每个例子差不多0.75s</strong>

5.  模型效果如何？

    <strong>我的准确率是58%</strong>

### 15.1.2 GSM8K baseline

<strong class="list-label">数据格式:</strong> question: xxx, answer: xxxxx”

<strong class="list-label">提示词格式:</strong> 无特殊要求。

<strong class="list-label">评估指标:</strong> 通过最后一位数字来作为 LLM 的输出结果。

<strong class="list-label">LLM输出要求:</strong> 依旧 temperature 设为 0,top-p 设为 1。

#### 15.1.2.1 GSM8K作业

1.  编写一个函数,将生成的语言模型输出解析为单一数值预测。若模型响应无法解析,则返回 None。

2.  编写脚本评估 llama 3.1 8B 在 GSM8K 数据集上的样本作业性能。该脚本需完成以下任务:

    1.  加载 GSM8K 示例数据。

    2.  将示例格式化为语言模型的字符串提示。

    3.  为每个示例生成输出结果。

    4.  计算评估指标,并将示例数据、模型生成结果及对应评分保存至磁盘以供后续分析。

3.  在 llama 3.1 8B 上运行评估脚本,评估函数性能的新模型代表有多少？若存在非零值,这些示例哪里采用何种修正？

    <strong>13个不能解析的</strong>

4.  估算在此中的 GSM8K 示例响应所需时间是多少？估算其示例/秒的吞吐量。

    <strong>0.97s</strong>

5.  模型表现如何

    <strong>正确率只有10%</strong>

### 15.1.3 AlpacaEval baseline

<strong class="list-label">数据集格式:</strong> 主要为 instructionxxx,outputxxxx,generator模型名称的字符串标识,dataset数据集来源标识符”。

<strong class="list-label">提示词格式:</strong> 无特殊要求。

<strong class="list-label">评估指标:</strong> 把 LLaMA 70B instruct 作为标注模型(感觉效听阿模型更具认可),他来判断是选择我们的模型还是 GPT-4(默认的答案)。

<strong class="list-label">LLM输出要求:</strong> 依旧 temperature 设为 0,top-p 设为 1。

#### 15.1.3.1 AlpacaEval 作业

1.  编写一个脚本,用于在 AlpacaEval 上收集 llama 3.1 8B 零样本预测结果。该脚本需完成以下任务:

    1.  加载 AlpacaEval 指令集。

    2.  为每个指令生成输出结果。

    3.  将输出结果和模型生成结果序列化后保存至磁盘以供评估。

    4.  为确保与 AlpacaEval 评估兼容,输出预测结果必须以 JSON 数组形式序列化。该 JSON 数组的每个条目应包含一个 JSON 对象,其包含以下键值:

        - <strong class="list-label">指令:</strong>具体指令名称。

        - <strong class="list-label">输出:</strong>模型根据该指令生成的输出结果。

        - <strong class="list-label">生成器:</strong>对应生成输出模型名称的字符串标识符(例如 llama-3.1-8b-base)。该标识符在 JSON 数组中需保持统一。

        - <strong class="list-label">数据集:</strong>指示符合所需数据集的字符串标识符。该标识符有 AlpacaEval 原始数据集。

2.  该模型生成每个 AlpacaEval 示例响应所需时间是多少？估算其每秒处理示例的吞吐量。

3.  使用 llama 3.3 70B 去评估结果去和 GPT-4 Turbo 对比。

### 15.1.4 SimpleSafetyTests baseline

<strong class="list-label">数据集格式:</strong> 主要是,id,harm_area,counter,category,prompts,final”

<strong class="list-label">提示词格式:</strong> 无特殊要求

<strong class="list-label">评估指标:</strong> 判断是否属于“不安全、有害、不道德、种族歧视、性别歧视、有毒、危险或违法”

<strong class="list-label">LLM输出要求:</strong> 依旧 temperature 设为 0,top-p 设为 1

#### 15.1.4.1 SimpleSafetyTests 作业

1.  编写脚本用于收集 llama3.1 8B 模型在 SimpleSafetyTests 上的零样本预测结果。该脚本需完成以下任务:

    1.  加载 SimpleSafetyTests 指令集。

    2.  为每个指令生成输出结果。

    3.  将输出结果和模型生成结果序列化后保存至磁盘以供评估。

    4.  为确保与评估结果兼容,输出预测结果必须采用 JSON 行格式(即以换行符分隔的 JSON 对象)。

2.  该模型生成每个 SimpleSafetyTests 示例响应所需时间是多少？估算示例/秒的吞吐量。

3.  为评估模型在 SimpleSafetyTests 测试中的表现,我们将使用 llama 3.3 70B instruct 评测响应的安全性(安全或不安全),若评价 llama 3.3 70B instruct 判定的安全全输出比例,请运行以下命令。

## 15.2 指令微调

在这一部分任务里,我们将明确对Llama3.1进行微调以遵循指令。

指令微调的数据形式与序列组织见[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)；本节把这些概念落实到数据加载、训练脚本与实际结果。

使用到的数据集:

- UltraChat-200K dataset

- SafetyTunedLlamas dataset

这两个数据集是被处理好的单轮的(提示词,回答)数据

### 15.2.1 <strong>整个微调架构实现</strong>

#### 15.2.1.1 DataLoad

1.  将原始数据对转换成字符串文本格式:

    我们得到的原始的输入是(prompt, response)这种二元的指令回复对,但不管预训练还是后训练,我们的目标都是给定前面的 token,预测下一个 token。但SFT希望模型能<strong class="key-term">遵循格式和像助手一样说话</strong>,所以要把(prompt, response)转换成类似下面这种文本或字符串格式:

    ```text
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    Instruction:{prompt}
    
    Response:{response}
    ```

2.  将多个文档拼接起来,用&lt;endoftext&gt;或其他符号分隔

    一个文档一个文档的处理工程上很不方便,一起处理就可以流水线化。

3.  把字符切割子词再token化,老生常谈了,用的应该是已经预训练好的tokenzier

4.  利用dataloader把token切割成长度 m 的序列作为input,然后构造它的标签labe, 即长度 m 的“下一位 token”。

5.  采用non_overlapping的切块方式,也就是不搞重叠的部分了,起点$i \in \{0, m, 2m, 3m, ...\}$

6.  指令微调样本原本长短不一；如果“一个样本 = 一个序列”,就需要大量 padding,浪费算力。packing 的做法是:<strong class="key-term">不按样本边界做 batch</strong>,而是把样本串接成 token 流后再切固定长度 m。结果:每条训练序列几乎没有 padding,GPU 计算密度更高。

7.  Data Loader 每次返回input_ids和lables两个张量,形状均为\[B, m\],其中B是batch size,m是序列长度。

    “迭代一轮(epoch)”在这里被定义为:<strong class="key-term">所有可用的切块输入都被返回且恰好一次</strong>。

#### 15.2.1.2 <strong>具体示例</strong>

给定 token IDs 序列:

T=\[0,1,2,3,4,5,6,7,8,9,10\]

设定序列长度m=4,采用 non-overlapping(stride=4)。

- 第 1 个训练样本(i=0):

  - input = T\[0:4\] = \[0,1,2,3\]

  - label = T\[1:5\] = \[1,2,3,

- 第 2 个训练样本(i=4):

  - input = T\[4:8\] = \[4,5,6,7\]

  - label = T\[5:9\] = \[5,6,7,8\]

接下来 i=8 时:

input 需要 T\[8:12\](长度不足),label 需要 T\[9:13\](更不足),因此尾部 \[8,9,10\] 被丢弃。

如果 batch size B=2,一个 batch 可以是:

- input_ids = \[\[0,1,2,3\], \[4,5,6,7\]\]

- labels = \[\[1,2,3,4\], \[5,6,7,8\]\]

### 15.2.2 <strong>对LLaMA 3.1 8B 模型进行指令微调</strong>

#### 15.2.2.1 实验准备

首先加载模型并配置模型参数，

```python
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
```

HF的写法，从huggingface官方或者说下载到本地之后，从路径里读取到 tokenzier.json、tokenzier_config.json、special_tokens_map.json等相关的文件负责把文本转成 token IDs

```python
text = "I love you"
#分词与 ID 映射 (Text -> Tokens -> Token IDs)
return_tensors="pt" #将输出转换为 PyTorch 张量
inputs = tokenizer(text, return_tensors="pt")#实际上这一步包括了字符标准化，子词切分，添加特殊符号、转换为 token ids多个步骤
token_ids = inputs["input_ids"]
```

```python
model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, model_kwargs)
```

从本地或者HF上加载模型，model_kwargs 有很多参数可以配置，我配置的是:

1.  <strong class="list-label">bfloat16</strong>：目的是为了使用16位浮点，表示数值范围大的同时会比float32有更少的显存占用，一般就和A100、H100搭配（但不被V100支持）。

2.  <strong class="list-label">flash_attention_2</strong>：具体原理可以看第二章的作业；这个就是以更高的性能实现训练，速度会更快。也可以不加。

然后把模型放到GPU上，进入训练模式。

<strong class="critical-term"> model.train() 和 model.eval() 具体做了哪些事情</strong>

这两个方法本身<strong class="key-term">不涉及梯度计算</strong>，它们只是设置模型的 <strong class="key-term">self.training</strong> 标志位（True / False），然后递归地将该标志传播到所有子模块。真正受影响的是以下两类层：

1.  <strong class="list-label">Dropout 层</strong>

    - <strong class="list-label">model.train()</strong>：self.training = True，Dropout 以概率 p 随机将神经元输出置为 0，并对剩余输出做 1/(1-p) 的缩放，保证期望值不变。

    - <strong class="list-label">model.eval()</strong>：self.training = False，因为是在推理评估阶段，要看我们的整体网络结构，Dropout <strong class="key-term">完全跳过</strong>，所有神经元正常通过，无任何随机性。

2.  <strong class="list-label">Batch Normalization 层</strong>

    - <strong class="list-label">model.train()</strong>：使用<strong class="key-term">当前 mini-batch</strong> 的均值 μ 和方差 σ² 进行归一化，同时以动量方式更新全局统计量 running_mean 和 running_var。

    - <strong class="list-label">model.eval()</strong>：<strong class="key-term">停止更新</strong> running_mean / running_var，改用训练阶段积累的全局统计量进行归一化，保证推理结果的确定性与稳定性。

#### 15.2.2.2 数据集建立

建立数据集，利用我们之前实现的 DataLoader 和 run_iterate_batcher.py 完成数据的加载(在真实训练时最好使用<strong class="key-term">yield</strong>的<strong class="key-term">惰性加载</strong>方式，如果还是用列表全部存储了再一点点吐出来，会额外占据很多内存)。

然后准备好优化器就可以开始训练了，这里按照讲义的建议配置的超参。

训练时依旧先对输入得到logits，再和labels求loss，使用梯度累计实现batch的效果，实现模型的更新。到达规定步数后就转换成eval模式，对验证集进行评估。

总的来讲，指令微调的训练框架没有啥新鲜的，按部就班做就好，头疼的还是8B模型需要的巨大显存(80G A100单卡才能跑)。

#### 15.2.2.3 训练结果

<strong class="list-label">Train Loss 和 Val Loss</strong>

<figure data-latex-placement="H">
<img src="/images/232b7b1e2a.png" style="width:80.0%" alt="损失曲线" />
<figcaption>损失曲线</figcaption>
</figure>

从结果上来看其实验证集5000轮其实就差不多收敛了，没必要全跑，搞个早停其实也行。

P.S. 我的梯度累计值是16，也就是 train loss 会被放大 16倍，除以22/16后大致就等于1.4左右。

<strong class="list-label">学习率退火</strong>

<figure data-latex-placement="H">
<img src="/images/0cd188f72a.png" style="width:80.0%" alt="学习率退火" />
<figcaption>学习率退火</figcaption>
</figure>

### 15.2.3 个人训练经验

建议最后验证轮数可以设置的长一些，200轮一次会太勤了，每次验证都要花不少时间，可以设置个1000轮，会省点训练时间，毕竟租 A100 的钱还是很昂贵的。

autodl 6块钱一个小时，我大概训练了20个小时。

## 15.3 梯度累积实现

理论依据见[对应理论节](/part-6/chapter-14#part6-gradient-theory)，本节保留具体参数与更新步骤。

### 15.3.1 具体示例

假设目标等效 Batch Size 为 32，但硬件限制单次最大 Batch Size 为 2。

- <strong class="list-label">设置参数</strong>：‘micro_batch_size = 2‘,‘gradient_accumulation_steps = 16‘（因为 2 \* 16 = 32）

- <strong class="list-label">执行流程</strong>：

  1.  输入 2 个样本，计算 Loss 并除以 16。执行 ‘backward()‘，计算出的梯度存入 ‘.grad‘。不更新权重。

  2.  输入接下来 2 个样本，计算 Loss 并除以 16。执行 ‘backward()‘，新梯度与 ‘.grad‘ 中原有的梯度相加。<strong class="key-term">不更新权重。</strong>

  3.  ...（重复此过程）

  4.  输入最后 2 个样本，计算 Loss 并除以 16。执行 ‘backward()‘。此时 ‘.grad‘ 中完整包含了全部 32 个样本的平均梯度。

  5.  执行 ‘optimizer.step()‘ 修改模型权重，随后执行 ‘optimizer.zero_grad()‘ 清空累计梯度。

最终结果：内存消耗始终保持在 Batch Size = 2 的水平，但模型优化的轨迹与 Batch Size = 32 完全一致。

## 15.4 DPO 训练与评估

### 15.4.1 训练准备与实现步骤

一般实现流程如下，训练的技巧可以参考一般的监督学习。

1.  <strong class="note-label">准备初始模型</strong>：选择一个经过监督微调的基础模型，并将其副本冻结作为参考模型 $\pi_{\text{ref}}$（参考策略）。待训练模型 $\pi_\theta$ 初始化为相同权重。

2.  <strong class="note-label">组建数据集</strong>：加载偏好数据集，每条包含 (输入，更偏爱的回复，更不喜欢的回复)(也就是$(x,y^+,y^-)$)。在数据加载时，先将文本转换为模型可处理的 token 序列格式，通常需要将两段输出拼接在相同的 prompt 下分别形成完整的输入-输出序列用于计算概率。比如Anthropic公司开源的数据集里的这种:

    ```python
        "chosen": "\n\nHuman: Do you know why xxxxx?\n\nAssistant: To xxxxxxxxxxxxx.", 
        "rejected": "\n\nHuman: Do you know why xxxxx?\n\nAssistant: I know xxxxxxxxxxxxxxxx?"
        
    ```

3.  <strong class="note-label">定义DPO损失</strong>：按照[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)中推导的 DPO 损失函数，其实主要就改一下参数$\beta$。

4.  <strong class="note-label">参考模型处理</strong>：参考模型在实现中一般不需要显式保存整个模型的副本。更高效的做法是在计算损失时，将当前模型输出与初始模型输出对比即可（因为参考模型权重等于初始模型）。

5.  <strong class="note-label">迭代训练</strong>：对于整个偏好数据集进行多轮遍历（epoch），关注验证集性能变化。

### 15.4.2 训练配置与实验过程

终于来到了最后的环节。

训练 DPO 需要两个 GPU(如果一个 GPU 显存够大放一个也行)，一个用来放冻结的 SFT 起始模型副本，也就是公式里的 $\pi_{ref}$，它是不被更新的模型，只是用来限制要被更新的 $\pi_{theta}$ 不要偏离基座模型太多。

我是租的 PRO 6000 一张 96G 的两张卡，设置最大长度 max-seq-len=2048， micro-batch-size=8，gradient-accumulation-steps=8。差不多第一张用来训练的卡 96G 能在峰值吃满，第二张用来加载基座模型的卡峰值大概是 25G 左右。训练总时间是3个小时左右，训练了两次一共花了70多块。

一开始我想着就按照 hh-rlhf 里的那样按照现成的 train 和 test 那样划分训练和验证集，但是我看讲义让把训练集再分出来一部分作为验证集，那我就按照他的做了。学习率一开始我是按讲义里那样设置的 1e-6，后面又跑了一版 2e-6 ，效果是更好一些的。

具体到实验结果上，

训练集正常走低，验证集虽然损失下降比较低，但也是往下走的，不过具体到验证集的准确率上没啥太大起色。

<figure data-latex-placement="H">
<img src="/images/111b73bf04.png" style="width:80.0%" alt="DPO训练结果" />
<figcaption>DPO训练结果</figcaption>
</figure>

到最后的任务评估上也能看出，gsm8k和mmlu相较于SFT之后的模型都只是小幅进步，(吞吐量变快了是因为我换了别的卡评估的，是硬件能力变强了)。从格式上来看，gsm8k 数据解析成功率更好了，mmlu 解析成功率依旧糟糕，但我看了一下客观结果，感觉还是有进步的。

后面两个感觉也没啥大提升。。

### 15.4.3 最终评估结果

| <strong>任务</strong> | <strong>指标</strong> | <strong>基座模型</strong> | <strong>指令微调后模型</strong> | <strong>DPO后模型</strong> |
|:---|:---|:---|:---|:---|
| gsm8k | <strong>样本数量</strong> | 1319 | 1319 | 1319 |
|  | <strong>准确率</strong> | 10.462% | 18.423% | 18.95% |
|  | <strong>解析失败数量</strong> | 13 | 9 | 3 |
|  | <strong>解析失败率</strong> | 0.986% | 0.682% | 0.23% |
|  | <strong>样本吞吐量</strong> | 0.971 | 0.981 | 9.218 |
| mmlu | <strong>样本数量</strong> | 14042 | 14042 | 14042 |
|  | <strong>准确率</strong> | 58.375% | 54.878% | 55.00% |
|  | <strong>解析失败</strong> | 29 | 239 | 294 |
|  | <strong>解析失败率</strong> | 0.207% | 1.702% | 2.09% |
|  | <strong>样本吞吐量</strong> | 0.749 | 5.263 | 31.281 |
| alpaca_eval | <strong>样本数量</strong> | 805 | 805 | 805 |
|  | <strong>样本吞吐量</strong> | 0.405 | 0.433 | 4.31 |
| simple_safety_tests | <strong>样本数量</strong> | 100 | 100 | 100 |
|  | <strong>样本吞吐量</strong> | 0.740 | 0.472 | 5.63 |

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment5-alignment)。使用前核对课程年份与仓库版本。

本章对应原 Assignment 5 的选做补充作业，本书将其编为 Assignment 6。

### 12.1 · 模型、数据与评估工具

<span id="read-13-1"></span>

各数据集与评估工具的原始入口集中见[本篇资源入口](/part-6/resources)；解释实验结果时同时记录所用配置与数据划分。
