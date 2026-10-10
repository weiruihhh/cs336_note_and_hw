---
outline: [2, 3]
---

# 第 10 章 · 数据处理与质量控制

<span id="guide-ch-11"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>熟悉文本文件、集合、哈希及[tokenizer](/part-1/chapter-1#guide-ch-1)；能运行[训练与验证流程](/part-1/chapter-4#guide-ch-4)。

<strong>准备工作：</strong>准备 Common Crawl 的 WET 文件和 Paloma 验证文本；获取及处理入口见[本篇资源入口](/part-4/resources)。

<strong>本章任务：</strong>完成网页文本处理、语种识别、隐私处理及精确/近似去重；生成 token 数据并比较过滤策略。

</div>

第四章的核心任务是构建一个处理Web数据的Pipeline。从原始的HTML网页上把原始数据清洗,得到高质量的结果作为训练集训练语言模型。

主要步骤:

1.  <strong class="list-label">提取 (Extraction)</strong>: HTML转文本。

2.  <strong class="list-label">过滤 (Filtering)</strong>: 去除有害内容、PII(个人信息)、低质量内容。

3.  <strong class="list-label">去重 (Deduplication)</strong>: 删掉重复信息。

数据来源:Common Crawl(CC) 一个公益网页爬虫平台

三种关键文件格式:

1.  <strong class="list-label">WARC (Web ARChive)</strong>: 最原始数据,包含 HTTP 请求头 + 原始 HTML。

2.  <strong class="list-label">WAT (Web Archive Transformation)</strong>: 元数据 (Metadata),JSON 格式,如链接、标题。

3.  <strong class="list-label">WET (Web Extracted Text)</strong>: 官方提取的纯文本。

## 10.1 Web网页文本提取

- <strong class="list-label">目标:</strong>从 raw HTML (bytes) 中提取纯净文本。

- <strong class="list-label">工具库:</strong> Resiliparse(讲义推荐,适合CC)

- <strong class="list-label">主要思路:</strong>

  1.  输入是网页内容str

  2.  先检测是什么编码格式(UTF-8, GBK, ...)

  3.  利用Resiliparse 进行HTML解析、提取纯文本

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · BeautifulSoup、Trafilatura、Resiliparse三种网页解析文本工具的比较</p>

BeautifulSoup 适合小规模爬虫,需要人工编写提取文本的规则。Trafilatura是目前LLM构建数据集里最流行、好用的方式之一,无需自己写规则,可以自动得到好的网页文本数据。Resiliparse优势在于处理超大规模的数据。

- <strong class="list-label">BeautifulSoup</strong>: 适用于<strong class="key-term">小规模、定向爬虫</strong>。例如:你需要从特定的10个技术博客抓取文章,且这些博客结构固定,你可以写死规则以获得100%的提取准确度。

- <strong class="list-label">Trafilatura</strong>: 适用于<strong class="key-term">构建通用预训练语料库(Pre-training Corpus)</strong>。例如:从任意URL列表中提取正文,不关心具体网站结构,追求在百万级网页上的平均高质量表现。它是目前RedPajama等数据集构建的首选工具之一。

- <strong class="list-label">Resiliparse</strong>: 适用于<strong class="key-term">超大规模清洗流水线(如处理CommonCrawl)</strong>。当你需要处理PB级数据,且主要瓶颈在于CPU解析HTML的时间和内存时,使用Resiliparse替换默认的解析器,或者用来做初步的ETL清洗。

</div>

## 10.2 语种识别(Language identification)

- <strong class="list-label">目标:</strong>LLM 训练通常集中在特定语言(如英语),混合语言数据可能不仅无用还会干扰模型,语种识别就是为了去除这种混合的语言数据。

- <strong class="list-label">技术方案:</strong> 使用分类器对文本进行打分(fastText)。

- <strong class="list-label">主要思路:</strong>

  1.  输入文本 -\> 输出‘(Language Label, Confidence Score)‘。

  2.  设定阈值(Threshold)过滤非目标语言文档。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · fastText技术原理</p>

fastText是一个<strong class="key-term">基于线性模型的浅层神经网络</strong>的模型,其核心优势在于在保持与深度学习模型相当精度的同时,推理速度比深度神经网络快几个数量级(通常在CPU上即可达到每秒百万词的处理速度)。

1.  <strong class="key-term">子词(Subword)N-gram 特征提取</strong>

    - 不同于传统的Word2Vec将每个单词视为原子单位,fastText 将单词拆解为字符级别的 N-gram。

    - <strong class="list-label">例如:</strong>单词 "apple" (n=3) 会被表示为 ‘\<ap, app, ppl, ple, le\>‘。

    - <strong class="list-label">优势:</strong>这使得模型能够处理<strong class="key-term">未登录词(OOV)</strong>,并捕捉词根、词缀等形态学信息(这对处理噪声很大的网页文本至关重要)。

2.  <strong class="key-term">特征平均与映射 (Hidden Layer Averaging)</strong>

    - 模型将输入文本中所有 N-gram 的向量进行<strong class="key-term">平均</strong>,得到整个文档的向量表示。

    - 这是一个简单的线性叠加过程,计算开销极低,不需要复杂的矩阵乘法或注意力机制计算。

3.  <strong class="key-term">层次 Softmax (Hierarchical Softmax)</strong>

    - 在输出层进行分类预测时,fastText 不使用标准的 Softmax(计算量随类别数线性增长 O(N))。

    - 它构建了一棵<strong class="key-term">霍夫曼树(Huffman Tree)</strong>,将标签预测转化为树上的路径搜索问题,计算复杂度降为对数级 O(logN)。这使得它能在几秒钟内处理数十万个类别的分类任务

具体示例: 在 LLM 数据清洗中,最典型的应用是<strong class="key-term">语种识别(Language identification)</strong>。

输入数据 (Input Document): "Hello world! 今天天气真不错。Voici un exemple."

处理流程:

1\. Tokenization: 拆解为字符 N-gram 序列。

2\. Vectorization: 查找 Embedding 表并取平均值。

3\. Inference: 模型计算属于各个语言标签的概率。

输出结果 (Output Prediction):

- ‘(English, 0.6)‘: 预测为英语,概率为 0.6。

- ‘(Chinese, 0.3)‘: 预测为中文,概率为 0.3。

- ‘(French, 0.1)‘: 预测为法语,概度为 0.1。

</div>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · N-gram</p>

N-gram(N元语法)是指在一段文本序列中,由连续出现的 N 个基本单元(Token,如字符、单词或字节)组成的滑动窗口片段。

| <strong>N 值</strong> | <strong>名称</strong> | <strong>提取出的 N-grams 集合</strong> | <strong>逻辑说明</strong> |
|:---|:---|:---|:---|
| N=1 | Unigram (一元) | \[我, 是, 一, 头, 猪\] | 仅关注单个字,完全丢失上下文顺序信息。 |
| N=2 | Bigram (二元) | \[我是, 是一, 一头, 头猪\] | 捕捉相邻两个字的搭配,能识别"我是"、"一头"等词汇。 |
| N=3 | Trigram (三元) | \[我是一, 是一头, 一头猪\] | 捕捉更长的语义依赖,"一头猪"被完整识别。 |

</div>

## 10.3 个人隐私处理

- <strong class="list-label">背景:</strong>防止模型在生成时泄露真实用户的隐私(训练数据记忆化问题)。

- <strong class="list-label">处理策略:</strong><strong class="key-term">掩码 (Masking)</strong>,即将敏感信息替换为特殊占位符。

- <strong class="list-label">实现思路:</strong>使用正则表达式匹配敏感信息,并将其替换为特殊占位符。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

简单的正则替换可能会有<strong class="critical-term">误报(False Positives)</strong>或<strong class="critical-term">漏报(False Negatives)</strong>,会对下游模型产生什么影响？

先不讨论漏报误报,即使正确把所有的关于个人隐私的信息包括ip,email,phone number全都按照设想转换成了‘\|\|\|IP_ADDRESS\|\|\|‘ ‘\|\|\|EMAIL_ADDRESS\|\|\|‘ ‘\|\|\|PHONE_NUMBER\|\|\|‘ ,那也会产生这三个替换的结果被过拟合。

漏报的话可能就使得模型学习到别人的隐私信息,产生<strong class="key-term">安全风险</strong>；误报的话就使得原本不是隐私的有价值的信息被替换,<strong class="key-term">降低模型的性能</strong>。

针对漏报,要加强一下正则之前的<strong class="key-term">前期处理</strong>,把格式搞得标准一些,不要因为一些全角、半角,中文等问题影响处理。

针对误报,可以增加一些<strong class="key-term">上下文的信息</strong>,如果一个“IP地址”前面出现了 "Version", "v", "Section" 等词,或者后面跟了 "release",那么可能就是软件版本号,而不是IP地址,产生了误报。

</div>

## 10.4 MinHash

[参考资料 10.1](/part-4/chapter-10#read-11-1)

### 10.4.1 为什么要使用MinHash算法(它是用来解决什么问题的)

在很多任务(例如搜索引擎的网页去重、推荐系统)中,我们关心的是<strong class="key-term">两个集合有多相似</strong>,典型例子:

- 两篇文章是否相似？

- 两个网页是否重复？

- 两段代码是否抄袭？

一个非常经典的相似度指标是:

<strong class="key-term">Jaccard 相似度:</strong>

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

两个集合的<strong class="key-term">交集占并集的比例</strong>,简单理解就是俩集合重合的也就是相同的内容占总内容的比例。 问题在于:

- <strong class="key-term">集合很大</strong>(几十万、几百万 token)<strong class="key-term">数据量极大</strong>(上亿文档)会相当占存储

- 直接算交并集会非常<strong class="key-term">消耗计算量,时间很漫长</strong>。

<strong class="list-label">MinHash 的解决方案</strong>:它将巨大的集合(高维向量)映射为一个<strong class="key-term">固定长度的短向量(签名/signature)</strong>,并保证:<strong class="key-term">两个集合签名的相似度,近似等于它们原始集合的 Jaccard 相似度。</strong>

### 10.4.2 MinHash的数学原理

核心思想:如果两个集合原本就很相似,或者说重复的元素很多,那么经过某种随机排列的方式之后,在这个新顺序中出现的第一个元素(实际为哈希函数映射后的<strong class="key-term">最小哈希值</strong>)也很可能相同。反之,如果两个集合原本就没啥共同元素,那么经过某种排列后新顺序下各自的第一个元素也大概率不同。因此,我们可以以这种方式来间接判断两个集合的Jaccard相似度。 用数学化的语言来表示: 定义 $h(S)$ 为集合 $S$ 在经过随机排列后,第一个出现的元素,其中$h()$表示某个哈希函数。

<strong class="key-term">MinHash 定理指出:</strong>

$$
P(h(A) = h(B)) \approx J(A, B)
$$

即两个集合经随机排列后,它们最小哈希值相等的概率,近似等于它们的 Jaccard 相似度。

### 10.4.3 MinHash算法流程

之前我们说了,我们希望用随机排列的手段获取新顺序下的最小哈希值；但如果要像random shuffle这种形式的随机排列的话是很耗时的。而使用哈希函数计算模拟随机排序的方法则更加省时。

#### 10.4.3.1 Random Shuffle 和 Hash的比较

e.g.想象一下实际场景:

- <strong class="list-label">全集元素(N)</strong>:词汇表可能有 <strong class="key-term">1,000</strong> 个词。

- <strong class="list-label">签名长度(K)</strong>:我们需要 <strong class="key-term">100</strong> 个哈希值来做签名。

<strong class="critical-term">方案 A:使用 Random Shuffle(随机排序)</strong>

1.  需要生成 100 个随机排列。

2.  每个排列是对 1000 个数字进行洗牌。这是巨大的内存和时间开销。

3.  <strong class="list-label">最致命的</strong>:如果要算出 $S_1$ 的签名,你需要遍历这 100 个长度为 1000 的排列序列,去寻找 $S_1$ 中存在的词。

    - <strong class="list-label">复杂度</strong>:$O(K \times N)$。$100 \times 1000 = 10$万 次操作才能算出一个文档的签名。

<strong class="critical-term">方案 B:使用哈希函数(MinHash 的做法)</strong>

1.  我们不需要真的把 1000 行重新物理排序。

2.  我们只需要定义 100 个简单的公式(例如 $h(x)=(3x+7) \bmod N$)。

3.  对于文档 $S_1$,假设它只包含 50 个词(也就是稀疏向量中只有 50 个 1)。

4.  我们只需要把这 50 个词的行号,带入到 100 个公式里算一下,剩下的950个元素和文档$S_1$没关系,因此不用管。

    - <strong class="list-label">复杂度</strong>:$O(K \times L)$ ($L$是文档的实际词数)。

    - $100 \times 50 = 5000$ 次操作。

#### 10.4.3.2 原始数据特征矩阵

我们先将原始数据给矩阵数学表示一下

| <strong>单词</strong> | <strong>文档1</strong> | <strong>文档2</strong> | <strong>文档3</strong> |
|:---|:---|:---|:---|
| <strong>单词1</strong> | 1 | 0 | 0 |
| <strong>单词2</strong> | 1 | 0 | 0 |
| <strong>$\cdots$</strong> |  |  |  |
| <strong>单词n</strong> | 0 | 0 | 1 |

- <strong class="list-label">定义</strong>:一个布尔矩阵(只有 0 和 1)。

- <strong class="list-label">行</strong>:代表<strong class="key-term">全集元素</strong>(例如:全部文档所有的 100 万个单词)。

- <strong class="list-label">列</strong>:代表<strong class="key-term">集合/文档</strong>(例如:你要处理的 600 个文档)。

- <strong class="list-label">数值</strong>:

  - `1`:表示该文档包含该单词。

  - `0`:表示该文档不包含该单词。

<strong class="critical-term">它的痛点:</strong>

1.  <strong class="list-label">极度稀疏 (Sparse)</strong>:因为一篇文档可能只有 500 个词,但词表有 100 万个词。所以这一列里有 99.95% 都是 0。

2.  <strong class="list-label">超级巨大</strong>:100万行$\times$ 600列,内存根本存不下,通常只能用稀疏矩阵存储格式(如链表)来存。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

在实际工程上,我们往往不会去单独存储原始数据的特征矩阵,因为太大了,而是会直接读取文档转换成签名矩阵。

</div>

### 10.4.4 MinHash签名矩阵

这是 MinHash 算法计算后的结果,也是我们真正用来做相似度计算的东西。

- <strong class="list-label">定义</strong>:一个整数矩阵。

- <strong class="list-label">行 (Rows)</strong>:代表<strong class="key-term">哈希函数</strong>(即我们前面说的 $K$,通常取 100$\sim$ 200)。

  - <strong class="key-term">这里的行不再是单词了,而是第几个哈希函数。</strong>

- <strong class="list-label">列 (Columns)</strong>:代表<strong class="key-term">集合/文档</strong>(这一点没变,列依然对应原来的文档)。

- <strong class="list-label">数值</strong>:存储的是该文档在该哈希函数下的<strong class="key-term">最小哈希值</strong>。

<strong class="critical-term">它的特点(优势):</strong>

1.  <strong class="list-label">高度稠密 (Dense)</strong>:里面填满了整数,没有那么多无意义的 0。

2.  <strong class="list-label">非常小巧</strong>:行数从 100 万变成了$K$(代表我们哈希函数的数量,也就是$K$种随机排列)。

#### 10.4.4.1 原始矩阵 -\> 签名矩阵

假设我们有 $K$ 个哈希函数 $h_1, h_2, \ldots, h_k$ 我们要构建签名矩阵 $M$,其中 $M(i, c)$ 表示第 $i$ 个哈希函数对第 $c$ 个文档算出的签名值。

| 哈希函数           | 文档1 | 文档2 | 文档3 |
|:-------------------|:------|:------|:------|
| $h_1$ | 1     | 5     | 6     |
| $h_2$ | 1     | 1     | 6     |
| $\cdots$ |       |       |       |
| $h_k$ | 4     | 4     | 6     |

#### 10.4.4.2 算法伪代码逻辑:

1.  <strong class="list-label">初始化</strong>:把签名矩阵 $M$ 的所有格子都填上无穷大 ($\infty$)。

2.  <strong class="list-label">遍历原始数据</strong>:

    我们一行一行地读取原始的特征矩阵(或者读文档中的词)。

    假设当前读到了 <strong class="key-term">行 $r$(代表单词 $w$)</strong>:

    - 先计算这一行的 $K$ 个哈希值:

      $$
      val_1 = h_1(r), \quad val_2 = h_2(r), \quad \ldots, \quad val_k = h_k(r)
      $$

    - 然后看哪些文档里有这个词(哪些列是 1):

      - 如果文档 $c$ 有这个词(特征矩阵中 $(r,c)=1$):

        - 我们就尝试更新签名矩阵的第 $c$ 列。

        - <strong class="list-label">规则</strong>:

          $$
          M(i, c) = \min(M(i, c), val_i)
          $$

        - *我之前记录的最小值是 $\infty$ 或者别的数,现在新来了一个单词,它的哈希值是 $val_i$,如果它更小,我就更新它。*

<figure data-latex-placement="H">
<img src="/images/b43c0ee853.png" style="width:80.0%" alt="原始矩阵到签名矩阵的转换示意图" />
<figcaption>原始矩阵到签名矩阵的转换示意图</figcaption>
</figure>

#### 10.4.4.3 LSH的分段(Band)和哈希桶

MinHash 本身只是生成了签名。如果我有 100 万个文档,即便有了签名,两两对比还是要算 $100\text{万}^2$ 次,依然很慢。 所以 MinHash 通常配合 <strong class="key-term">LSH 的 Banding 技术和哈希桶</strong>使用 LSH就像一个<strong class="key-term">“过滤器”</strong>:把不相似的文档(绝大多数)直接过滤掉。只有那极少数真正相似的文档,才会因为在某一个 Band 上特征一致,而被扔进同一个桶里,等待最终的核实。这就把 $O(N^2)$ 的复杂度降到了近似 $O(N)$ 。

#### 10.4.4.4 LSH 分段

<strong class="critical-term">核心目标:让相似的东西更容易被分到同一个桶里</strong>

而且:

- <strong class="key-term">不相似的东西大概率不会进同一个桶</strong>

- 只在同一个桶里做精算

做法: 我们将长长的签名向量(对应的是签名矩阵的某一列,比如 100 个整数),切成 $b$ 个段(Band),每段有 $r$ 行。

- <strong class="list-label">例如:</strong>签名长度 100,分成 <strong class="key-term">20 个段</strong>,每个段包含 <strong class="key-term">5 个整数</strong>。

#### 10.4.4.5 哈希桶

对于我们之前分的每一个段(Band),我们建立一个哈希表(Hash Table)。

- 对于文档 A,看它的第 1 段(包含5个整数,比如 `[12, 45, 1, 99, 3]`)。

- 将这 5 个数作为一个整体,哈希到一个桶里(Bucket)。

- <strong class="critical-term">核心逻辑</strong>:只有当两个文档在<strong class="key-term">同一个段内的数值完全一模一样</strong>,它们才会掉进同一个桶里。

#### 10.4.4.6 候选对生成

- 只要文档 A 和文档 B 在 <strong class="key-term">任意一个段</strong>(无论是第1段还是第20段)掉进了同一个桶,我们就说它们是"<strong class="key-term">候选相似对</strong>"。

- 如果不相似的文档,它们在每一个段里大概率数值都不一样,永远碰不到面。

这样做其实是充分利用了概率的想法(<strong class="key-term">S曲线效应</strong>),举个具体的数字例子: 还是假设分段 $b = 20$, 包含的整数数量 $r = 5$

- <strong class="list-label">如果两个文档很像 (有80%的内容一样,即$s=0.8$)</strong>:

  - 在一个段内撞桶概率:$0.8^5 \approx 0.32$

  - 在20个段里<strong class="key-term">至少</strong>有一次撞桶的概率:$1 - (1 - 0.32)^{20} \approx 0.999$

  - 结果:高相似文档几乎 100% 会被捕获。

- <strong class="list-label">如果两个文档不像 ($s=0.3$)</strong>:

  - 在一个段内撞桶概率:$0.3^5 \approx 0.002$

  - 在20个段里至少有一次撞桶的概率:$1 - (1 - 0.002)^{20} \approx 0.04$

  - 结果:低相似文档只有 4% 的概率会被误判为候选对(即便误判,后续算一下真实值也就排除了)。

#### 10.4.4.7 计算相似度

最后我们得到了这些候选对之后,我们再去计算他们的Jaccard相似度,通过签名矩阵看这两个文档

- 我们数一下这两列在<strong class="key-term">对应行</strong>上有多少个值是相等的。

- <strong class="key-term">相等行的数量 / 总行数 $K$ $\approx$ 这两个文档的 Jaccard 相似度。</strong>

## 10.5 精确去重与语义去重

最常见的几种去重方式包括<strong class="key-term">精确去重、模糊去重、语义去重</strong>。其中模糊去重我们上一章已经讲了,主要利用了MinHash+LSH去实现。 接下来我们重点关注精确去重和语义去重。

### 10.5.1 精确去重

精确去重是要求完全一模一样的字符,一般会使用<strong class="key-term">加密哈希匹配</strong>的方式去重。

假设我们要以文档为单位来去重,那么就可以对每一个文档进行SHA-256哈希算法,输出的哈希值存到<strong class="key-term">集合</strong>里,接下来的文档如果会和前面的哈希值重复,那么就表示这两个文档内容重复,被丢弃。

不过我在实际的作业完成中,是以文档中的行为单位,如果有重复的行就直接去重删掉。

#### 10.5.1.1 SHA-256加密算法

<strong class="key-term">SHA-256 (Secure Hash Algorithm 256-bit)</strong> 它的核心作用是将任意长度的输入数据,通过一系列复杂的<strong class="key-term">位运算</strong>和<strong class="key-term">非线性函数</strong>,转换为一个固定长度为 <strong class="key-term">256位 (32字节)</strong> 的输出(摘要digest/哈希值)。

在架构设计层面,我们需要关注以下几个核心特性:

1.  <strong class="list-label">确定性</strong>:对于相同的输入,必须永远产生相同的输出。

2.  <strong class="list-label">雪崩效应</strong>:输入数据哪怕改变 1 个bit,输出的哈希值也应该发生巨大且不可预测的变化。

3.  <strong class="list-label">单向性</strong>:从哈希值无法通过计算的方式反推原始数据(即不可逆)。

4.  <strong class="list-label">抗碰撞性</strong>:找到两个不同的输入产生相同的哈希值,在目前的算力下应当是不可能的(虽然理论上存在,但概率极低,约为 $\frac{1}{2^{128}}$)。

sha-256算法在python中调用非常方便,直接hash.sha256(),例如

```python
input_string = "i am pig"
# 创建 SHA-256 哈希对象
sha256_obj = hashlib.sha256()
# 更新缓冲区,必须先 encode 为 bytes
sha256_obj.update(input_string.encode(encoding))
# 返回十六进制摘要
return sha256_obj.hexdigest()
```

具体到SHA-256算法内部的数学操作,其实它是通过多次<strong class="key-term">移位</strong>和<strong class="key-term">异或</strong>等<strong class="key-term">位运算</strong>,使得数据无法还原成原有的样子。 SHA-256算法还有一个更简单的替代MD-5,也是比较原始的加密算法。

<strong class="critical-term">SHA-256 的缺陷</strong>

哈希算法(如 SHA-256)是确定性的。

- 如果用户 A 的密码是 ‘"123456"\`,哈希值永远是 ‘8d969e...‘。

- 如果用户 B 的密码也是 ‘"123456"\`,哈希值也完全一样。

这带来了两个致命弱点:

1.  <strong class="list-label">彩虹表攻击 (Rainbow Table Attack)</strong>:黑客可以预先计算出所有常见密码(如 123456, password, admin)的哈希值,存入巨型数据库(彩虹表)。一旦拿到数据库泄露的哈希值,直接查表就能反查出原始密码。

2.  <strong class="list-label">模式识别</strong>:如果数据库中两个用户的哈希值相同,黑客立即知道他们使用了相同的密码。

<strong class="critical-term">解决方案:</strong>

<strong class="key-term">Salt (盐)</strong> 是一个随机生成的字符串。在进行哈希计算之前,我们将 Salt 追加或拼接到密码上。 公式从 ‘Hash(Password)‘ 变为:Hash(Password + Salt)

<strong class="critical-term">关键设计原则:</strong>  

1.  <strong class="list-label">唯一性</strong>:<strong class="key-term">每个用户</strong>必须拥有独立的、唯一的盐。不可以使用全局通用的盐。

2.  <strong class="list-label">随机性</strong>:盐必须随机生成,不可预测。

3.  <strong class="list-label">公开性</strong>:盐<strong class="key-term">不需要</strong>保密。它通常明文存储在数据库中,与哈希值放在一起。它的作用是防止查表,而不是加密。

此外,还可以加<strong class="key-term">胡椒</strong>,Hash(Password + Salt + Pepper),它的特点是不在数据库里,即使Hash和Salt泄露了,没有胡椒也不会被破解。

<strong class="critical-term">bcrypt/Argon2算法</strong>

前面我们提到了为了防止黑客破解,我们可以加盐加胡椒,但如果人家就最原始的暴力破解,比如一遍遍穷举,那么也总有试出来的时候,<strong class="key-term">bcrypt/Argon2</strong>的思路就是把哈希计算的速度变慢<strong class="key-term">变慢</strong>,这样以来,你穷举的成本被大大提高,就可以间接保护密码不被泄露了。

### 10.5.2 语义去重

MinHash+LSH的模糊去重的原理是基于<strong class="key-term">字面结构</strong>的相似。它关注的是<strong class="key-term">字符或单词的重叠率</strong>。 而<strong class="key-term">语义去重</strong>:基于<strong class="key-term">潜在含义的相似</strong>。它关注的是<strong class="key-term">向量空间中的几何距离</strong>。 就是Transformer转为稠密向量那一套,语义去重的核心是将离散的文本转化为连续的稠密向量,利用<strong class="key-term">向量的方向一致性来判断语义相似度</strong>。

1.  <strong class="list-label">向量化</strong>

    使用预训练的语义模型(如<strong class="key-term">BERT, RoBERTa, BGE, text-embedding-ada-002</strong>)将每条文本映射为一个固定维度的向量(例如 768维 或 1536维)。

    > <strong class="list-label">本质</strong>:将语义信息压缩到向量数值中。

2.  <strong class="list-label">相似度矩阵计算</strong> 计算向量之间的<strong class="key-term">余弦相似度</strong>。对于 N 条数据,理论上需要计算 $N \times N$ 的矩阵。

    - <strong class="list-label">比如:</strong>

      - $V_A = [0.85, 0.12]$

      - $V_B = [0.84, 0.13]$

      - $\text{Cosine}(V_A, V_B) = \frac{V_A \cdot V_B}{\|V_A\| \|V_B\|} \approx 0.99$ 方向非常接近。

    - <strong class="list-label">优化:</strong>由于 $O(N^2)$ 复杂度过高,通常配合 <strong class="key-term">向量检索库 (FAISS)</strong> 或 <strong class="key-term">聚类算法 (K-Means/DBSCAN)</strong> 来缩小对比范围,仅计算同一个<strong class="key-term">簇(Cluster)内</strong>的数据。

      - 两种方式其实比较接近,都是希望找最接近的那一批再精细化比较,避开粗颗粒上就离得比较远的数据。

      - FAISS 的 IVF(<strong class="key-term">倒排索引</strong>)会预先训练出一组<strong class="key-term">虚拟的中心点</strong>,因此,有新的向量之后会先去计算距离哪个中心点比较近,之后就只去和这个中心点附近单元的数据比较,避免了遍历所有的数据。它的特点是它是动态的,<strong class="key-term">数据是源源不断进来的</strong>

      - 聚类的算法比较容易,就是一次性把所有的数据<strong class="key-term">分簇</strong>,只需要在同一簇内的向量进行相似度比较就行了,不同簇的相似度大概率不如同一簇内向量的相似度,不用比了。

3.  <strong class="list-label">阈值判定</strong> 设定一个较高的语义相似度阈值(通常 ≥0.90或 ≥0.95)。

    - 若 Similarity(A,B)\>Threshold,则判定 A 与 B 语义重复。

4.  <strong class="list-label">筛选</strong>

#### 10.5.2.1 简单例子

- <strong class="list-label">文本 A</strong>:"光速是多少？"

- <strong class="list-label">文本 B</strong>:"真空中的光传播速度数值。"

如果采用模糊去重,那么这两句话因为重复的文本很少,就会被认为是不相似。但实际上他们的语义及其相似。利用语义向量来计算余项相似度的话就能看到他们的方向及其接近。

## 10.6 数据处理综合实验

### 10.6.1 作业背景

第四章大作业的目标是<strong class="critical-term">对 CC WET 文件集合进行过滤处理，生成语言模型训练数据，并进行训练测试</strong>。

共计要 5000个 WET files `/data/CC/CC*.warc.wet.gz`

因为即便是 5000 个WET文件，大小也很大，压缩后的文本大小约为375GB，因为存储不太够了，我在实际训练中只用了 1500 个，最后也算达到效果了。

具体来讲的完整流程：

1.  先去 CC 上下载好对应的 `.warc.wet.gz` 类型数据

2.  得到原始数据后先去过滤一下，过滤的标准就是按照我们之前，比如邮件、脏话、广告什么的。

3.  过滤好的数据会 merge 成一份

4.  之后就是用 tokenzier 把文本数据 token 化，得到一个 bin 文件

5.  然后就可以在 cs336-basics 目录下进行最终训练了，把 `your_data.yaml` 配置改成对应的 train 和 val 的路径。

目标就是 Paloma 基准测试的 C4 100 领域子集(也就是测试集)损失最低。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读</p>

对于验证集 `valid_bin` 文件，讲义里是直接用斯坦福服务器下的 paloma.bin，我是在 <https://huggingface.co/datasets/allenai/paloma> paloma 官网手动下载了val 的数据，并亲自训练成了 bin

</div>

> <strong class="critical-term">注意不应修改模型架构或训练流程，因为目标是构建最优数据集。</strong>

### 10.6.2 实践操作

#### 10.6.2.1 数据集的获取

依旧无法访问斯坦福官网的集群，替代思路是直接从 <strong class="key-term">Common Crawl</strong> 上下载 <https://commoncrawl.org/overview>

##### 下载 WET 的黄金方案：CC-download

<https://github.com/commoncrawl/cc-downloader>

具体下载 wet_files 文件时，使用 commoncrawl 官方推荐的 WET文件下载工具 <strong class="key-term">cc-downloader</strong> 去下载 5000 个WET files，速度极快，远超 wget 等命令。我觉得也比讲义上推荐的 python 的多线程处理库方便，但如果要边下载边进行过滤处理的话，还是写 python 脚本吧。

<figure>
<img src="/images/707ec4c1f0.png" style="width:80.0%" alt="cc-downloader 下载 WET files 示意" />
<figcaption>cc-downloader 下载 WET files 示意</figcaption>
</figure>

在 CC官网上找到对应的 WET files 下载链接索引，然后就可以直接使用 cc-downloader 去下载了，要是想要精准控制5000个，那么可以先给压缩的WET索引文件解压了，改成5000个，之后再压缩成gz（因为cc-downloader要求格式就是gz）。

<figure>
<img src="/images/a747fa26a9.png" style="width:80.0%" alt="commoncrawl 官网 WET files 链接" />
<figcaption>commoncrawl 官网 WET files 链接</figcaption>
</figure>

#### 10.6.2.2 数据集的过滤处理

下载完成数据集之后，就是对它们进行过滤或者说清洗。

我们得到的 WET 文件是 Common Crawl 已经爬好并提取完纯文本的结果，里面会包含来源 URL 的主域名等 metadata。

清洗时第一步就是先判断这个域名可不可信，不要是什么黄色网站、赌博网站等不好的站点。判断的方式是建立一个白名单库或者黑名单库(这里我直接省略了)。

<div class="center">

<strong class="critical-term">输入 → Gopher 质量过滤 → 语言检测过滤 → NFSW 过滤 → 毒性内容过滤 → 质量分类过滤 →个人隐私脱敏(phonenumber email ip) → 输出</strong>

</div>

<figure>
<img src="/images/2b18fab314.png" style="width:80.0%" alt="完整实现流程" />
<figcaption>完整实现流程</figcaption>
</figure>

根据我的实践经验来看，由于文本长度不达标被过滤的内容最多，其次是内容质量。

<figure>
<img src="/images/bc716baf6d.png" style="width:80.0%" alt="内容过滤结果" />
<figcaption>内容过滤结果</figcaption>
</figure>

#### 10.6.2.3 分词处理

对于过滤后的文本内容，为了方便后续处理，我的操作是把它们 merge 成一个 txt 文件。

自此我们算是处理好了数据这一步。接下来就是 token 分词。

这一步老生常谈，就是把文本 token 化。

最后得到一个 bin 文件

#### 10.6.2.4 训练

将 config.yaml 和我们自己的 yaml 文件配置好之后，就可以直接调用讲义里提供的 gpt2 的脚本就可以(cs336-basics/scripts/train.py)开启训练了。

我使用的是 a100 单卡，具体使用了多少显存我记不得了。

#### 10.6.2.5 实验结果

<figure>
<img src="/images/04a7c7942e.png" style="width:80.0%" alt="训练实验结果" />
<figcaption>训练实验结果</figcaption>
</figure>

训练集持续下降，验证集到后期有点反弹，多少有点过拟合。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment4-data)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 10.1 · MinHash

<span id="read-11-1"></span>

<https://www.cnblogs.com/sddai/p/6110704.html>
