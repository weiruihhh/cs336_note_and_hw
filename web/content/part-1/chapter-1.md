---
outline: [2, 3]
---

# 第 1 章 · 分词器与 BPE

<span id="guide-ch-1"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong><strong class="critical-term">Python</strong> 基本语法, <strong class="critical-term">uv</strong> 基本使用方法

<strong>准备工作：</strong>最好是 <strong class="critical-term">Linux 系统</strong> 与带有 <strong class="critical-term">GPU</strong> 的环境(至少4060Ti), 下载 Assignment 1 仓库, 下载 TinyStories 与 OpenWebText 数据集。

<strong>本章任务：</strong>实现字节级 BPE 训练、特殊 token 处理、编码和解码；保存词表与 merges，并比较压缩率和吞吐量。

</div>

## 1.1 Byte-pair encoding (BPE) tokenizer

<span id="sec-1-2"></span>

[参考资料 1.1](/part-1/chapter-1#read-1-1)

[参考资料 1.2](/part-1/chapter-1#read-1-2)

整体实验目标: 实现 BPE 分词器。

对应测试文件路径: tests/test_tokenizer.py

### 1.1.1 Unicode字符集

在计算机发展的早期,不同国家和地区为了表示本国语言的字符,分别制定了自己的编码系统,例如: 美国的<strong class="key-term">ASCII</strong>,中国的<strong class="key-term">GB2312</strong>等。但这些编码系统互不兼容,就很不方便。

而Unicode(统一码) 是一种统一的字符集标准,为世界上所有文字和符号分配了一个唯一的编号(称为“码点” code point)。比如汉字“牛”的码点是“U+29275”,其中“U+”是一个无意义前缀,表示这是一个Unicode码点,“29275”表示这个码点的十进制值。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · Unicode字符集代码实操</p>

在python里,我们可以使用 <strong class="key-term">ord()</strong> 函数获取一个字符的码点,使用 <strong class="key-term">chr()</strong> 函数获取一个码点对应的字符。 比如:

</div>

```python
    ord('牛')
    >>> 29275
    chr(29275)
    >>> '牛'
```

Unicode定义了“<strong class="key-term">字符</strong>$\leftrightarrow$<strong class="key-term">码点</strong>”的映射,而UTF(Unicode Transformation Format) 则进一步定义如何把<strong class="key-term">码点</strong>编码为<strong class="key-term">字节</strong>。常见的UTF编码有UTF-8,UTF-16,UTF-32等。

#### 1.1.1.1 UTF-8编码

UTF-8 是一种<strong class="key-term">变长编码</strong>,使用 <strong class="key-term"> $1 \thicksim 4$ 个字节</strong>表示。 unicode 字符,是互联网的主导编码格式(占所有网页的98%以上)。

变长编码也就是说不同的字符编码的结果长度不一定相同,有的是 1 个字节,有的是2 个字节或 3、 4个字节。

UTF-8 编码和 Unicode 码点范围的对应关系:

| <strong>字节数</strong> | <strong>UTF-8 字节序列 (二进制)</strong> | <strong>Unicode 码点范围 (十六进制)</strong> | <strong>Unicode 码点范围 (十进制)</strong> |
|:--:|:--:|:--:|:--:|
| 1 | `0xxxxxxx` | U+0000 至 U+007F | 0~127 (7 bits) |
| 2 | `110xxxxx 10xxxxxx` | U+0080 至 U+07FF | 128~2047 (11 bits) |
| 3 | `1110xxxx 10xxxxxx 10xxxxxx` | U+0800 至 U+FFFF | 2048~65535 (16 bits) |
| 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` | U+10000 至 U+10FFFF | 65536~1114111 (21 bits) |

UTF-8编码规则对照表

所有存储的字节被分为两类:<strong class="key-term">领头字节</strong>(Leading Byte)和<strong class="key-term">后续字节</strong>(Continuation Byte)。

1.  <strong class="note-label">单字节字符 (ASCII范围)</strong>

    - <strong class="list-label">规则</strong>:如果一个字节的最高位(第1位)是“0”,那么它就是一个单字节字符。

    - <strong class="list-label">格式</strong>: 0xxxxxxx

    - <strong class="list-label">解读</strong>:1 字节时,UTF-8 完全等价于 ASCII:ASCII 0x41(A) $\leftrightarrow$ UTF-8 0x41 ,UTF-8 编码格式是 0xxxxxxx,正好能容纳 ASCII 0 127。

2.  <strong class="note-label">多字节字符</strong>

    - <strong class="list-label">规则</strong>:如果一个字节的开头是 “1”,那它就是一个多字节字符的其中一部分(<strong class="key-term">开头有几个连续的1就代表几个字节</strong>)。

    - <strong class="list-label">领头字节</strong>:

      - ‘110xxxxx‘:表示这是一个<strong class="key-term">双字节</strong>字符的第一个字节。

      - ‘1110xxxx‘:表示这是一个<strong class="key-term">三字节</strong>字符的第一个字节。

      - ‘11110xxx‘:表示这是一个<strong class="key-term">四字节</strong>字符的第一个字节。

    - <strong class="list-label">后续字节</strong>:

      - ‘10xxxxxx‘:所有非领头的后续字节,都必须以‘10‘开头。

<div class="custom-block tip">

<p class="custom-block-title">例子 · UTF-8编码示例</p>

汉字“中”的Unicode码点是U+4E2D。U+4E2D 在 U+0800 到 U+FFFF 之间,根据规则,它需要用3个字节来编码。

首先将U+4E2D转换为二进制:

$$
\begin{aligned}
        \text{4} &\rightarrow 0100 \nonumber \\
        \text{E} &\rightarrow 1110 \nonumber \\
        \text{2} &\rightarrow 0010 \nonumber \\
        \text{D} &\rightarrow 1101 \nonumber
    \end{aligned}
$$

所以,U+4E2D 的二进制是 0100 1110 0010 1101。(总共16位)

3字节的UTF-8模板是:1110xxxx 10xxxxxx 10xxxxxx,将二进制填入模板中,得到:11100100 10111000 10101101。

所以,汉字“中”的UTF-8编码是:11100100 10111000 10101101。

</div>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · Unicode编码实操</p>

在python里,对于string类型数据,我们可以使用encode()方法将字符串编码为UTF-8编码,使用decode()方法将UTF-8编码解码为字符串。 比如:

</div>

```python
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8"
    >>> b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
    list(utf8_encoded)
    >>> [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
    utf8_encoded.decode("utf-8")
    >>> "hello! こんにちは!"
```

通过UTF-8编码将Unicode码点转换为字节序列,我们本质上是在将码点序列(0到154997范围内的整数)转换为字节值序列(0到255范围内的整数)。256长度的字节词汇表处理起来要容易得多。使用字节级分词时,我们无需担心词汇表外的标记,因为我们知道<strong class="key-term">任何输入文本都可以表示为0到255的整数序列</strong>。

#### 1.1.1.2 overlong编码

UTF-8 要求用<strong class="key-term">最短的字节数</strong>编码每个字符。

比如U+002F(十进制下为47,十六进制下为0x2F,表示符号为“/”)按照规则只能用1个字节编码,但如果我们非要把它编码为2个字节(0xC0 0xAF),它解码出来:

- 0xC0 0xAF $\rightarrow$ 11000000 10101111

- 11000000 10101111 $\rightarrow$ 47 $\rightarrow$ U+002F

同样也是“/”,这就是<strong class="key-term">overlong编码</strong>。

overlong编码不守UTF8的最小字节编码的规矩,因此是被明确禁止的,如果采用overlong编码有可能绕过防火墙啥的,有很大<strong class="key-term">危险性</strong>。

#### 1.1.1.3 Unicode相关Problem

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>chr(0)返回什么Unicode字符？</strong>

答:返回空字符

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>这个字符的字符串表示(\_\_repr\_\_())与其打印表示有何不同？</strong>

答:它的打印表示通常是不可见的（没有视觉输出）,而其字符串表示 \_\_repr\_\_() 是一个明确的转义序列’\
x00’。

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>当这个字符出现在文本中会发生什么？</strong>

答:会出现一个截断,效果类似空格。

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>选择在UTF-8编码字节而非UTF-16或UTF-32上训练分词器的原因有哪些?</strong>

答:1. UTF-8 是一种变长编码,对英文、数字和常用符号等只用1个字节表示,而像汉字等字符通常用3个字节。相比之下,UTF-16 至少需要2个字节,UTF-32 固定为4个字节。对于以英文为主的语料库,UTF-8 的存储和处理效率远高于后两者

2\. 一个分词器的词汇表(Vocabulary)大小是有限的。如果直接在Unicode字符(UTF-32)上操作,遇到词汇表中没有的字符(例如一个新的 Emoji或一个罕见的汉字),就只能将其标记为<strong class="key-term">\<UNK\></strong>。 而基于UTF-8字节的分词器,其基础词汇表是固定的256个字节(从 0x00 到 0xFF)。任何未知的、罕见的字符,甚至是乱码,都可以被分解成一串已知的字节序列来表示。这样就从根本上消除了\<UNK\>符号,使得模型能够处理任何形式的文本输入,而不会丢失信息。

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>为什么如下函数是错误的？</strong>

```python
            def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
                return "".join([bytes([b]).decode("utf-8") for b in bytestring])
            >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
            'hello'
        
        
```

答:UTF-8 是一种变长编码,一个Unicode字符可能由1到4个字节组成。

对于标准的ASCII字符(如 ’h’, ’e’, ’l’, ’l’, ’o’),它们在UTF-8中确实只由单个字节表示,所以这个函数对纯ASCII字符串"hello".encode("utf-8")能够侥幸成功。

但是,对于任何非ASCII字符,需要用多个字节表示,比如汉字“中”,它的UTF-8编码是 $b'\backslash xe4 \backslash xb8 \backslash xad'$,由三个字节组成。 当 decode_utf8_bytes_to_str_wrong 函数处理 $b'\backslash xe4 \backslash xb8 \backslash xad'$ 时,它会： 取出第一个字节 $b'\backslash xe4'$,并尝试执行 $bytes([b'\backslash xe4']).decode("utf-8")$。 0xe4 (二进制 11100100) 是一个多字节字符的“起始字节”,它告诉解码器“后面还跟着2个字节”。单独解码它必然会失败,因为它的序列不完整。

此时,Python会抛出 UnicodeDecodeError 异常,因为遇到了一个不完整或无效的UTF-8序列。 <strong class="key-term">正确的解码方式必须在完整的字节序列上进行,而不是逐字节地割裂进行。</strong>

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>提供一个双字节序列,该序列无法解码为任何Unicode字符</strong>

答:一个双字节字符的起始字节的二进制格式必须是 110xxxxx 10xxxxxx,因此只要不满足这个格式,就不能解码。

</div>

<div class="custom-block tip">

<p class="custom-block-title">薇言大义</p>

Unicode尤其UTF8编码是后面训练BPE分词器的基础概念,内容实际上就是信息论的延伸,理解了会对很多方面都有帮助。

</div>

### 1.1.2 subword tokenizer(子词分词器)

[参考资料 1.3](/part-1/chapter-1#read-1-3)

<strong class="critical-term">核心知识:</strong><strong class="key-term">词级(word-level)分词器、字符级(character-level)分词器、字节级(byte-level)分词器、子词级(subword-level)分词器</strong>之间的区别和联系

#### 1.1.2.1 词级(word-level)分词器

- <strong class="note-label">核心思想:</strong> 最符合人类直觉的方式,直接将句子按照空格或标点符号切分成一个个独立的单词。对于中文等没有天然分隔符的语言,则需要依赖特定的分词算法(<strong class="key-term">如jieba分词</strong>)。

- <strong class="note-label">工作流程:</strong>

  1.  <strong class="list-label">预处理（可选）</strong>:可能包括小写化、去除标点、处理特殊字符等。

  2.  <strong class="list-label">切分</strong>:主要根据空格和标点符号将句子切分成单词。例如,"Hello, world!" 可能会被切分为\["Hello", ",", "world", "!"\]或\["Hello", "world"\](如果标点被移除或单独处理)。

  3.  <strong class="list-label">词汇表构建</strong>:

      - 在训练数据上统计所有出现的词语及其<strong class="key-term">频率</strong>。

      - 选择频率最高的 N 个词语构成词汇表 (vocabulary)。

      - 词汇表之外的词语（未登录词,<strong class="key-term">Out-Of-Vocabulary, OOV</strong>）通常会被映射到一个特殊的\<UNK\>(unknown) 标记。

  4.  <strong class="list-label">Token ID 映射</strong>:将每个词语映射到其在词汇表中的唯一整数 ID。

- <strong class="note-label">优点:</strong>语义完整,每个token都是一个有完整意义的词,非常直观。

- <strong class="note-label">缺点:</strong>

  - <strong class="list-label">词汇表巨大</strong>:需要为语言中几乎所有的词都创建一个条目,词汇表高达到几十万甚至上百万。

  - <strong class="list-label">OOV (Out-of-Vocabulary) 问题严重</strong>:当遇到一个词汇表中没有的词（如新词、拼写错误、专业术语）,分词器就无法处理,通常会将其替换为一个特殊的\<UNK\>(unknown)符号,导致信息丢失。例如,模型没见过"chatbot",就会将其视为\<UNK\>。

  - <strong class="list-label">无法处理词形变化</strong>:run、running、ran 会被视为三个完全不同的词,模型无法直接看出它们之间的关联,增加了学习负担。

#### 1.1.2.2 字符级(character-level)分词器

- <strong class="note-label">核心思想:</strong> 将文本拆分成一个个独立的字符。

- <strong class="note-label">示例:</strong>

  1.  <strong class="list-label">英文：</strong>"I am a pig" $\rightarrow$ \["I","a","m","a","p","i","g"\]

  2.  <strong class="list-label">中文:</strong> "我是一头猪" $\rightarrow$ \["我","是","一","头","猪"\]

- <strong class="note-label">优点:</strong>

  - <strong class="list-label">词汇表小</strong>:词汇表只包含所有基本字符(如a-z, A-Z, 0-9, 标点,中文字符等),大小非常可控。

  - <strong class="list-label">无OOV问题</strong>:任何单词都可以由字符组成,因此不存在未知词的问题。

- <strong class="note-label">缺点:</strong>

  - <strong class="list-label">序列过长</strong>:一个单词会被切成多个字符,过于琐碎导致输入序列的长度急剧增加,对模型的计算和内存都是巨大挑战。

  - <strong class="list-label">语义丢失</strong>:单个字符通常不具备独立的语义,模型需要从头学习如何将字符组合成有意义的词,学习效率非常低。

#### 1.1.2.3 字节级(byte-level)分词器

- <strong class="note-label">核心思想:</strong> 比字符级更底层的切分方式,它直接操作文本的原始字节(Bytes)。所有文本最终都以字节形式存储(如UTF-8编码),一个英文字母通常占1个字节,一个汉字可能占3个字节。

- <strong class="note-label">示例(UTF-8编码):</strong>

  1.  <strong class="list-label">英文：</strong>"cat" $\rightarrow$ \["c","a","t"\] $\rightarrow$ \[99, 97, 116\]

  2.  <strong class="list-label">中文:</strong> "猫" $\rightarrow$ \[227, 149, 131\](三个字节共同表示一个“猫”字)

- <strong class="note-label">优点:</strong>

  - <strong class="list-label">词汇表小且固定</strong>:字节的取值范围永远是0-255,所以词汇表大小固定为256。

  - <strong class="list-label">无OOV问题</strong>:任何文本都可以由字节组成,因此不存在未知词的问题。

- <strong class="note-label">缺点:</strong>

  - <strong class="list-label">序列更长</strong>:比字符级分词器更长,因为一个字符可能由多个字节组成。

  - <strong class="list-label">语义几乎破碎</strong>:模型需要学习从毫无关联的字节序列中重构语义,学习难度极大。

#### 1.1.2.4 子词级(subword-level)分词器

- <strong class="note-label">核心思想:</strong> 目前LLM(如GPT、BERT系列)的标配,介于字符级和词级之间,它将单词拆分成更小的子词(Subword)。核心思想是：<strong class="key-term">高频词汇作为一个整体保留,低频词汇或未见过的词则拆分为更小的、有意义的子词单元。</strong>

  我们要实现的BPE分词器就是一种子词级分词器。

- <strong class="note-label">示例:</strong>

  1.  "the" $\rightarrow$ \["the"\] 由于"the"高频出现,所以作为一个整体保留。

  2.  "wonderful" $\rightarrow$ \["wonder","ful"\] 由于"wonderful"低频出现,所以拆分为"wonder"和"ful"两个子词。

- <strong class="note-label">优点:</strong>

  - <strong class="list-label">平衡了词汇量和序列长度</strong>:词汇表大小适中(通常3万-10万),序列长度也比字符/字节级短得多。

  - <strong class="list-label">有效处理OOV问题</strong>:任何新词都可以由已知的子词组合而成,例如模型不认识"webinar",但可能认识"web"和"inar",可以将其切分为\["web", "inar"\],从而理解其含义。

  - <strong class="list-label">更灵活的词形变化处理</strong>:例如"laughing"和"laughed"都可以被拆分为\["laugh","ing"\]和\["laugh","ed"\],模型能轻易捕捉到"laugh"这个共同的词根,理解不同词形间的关系。

- <strong class="note-label">缺点:</strong>

  - <strong class="list-label">词汇表大小不易控制</strong>:需要手动调整合并频率来平衡词汇量和模型效果,这在实际应用中可能比较麻烦。

  - <strong class="list-label">训练成本较高</strong>。

| <strong>类型</strong> | <strong>单位</strong> | <strong>优点</strong> | <strong>缺点</strong> | <strong>举例</strong> |
|:---|:---|:---|:---|:---|
| <strong>词级(word)</strong> | 词 | 语义清晰 | 词表极大,OOV | “I love NLP” $\rightarrow$ \["I", "love", "NLP"\] |
| <strong>字符级(character)</strong> | 字符 | 词表极小,无未知词 | 语义太碎,序列太长 | “I love” $\rightarrow$ \["I", " ", "l", "o", "v", "e"\] |
| <strong>字节级(byte)</strong> | 字节 | 能统一多语言、符号 | 可读性差,序列长 | “Hi” $\rightarrow$ \[72, 105\](ASCII码) |
| <strong>子词级(subword)</strong> | 词根词缀 | 权衡两者,处理新词 | 稍复杂,需要训练 | “unbelievable” $\rightarrow$ \["un", "believ", "able"\] |

## 1.2 BPE Tokenizer Training

<span id="sec-1-3"></span>

[参考资料 1.4](/part-1/chapter-1#read-1-4)

整个BPE分词器训练过程可以分为以下三个步骤:

1.  <strong class="critical-term">初始化词汇表</strong>

2.  <strong class="critical-term">预分词</strong>

3.  <strong class="critical-term">迭代合并</strong>

### 1.2.1 初始化词汇表

之前说到,BPE分词器是<strong class="key-term">子词级分词器</strong>,它的词汇表是由子词组成的。对应初始的词汇表等价于字节级的,也就是<strong class="key-term">固定为256个字节</strong>。

BPE算法后续的每一步都是“<strong class="key-term">找到频率最高的相邻符号对并合并</strong>”,这会逐步改变我们初始的词汇表。也就是说原始词汇表可能是:(a:0x01,b:0x02,c:0x03,...),经过一轮合并之后可能就变成(a:0x01,b:0x02,c:0x03,...,ab:0x257)。即基础256个字节加上后面合并的子词。

最终这个词汇表里的每一个条目,也就是我们常说的<strong class="key-term">token</strong>,都会对应一个唯一的整数ID。

### 1.2.2 预分词

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>为什么需要预处理分词？</strong>

按理说,有了初始词汇表,就可以遍历语料库,把最频繁出现的字节对开始合并,形成更大的 token 了。但这样做有两个缺点：

1.  语料库遍历很费计算。

2.  如果没有预分词,BPE算法会直接处理一整串字符,比如 "goes."。它可能会发现 ’s’ 和 ’.’ 在语料中经常一起出现（例如在很多句末）,于是把它俩合并成一个新的token ’s.’。这显然是不合理的,因为它混淆了单词本身（go的第三人称单数形式）和句法结构（句号）。

预分词的作用就是先进行一次清晰的切分,告诉BPE：“goes是一个独立的单元,.是另一个独立的单元,你可以在goes内部进行合并,但绝对不能把goes的尾巴和 . 合并在一起。”

</div>

<strong class="list-label">预分词的做法：</strong>

预分词像是一种<strong class="key-term">对词汇表粗颗粒度的分词</strong>,常用的预分词方法是使用<strong class="key-term">正则表达式(Regular Expression)</strong>来切分文本。例如,GPT2论文中使用的是:

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

<strong class="note-label">基础速查：</strong>正则表达式的概念、符号表及 Python 使用示例见[附录：正则表达式与文本匹配](/appendix#app-regex)。

### 1.2.3 BPE合并

等到预分词进行粗颗粒度划分之后,每一个划分后的部分再转换成UTF-8序列,就可以开始BPE合并(即训练BPE分词器)了。比如原始输入文本是“I am a pig",预分词后得到\[“I”, “am”, “a”, “pig”\],其中每一部分再转换成UTF-8序列得到

- “I \</w\>”: 1,

- “a m \</w\>”: 1,

- “a \</w\>”: 1,

- “p i g \</w\>”: 1

(注意这里的“ \</w\>”是特殊符号,表示一个词的结束, 1 表示频次),然后开始BPE合并。

从高层次来看,BPE算法会迭代统计每个字节对,并识别出现<strong class="key-term">频率最高的字节对</strong>（“A”, “B”）。然后将这个最频繁出现的字节对（“A”,“B”）的所有实例进行合并,即替换为一个新标记“AB”。这个新合并的标记会被添加到我们的词汇表中；因此,BPE训练后的最终词汇表大小等于初始词汇表（在我们的案例中是256个）,加上训练过程中执行的BPE合并操作次数。

BPE在合并的时候不考虑跨边界合并。例：若预处理将“dog!"和“dog.”切分为两个独立标记,则“dog!”中的 “g” 与 “!” 可合并,但 “dog!” 的 “!” 与 “dog.”的 “d” 不会被统计（因属于不同标记,会有\</w\>符号将其分开）。

BPE在频率相同时,采用选择<strong class="key-term">字典序更大的对优先的原则</strong>。例如,若字节对（“A”,“B”）、（“A”,“C”）、（“B”,“ZZ”）和（“BA”,“A”）的频率均为最高,则我们会选择合并（“BA”,“A”）。

字典序一般就是<strong class="key-term">位数多的字符排在位数低的后面,位数相同就按照英文单词顺序排</strong>,比如“A”在“B”之前,“B”在“C”之前。

### 1.2.4 特殊标记

在文本编码过程中,经常会使用特定字符串（如\<\|endoftext\|\>）来存储元数据（例如文档间的分界标记）。进行编码时,通常需要将某些字符串视为"特殊标记"即表示这些标记<strong class="key-term">永远不应被拆分为多个子标记（即始终作为独立标记保留）</strong>。比如说,序列终止符\<\|endoftext\|\>必须始终作为独立标记（对应单一整数ID）存在,以便语言模型知晓何时停止生成内容。这类特殊标记必须一开始就被加入词汇表,从而获得对应的固定标记ID。

### 1.2.5 BPE分词器训练示例

1.  原始文本(末尾包含\<\|endoftext\|\>):

    ```python
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    ```

2.  <strong class="list-label">初始化词汇表</strong>: 256个固定字节以及特殊标记\<\|endoftext\|\>

3.  <strong class="list-label">预分词</strong>: 按照空格划分结果–\>{low: 5, lower: 2, widest: 3, newest: 6}

4.  <strong class="list-label">BPE合并</strong>: 首先把预分词的结果再去划分一下–\>{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,s,t): 3, (n,e,w,e,s,t): 6};然后不断迭代合并统计频率最高的字节对,得到新的词汇表。第一轮频率统计:{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6} ,(’es’)和(’st’)并列频率最高，按照字典序最大原则选择(’st’)作为这一轮合并的token添加到字典里，然后继续下一轮，以此类推。

5.  一般来讲，在大型的BPE合并时，<strong class="key-term">训练终止的条件是字词大小到达某个预设值</strong>，比如预设5000，我们初始词典大小时256(假设无特殊标记)，那么合并4744次之后，词汇表大小达到5000，训练终止。

## 1.3 BPE分词器训练实操

[参考资料 1.5](/part-1/chapter-1#read-1-5)

接下来要实现在TinyStory数据集上训练BPE分词器。(TinyStory数据集可从github上下载)

我们之前说了像\<\|endoftext\|\>这样的特殊标记必须一开始就被加入词汇表,从而获得对应的固定标记ID。在预分词的时候也要特殊对待一下特殊符号，一般就会把特殊符号也作为一个分隔符。

我们之前所说的BPE训练的办法是基本、朴素的原理，但在实际操作中这种方法效率会比较低，速度较慢(我一开始用的就是这种朴素的算法，可以说非常慢了)，合适的办法是<strong class="key-term">直接记录下来所有词的频率到计数器中去，每次merge只需要把对应的计数器内容给改了就好</strong>。虽然都是两重循环，但第二种办法只用该计数器，它避免了对重复内容的重复处理，因此会快不少。 举个例子:

- <strong class="note-label">第一种方法（低效）</strong>：

  1.  拿到班级花名册，上面写着每个学生的姓名

  2.  <strong class="list-label">花名册上有：</strong>"张三、李四、张三、王五、张三、李四、张三..."（总共1000个名字）

  3.  <strong class="list-label">你一个一个地数：</strong>张三、李四、张三、王五、张三、李四...

  4.  每个名字都要处理一遍，即使是重复的

- <strong class="note-label">第二种方法（高效）</strong>：

  1.  <strong class="list-label">你先把花名册整理成：</strong>张三(500次)、李四(300次)、王五(200次)

  2.  <strong class="list-label">然后直接统计：</strong>张三出现500次，李四出现300次，王五出现200次

  3.  只需要处理3个唯一的名字

另外一些实践技巧: 利用 <strong class="key-term">cProfile 或 scalene 等工具</strong>可以来帮我们分析代码中的瓶颈;在直接测试TinyStory数据集时,可以先测试小部分数据快速检验效果，避免因为数据量太大而浪费时间。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · cProfile 和 scalene</p>

cProfile 是 Python 内置的<strong class="key-term">性能分析（Profiler）模块</strong>，用于测量程序运行过程中<strong class="key-term">各个函数的执行时间、调用次数等性能数据</strong>，帮助开发者定位程序中的性能瓶颈。cProfile 会在程序运行时记录每个函数被调用的次数、每次调用消耗的时间以及总耗时，最终生成一份详细的统计报告。

基本使用方法: python -m cProfile test.py

cProfile 会输出文本报告统计，保存到result.prof,也可以结合SnakeViz来查看可视化报告，即snakeviz result.prof

Scalene 是一个 高精度的 Python 性能分析器（profiler），比 cProfile 更先进。它不仅能分析CPU时间，还能同时分析： <strong class="key-term">CPU 使用（Python 与本地代码分离）</strong>、<strong class="key-term">内存使用（包括分配与释放）</strong>、<strong class="key-term">GPU 使用（可选）</strong>、<strong class="key-term">行级别的性能分析结果</strong>

可以在命令行中直接使用查看文本报告: scalene test.py

也可以生成网页可视化报告，即 scalene –html example.py

</div>

## 1.4 利用BPE分词器进行编码和解码

[参考资料 1.6](/part-1/chapter-1#read-1-6)

我们训练 BPE tokenizers 的最终目的还是希望它对新的文本进行编解码，以便后续的操作。

### 1.4.1 编码

目标是利用我们已经训练好的BPE分词器将一个原始的文本字符串转换成一个整数Token ID列表。具体实现步骤和训练BPE分词器的过程类似，但是不需要进行BPE合并。

1.  <strong class="critical-term">处理特殊符号</strong>：

    1.  <strong class="note-label">最优先处理</strong>：在进行任何其他操作之前，你需要先将文本中的特殊符号替换为它们对应的占位符或直接分割出来。

    2.  <strong class="note-label">策略：</strong>

        - 如果特殊符号在 vocab 中已经有预定义的 ID（通常是在训练BPE之前就加入的），可以用一个独特的、不会在普通文本中出现的字节序列来临时替换它们。或者先用特殊符号将整个文本分割成多个部分。例如，如果文本是 "Hello \<\|endoftext\|\> World"，你可以先将其分割成 \["Hello ", "\<\|endoftext\|\>", " World"\]。

        - 对于每个非特殊符号的文本块，进行下面的预分词和合并。

        - 对于特殊符号块，直接查找其在 vocab 中的 ID。

2.  <strong class="critical-term">Pre-tokenize (预分词)</strong>：

    1.  <strong class="note-label">目的：</strong>将文本分割成一些“词块”（word chunks）。BPE合并只在这些词块内部进行，不会跨越词块边界。这通常是为了防止合并无意义的字符组合（比如一个词的末尾和下一个词的开头）。

    2.  <strong class="note-label">方法：</strong>使用与BPE训练时相同的正则表达式。这个正则表达式通常会根据空格、标点符号等来切分文本。

    3.  <strong class="note-label">输出：</strong>一个字符串列表，每个字符串是一个预分词块。例如，"Hello world!" 可能被预分词为 \["Hello", " world", "!"\] (注意空格可能被归属到某个块)。

3.  <strong class="critical-term">Apply the merges (应用合并规则)</strong>：

    1.  <strong class="note-label">对每一个预分词块单独执行以下操作：</strong>

        - <strong class="list-label">转换为字节序列列表：</strong>将预分词块（字符串）编码为 UTF-8 字节序列，然后将这个字节序列拆分成单个字节的列表。例如，"the" -\> b’the’ -\> \[b’t’, b’h’, b’e’\]。

        - <strong class="list-label">迭代应用合并规则：</strong>

          - 遍历merges列表中的每一条合并规则 (pair_A, pair_B)。

          - 在当前的字节序列列表中，查找所有连续出现的 (pair_A, pair_B)。

          - 将找到的第一个（或所有，取决于实现策略，但通常是迭代地、贪婪地合并最先出现的） (pair_A, pair_B) 替换为它们合并后的新字节序列 pair_A + pair_B。 (<strong class="critical-term">重要：</strong> 每应用一次合并，字节序列列表的结构就可能发生变化。你需要重新从merges列表的开头开始检查，或者更高效地只检查与新合并的token相关的可能合并。)

          - 例如，当前序列是 \[b’t’, b’h’, b’e’\]，merges中有 (b’t’, b’h’)。应用后变成 \[b’th’, b’e’\]。然后假设merges中还有 (b’th’, b’e’)，应用后变成 \[b’the’\]。

        - <strong class="list-label">查找Token ID：</strong>当一个预分词块不能再进行任何合并时，它内部的每个（可能是合并后的）字节序列都应该对应vocab中的一个Token ID。将这些字节序列转换为它们的ID。

    2.  <strong class="note-label">拼接结果：</strong>将所有预分词块（以及特殊符号）得到的Token ID列表按顺序拼接起来，得到最终的编码结果。

<div class="custom-block tip">

<p class="custom-block-title">例子 · 举例编码过程</p>

- <strong class="critical-term">vocab:</strong> 0: b“ ”, 1: b“a”, 2:b“c”, 3: b“e”, 4: b“h”, 5: b“t”, 6: b“th”, 7: b“ c”, 8: b“ a”, 9:b“the”, 10: b“ at”

- <strong class="critical-term">merges:</strong> \[(b“t”, b“h”), (b“ ”, b“c”), (b“ ”, b“a”), (b“th”, b“e”), (b“ a”, b“t”)\]

- special_tokens: (假设没有)

处理过程:

1.  <strong class="critical-term">Pre-tokenize:</strong> \[“the”, “ cat”, “ ate”\] (注意空格的归属)

2.  <strong class="critical-term">处理 “the”:</strong>

    - <strong class="list-label">初始字节:</strong> \[b“t”, b“h”, b“e”\]

    - 遍历 merges: (b“t”, b“h”) 可应用 -\> \[b“th”, b“e”\]

    - 从头遍历 merges (对于 \[b“th”, b“e”\]): (b“th”, b“e”) 可应用 -\> \[b“the”\]

    - 从头遍历 merges (对于 \[b“the”\]): 没有可应用的。

    - <strong class="list-label">查找ID:</strong> b“the” -\> 9. 结果: \[9\]

3.  <strong class="critical-term">处理 “ cat”:</strong> (假设预分词包含前导空格)

    - <strong class="list-label">初始字节:</strong> \[b“ ”, b“c”, b“a”, b“t”\] (UTF-8编码的空格字节，这里用b“ ”示意)

    - 遍历 merges: (b“ ”, b“c”) 可应用 -\> \[b“ c”, b“a”, b“t”\]

    - 从头遍历 merges (对于 \[b“ c”, b“a”, b“t”\]): 根据讲义结果 \[7, 1, 5\]，它实际上是： b“ c” -\> 7 b“a” -\> 1 b“t” -\> 5 这意味着在 \[b“ c”, b“a”, b“t”\] 状态下，没有进一步的合并可以应用了，或者 (b“a”, b“t”) 这个合并规则不存在或顺序靠后。或者，更可能的是，“ cat” 预分词结果是 \[“ ”, “cat”\]，然后 "cat" -\> \[b“c”, b“a”, b“t”\]，没有合并，直接查ID得 \[2,1,5\]。但讲义结果是\[7,1,5\]，对应b“ c”, b“a”, b“t”。这暗示了“ cat”的预分词结果就是“ cat”，然后它被字节化为\[b“ ”, b“c”, b“a”, b“t”\]。第一个合并是 (b“ ”, b“c”) -\> b“ c” (ID 7)。剩下 \[b“a”, b“t”\]。这两个不能再合并，所以分别是 b“a” (ID 1) 和 b“t” (ID 5)。

4.  <strong class="critical-term">处理 “ ate”:</strong> (假设预分词包含前导空格)

    - <strong class="list-label">初始字节:</strong> \[b“ ”, b“a”, b“t”, b“e”\]

    - 遍历 merges: (b“ ”, b“a”) 可应用 -\> \[b“ a”, b“t”, b“e”\]

    - 从头遍历 merges (对于 \[b“ a”, b“t”, b“e”\]): (b“a”, b“t”) (这里指b“ a” 和 b“t”合并) 可应用 -\> \[b“ at”, b“e”\] (注意b“The at”是ID 10)

    - 从头遍历 merges (对于 \[b“ at”, b“e”\]): 没有可应用的。

    - <strong class="list-label">查找ID:</strong> b“ at” -\> 10, b“e” -\> 3. 结果: \[10, 3\]

5.  <strong class="critical-term">最终结果:</strong> \[9\] + \[7, 1, 5\] + \[10, 3\] = \[9, 7, 1, 5, 10, 3\]。

</div>

### 1.4.2 解码

目标是将一个整数Token ID列表转换回原始的文本字符串。

详细步骤：

1.  <strong class="critical-term">ID to Bytes (ID转字节序列)</strong>:

    - 遍历输入的Token ID列表。

    - 对于每个ID，使用vocab查找其对应的字节序列。

    - 将所有查找到的字节序列拼接起来，形成一个单一的字节串。

    - 例如，\[9, 7, 1, 5, 10, 3\] -\> b“the” + b“ c” + b“a” + b“t” + b“ at” + b“e” -\> b“the cat ate”。

2.  <strong class="critical-term">Bytes to String (字节串转字符串)</strong>:

    - 使用UTF-8解码器将拼接后的字节串转换回Unicode字符串。

    - <strong class="list-label">errors=’replace’</strong>： 这很关键。如果解码过程中遇到无效的UTF-8字节序列（比如用户提供了一个非法的ID序列，或者你的vocab中存在一些不能组成有效UTF-8的字节片段），errors=’replace’会用Unicode的替换字符 U+FFFD 来代替这些无效部分，而不是抛出UnicodeDecodeError。

## 1.5 BPE讲义相关Problems

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>从TinyStories和OpenWebText中各抽取10份文档样本。使用您先前训练的TinyStories和OpenWebText分词器（词汇表大小分别为1万和3.2万），将这些抽样文档编码为整数ID。这两个分词器的压缩比（字节/标记）分别是多少？</strong>

答:对每个文档：计算原始 UTF-8 编码下的字节数。用对应的 tokenizer 编码成 token ID 序列，并统计 token 数。最后可以得到平均压缩率：bytes/token = 总字节数/总 token 数

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>如果用TinyStories分词器处理OpenWebText样本会发生什么？比较压缩率和/或定性描述产生的结果。</strong>

答:如果用TinyStories分词器处理OpenWebText样本压缩比会降低很多。原因一是因为TinyStories分词器的词汇表大小只有1万，而OpenWebText分词器的词汇表大小有3.2万，能表示的token会少很多。二是TinyStories tokenizer是在简单的儿童故事上训练的，而OpenWebText包含各种网络文本（新闻、技术文章、论坛讨论等）。当tokenizer遇到训练时未见过的词汇模式时，会将其分解成更多的小片段，降低压缩效率。

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>估算你的分词器吞吐量（例如以字节/秒为单位）。处理Pile数据集（825GB文本）需要多长时间？</strong>

答:这个在不同的分词器的实现方式和硬件条件下吞吐量也有很大不同。

</div>

<div class="custom-block info">

<p class="custom-block-title">想一想</p>

<strong>使用您的TinyStories和OpenWebText分词器，将相应的训练集和开发集编码为整数标记ID序列。我们稍后将用此来训练语言模型。建议将标记ID序列化为uint16数据类型的NumPy数组。为何uint16是合适的选择？</strong>

答:uint16范围：$0 ~ 65,535（2^{16} - 1）$，选择uint16是因为它可以表示0到65,535的整数范围，足够覆盖10K和32K词汇量的所有token ID，同时相比uint32节省了一半的存储空间。

</div>

## 1.6 BPE章节实验

1.  train_bpe.py:编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE分词器。您的BPE训练函数应至少处理以下输入参数：input_path(输入文本文件路径,这里是TinyStories和OpenWebText数据集), vocab_size(词汇表大小), special_tokens(特殊符号列表),vocab(分词器词汇表，一个从整型（词汇表中的标记ID）到字节（标记字节）的映射关系。),merges(训练生成的BPE合并操作列表。每个列表项为一个字节元组(\<token1\>, \<token2\>)，表示\<token1\>与\<token2\>进行了合并。这些合并操作应按创建顺序排列。)

    uv run pytest tests/test_train_bpe.py 来运行测试文件。

2.  tokenizer.py :实现一个分词器类，该分词器在给定词汇表和合并规则列表的情况下，能够将文本编码为整数ID，并将整数ID解码回文本。该分词器还应支持用户提供的特殊标记（若这些标记尚未存在于词汇表中，则将其追加至词汇表）。

    uv run pytest tests/test_tokenizer.py 来运行测试文件。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 1.1 · 作业总览

<span id="read-1-1"></span>

- <strong class="list-label">uv官方文档:</strong><https://docs.astral.sh/uv/>

- <strong class="list-label">uv中文基础使用教程:</strong> <https://www.runoob.com/python3/uv-tutorial.html>

- <strong class="list-label">服务器租借平台:</strong> <https://www.autodl.com/>(平台不止这一个,我感觉这个比较方便)

### 1.2 · Byte-pair encoding (BPE) tokenizer

<span id="read-1-2"></span>

UTF-8编码详解-CSDN博客: <https://blog.csdn.net/whahu1989/article/details/118314154>

UTF-8编码原理及与ASCII的兼容性-CSDN博客: <https://blog.csdn.net/baidu_25299117/article/details/139633315>

### 1.3 · subword tokenizer(子词分词器)

<span id="read-1-3"></span>

jieba分词: <https://blog.csdn.net/qq_33957603/article/details/124640588>

### 1.4 · BPE Tokenizer Training

<span id="read-1-4"></span>

<strong class="note-label">基础资料：</strong>正则表达式教程已随相关介绍移至[附录](/appendix#app-regex)。

### 1.5 · BPE分词器训练实操

<span id="read-1-5"></span>

cProfile 相关知识: <https://docs.python.org/3/library/profile.html> Scalene 相关知识: <https://github.com/plasma-umass/scalene>

### 1.6 · 利用BPE分词器进行编码和解码

<span id="read-1-6"></span>

BPE分词器编码和解码相关知识: <https://github.com/huggingface/tokenizers>
