---
outline: [2, 3]
---

# 第 8 章 · 分布式训练与并行策略

<span id="guide-ch-9"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>理解[训练循环](/part-1/chapter-4#guide-ch-4)、[性能分析](/part-2/chapter-6#guide-ch-7)和进程概念；批量口径见[实验复现](/appendix#app-repro)；通信基础见[CUDA 通信机制知识补充](/appendix#app-gpu-cuda)。

<strong>准备工作：</strong>先确认单进程基线正确，再准备多进程启动配置；多卡实验记录 rank、world size、设备及通信后端。

<strong>本章任务：</strong>理解 AllReduce 与 DDP；完成正文涉及的梯度同步和优化器状态分片；比较 DP、TP、PP、ZeRO 的适用条件。

</div>

[参考资料 8.1](/part-2/chapter-8#read-9-1)

cs336第二篇的主旋律是优化已经实现的训练模型程序的性能；在第一节里我们重点放在了程序本身的优化,包括如何与硬件协同。

第二节我们优化性能的方式从直观理解上看更加简单粗暴:

我一个人做一件任务要花很长时间,那么两个人并行分担任务,那时间自然就降下来了(但不见得时间就一定变为原来的一半)。这到我们训练的场景上就是原先用一块GPU训练,现在用很多块GPU一起训练以实现效率提升,即<strong class="key-term">分布式并行(distribute parallel)</strong>。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 补充知识:线程(Thread)和进程(Process)的比较</p>

| <strong>比较项</strong> | <strong>进程 (Process)</strong> | <strong>线程 (Thread)</strong> |
|:---|:---|:---|
| 定义 | 操作系统下分配资源的基本单位 | 进程下分配资源的基本单位,是<strong>CPU调度和执行</strong>的基本单位 |
| 比喻 | 工厂 | 工人 |
| 资源分配 | 拥有独立的内存地址空间、文件句柄、IO设备 | 共享所属进程的绝大部分资源 |
| 数据共享 | 进程之间数据共享困难 | 线程间通信非常高效 |
| 创建开销 | 创建新进程的资源开销较大 | 创建新线程的开销较小 |
| 稳定性 | 相互独立,一个进程崩溃不影响其他进程 | 一个线程崩溃会影响整个进程及其他线程 |
| 依赖关系 | 进程包含线程,至少包含一个线程才能执行 | 线程一定属于某个进程,无法独立存在 |

</div>

## 8.1 单机多进程

<span id="sec-9-1"></span>

顾名思义,单机多进程是在一个物理服务器(节点)上启动多个独立的操作系统进程,共同协作完成一个计算任务。在现在主流的深度学习训练中,一般就是每个进程独立的控制一块GPU(比如DDP),但也有一个进程控制多个GPU的情况(比如Model parallelism、Pipeline Parallelism)。

单机多进程是实现数据并行(DP,Data Parallelism)的前提,其核心思想是<strong class="key-term">“模型复制,数据分片”</strong>,每个进程独立完成各自的任务,再通过<strong class="critical-term">集合通信(Collective Communication)</strong>来同步梯度,加速训练。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么不采用多线程？</p>

规避Python的全局解释器锁(GIL):Python的GIL机制规定,在同一进程中,任意时刻只允许一个线程执行Python字节码。这使得Python的多线程在计算密集型任务中无法实现真正的并行。而多进程模型中,每个进程都有自己独立的Python解释器和内存空间,因此不受GIL的限制,能够实现真正的并行计算。

</div>

下面以讲义上所说的一个简单的单机多进程代码作为示例

这个示例是在CPU上进行多进程,代码的目标是启动4个工作进程,每个进程都生成一个随机整数张量,然后通过一个名为<strong class="key-term">all-reduce</strong>的分布式操作,将这4个张量相加,并将最终的总和结果广播回每个进程。

1.  开局指定IP地址和端口号,指定通信方式是Gloo(Gloo既可用于CPU也可用于GPU通信,但GPU通信性能不如<strong class="key-term">Nccl</strong>)

2.  每个进程创建一个张量,进行如下操作:

    1.  <strong class="list-label">Reduce (规约)</strong>:从所有进程收集data张量。

    2.  <strong class="list-label">Op (操作)</strong>:对收集到的所有张量执行指定的操作,这里是SUM(求和)。

    3.  <strong class="list-label">All</strong> :将计算出的最终结果(总和)分发回所有的进程,并用该结果覆盖它们各自原来的data。

<figure data-latex-placement="H">
<img src="/images/848b15ceec.png" style="width:80.0%" alt="各种Rank之间的关系" />
<figcaption>各种Rank之间的关系</figcaption>
</figure>

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 区分Node Rank、Local Rank 和 Global Rank</p>

<strong class="critical-term">场景描述</strong>:一个分布式应用运行在2台机器(MACHINE 1 和 MACHINE 2)上,world_size为8(因为GLOBAL RANK从0到7)。

- <strong class="list-label">Node Rank (节点排名)</strong>:NODE RANK 为0和1,代表这是第0号和第1号机器。

- <strong class="list-label">Local Rank (本地排名)</strong>:每台机器上都运行了4个进程,因此在各自的机器内部,它们的LOCAL RANK都是从0到3。

- <strong class="list-label">Global Rank (全局排名)</strong>:GLOBAL RANK是全局唯一的。MACHINE 1上的4个进程拥有全局排名0, 1, 2, 3；而MACHINE 2上的4个进程则接着拥有全局排名4, 5, 6, 7。

</div>

## 8.2 AllReduce算法

<span id="sec-9-2"></span>

[参考资料 8.2](/part-2/chapter-8#read-9-2)

我们理解计算机的算法都是基于一个一个函数操作组合在一起得到的,那么我们在讲解分布式算法之前,我们必须先了解一下组成这种算法所应用于硬件的函数操作——集合通信的基本概念,

<strong class="list-label">Broadcast(广播)</strong>:将根服务器(Root Rank)上的数据分发广播给所有其他服务器(Rank)

<figure data-latex-placement="H">
<img src="/images/55ec353498.png" style="width:50.0%" alt="Broadcast" />
<figcaption>Broadcast</figcaption>
</figure>

<figure data-latex-placement="H">
<img src="/images/bdff2f9112.png" style="width:50.0%" alt="Broadcast" />
<figcaption>Broadcast</figcaption>
</figure>

如图所示,当一台服务器计算完成了自己部分的参数数据,在分布式训练中想要把自己这部分数据同时发送给其他所有服务器,那么这种操作方式就叫做广播(broadcast)。

<strong class="list-label">Scatter(散射)</strong>:将根服务器上的数据散射为<strong class="key-term">同等大小的数据块</strong>,每一个其他服务器得到一个数据块

<figure data-latex-placement="H">
<img src="/images/5da5c6217a.png" style="width:50.0%" alt="Scatter" />
<figcaption>Scatter</figcaption>
</figure>

如图所示,当一台服务器计算完成自己部分的参数数据,但是因为有时候服务器上全部的参数数据过大,于是我们想要把这台服务器上的数据切分成几个同等大小的数据块(buffer),再按照序列(rank index)向其他服务器发送其中的一个数据块,这就叫做散射(Scatter)。

<strong class="list-label">Gather(聚集)</strong>:将其他服务器上的数据块直接<strong class="key-term">拼接到一起</strong>,根服务器(Root Rank)获取这些数据

<figure data-latex-placement="H">
<img src="/images/5da5c6217a.png" style="width:50.0%" alt="Gather" />
<figcaption>Gather</figcaption>
</figure>

如图所示,当服务器都做了散射之后,每个服务器获得了其他服务器的一个数据块,我们将一台服务器获得的数据块拼接在一起的操作就叫做聚集(Gather)。

<strong class="list-label">AllGather(全聚集)</strong>:所有的服务器都做上述Gather的操作,于是所有服务器都获得了全部服务器上的数据

<figure data-latex-placement="H">
<img src="/images/cb5b997061.png" style="width:50.0%" alt="AllGather" />
<figcaption>AllGather</figcaption>
</figure>

如图所示,所有的服务器都将自己收到的数据块拼接在一起(都做聚集的操作),那么就是全聚集(AllGather)。

<strong class="list-label">Reduce(规约)</strong>:对所有服务器上的数据做一个规约操作(<strong class="key-term">如最大值、求和</strong>),再将数据写入根服务器

如图所示,当所有服务器都做广播或散射的时候,我们作为接收方的服务器收到各服务器发来的数据,我们将这些收到的数据进行某种规约的操作(常见如求和,求最大值)后再存入自己服务器内存中,那么这就叫规约(Reduce)。

<strong class="list-label">AllReduce(全规约)</strong>:对所有服务器上的数据做一个规约操作(如最大值、求和),再将数据写入根服务器

<figure data-latex-placement="H">
<img src="/images/1138ec4d41.png" style="width:80.0%" alt="Reduce" />
<figcaption>Reduce</figcaption>
</figure>

如图所示,同样<strong class="key-term">每一个服务器都完成上述的规约操作,那么就是全规约</strong>。这也就是分布式训练最基础的框架,将所有的数据通过规约操作集成到各个服务器中,各个服务器也就获得了完全一致的、包含原本所有服务器上计算参数的规约数据。

<strong class="list-label">ReduceScatter(散射规约)</strong>:服务器将自己的数据分为同等大小的数据块,每个服务器将根据index得到的数据做一个规约操作即,即先做Scatter再做Reduce。

<figure data-latex-placement="H">
<img src="/images/a91b52277b.png" style="width:80.0%" alt="ReduceScatter" />
<figcaption>ReduceScatter</figcaption>
</figure>

简单来讲,就是先做散射(Scatter),将服务器中数据切分成同等大小的数据块,再按照序列(Rank Index),每一个服务器所获得的参数数据做规约(Reduce)。这就类似于全聚集,只不过我们将数据不是简单拼接到一起而是做了规约操作(求和或最大值等操作)。

### 8.2.1 分布式通信算法

#### 8.2.1.1 <strong>PS算法 (Parameter Server, 参数服务器)</strong>

PS算法是一种<strong class="key-term">中心化</strong>的分布式训练架构。它将计算节点分为两种角色:

- 参数服务器/根服务器(Parameter Server, PS)负责<strong class="key-term">存储</strong>和<strong class="key-term">更新</strong>模型的全部参数；

- 工作节点(Worker)负责使用本地数据<strong class="key-term">计算梯度</strong>,并将梯度推送给PS做累积(Reduce),然后再从PS拉取最新的参数。

<figure data-latex-placement="H">
<img src="/images/4abbfcffd9.png" style="width:50.0%" alt="PS算法示意图" />
<figcaption>PS算法示意图</figcaption>
</figure>

<strong class="list-label">缺点:</strong>

- 每一轮的训练迭代都需要所有卡都将数据同步完做一次Reduce才算结束,并行的卡很多的时候,<strong class="key-term">木桶效应</strong>就会很严重,一旦有一张卡速度较慢会拖慢整个集群的速度,计算效率低。

- <strong class="key-term">Reducer服务器任务过重</strong>,成为瓶颈,所有的节点需要和Reducer进行数据、梯度和参数的通信,当模型较大或者数据较大的时候,通信开销很大,根节点收到巨量的数据,从而形成瓶颈。

#### 8.2.1.2 <strong>Halving and Doubling (HD) 算法 (Recursive Doubling)</strong>

<strong class="list-label">阶段一:</strong>规约-散射 (Reduce-Scatter):

1.  在 ‘log₂(N)‘ 步中,节点间的通信距离(步长)按 1, 2, 4, ..., N/2 的规律加倍。

2.  第1步 (距离=1):节点 ‘i‘ 与节点 ‘i+1‘ 配对。它们交换自己数据数组的一半<strong class="key-term">的一半</strong>,并进行累加。例如,节点 ‘i‘ 发送前半部分给 ‘i+1‘,接收 ‘i+1‘ 的前半部分并累加；同时发送后半部分,接收后半部分并累-加。完成后,节点 ‘i‘ 保留累加的前半部分,节点 ‘i+1‘ 保留累加的后半部分。

3.  第2步 (距离=2):节点 ‘i‘ 与节点 ‘i+2‘ 配对。它们交换的数据块是上一步规约后的结果,数据量仍然是数组的一半。

4.  ...依此类推...

5.  <strong class="list-label">阶段结果:</strong>经过 ‘log₂(N)‘ 步后,每个节点都只拥有整个数组的 ‘1/N‘,但这 ‘1/N‘ 的部分是所有节点对应部分的全局规约结果。

<strong class="list-label">阶段二:</strong>全收集 (All-Gather):

1.  这个阶段是第一阶段的<strong class="key-term">逆过程</strong>。

2.  在 ‘log₂(N)‘ 步中,节点间的通信距离按 N/2, ..., 4, 2, 1 的规律减半。

3.  第1步 (距离=N/2):在第一阶段最后一步通信的节点对(例如 ‘i‘ 和 ‘i+N/2‘)互相交换它们持有的 ‘1/N‘ 的最终结果。现在它们各自拥有了 ‘2/N‘ 的全局结果。

4.  ...依此类推...

5.  <strong class="list-label">最终结果:</strong>经过 ‘log₂(N)‘ 步的逆向操作,每个节点都逐步从邻居那里收集到了所有的最终数据块,最终恢复出完整的全局规约结果。

<figure data-latex-placement="H">
<img src="/images/db8c9b3c48.png" style="width:50.0%" alt="HD算法示意图" />
<figcaption>HD算法示意图</figcaption>
</figure>

#### 8.2.1.3 <strong>Ring算法 (Ring-AllReduce)</strong>

<strong class="list-label">核心思想</strong>:将数据分成N个块(chunk),通过两个阶段,每个阶段‘N-1‘步,完成规约和分发。

<strong class="list-label">工作流程</strong>:

<strong class="list-label">阶段一:</strong>散射-规约 (Scatter-Reduce):

1.  <strong class="list-label">分块:</strong>每个节点都将自己的数据数组分成N个小块。

2.  <strong class="list-label">传递与累加:</strong>在 ‘N-1‘ 步中,进行循环传递。

3.  在第 ‘k‘ 步,每个节点 ‘i‘ 会将自己的第 ‘(i-k+N)

4.  同时,它会从上一个节点 ‘(i-1+N)

5.  <strong class="list-label">阶段结果:</strong>经过 ‘N-1‘ 步后,每个数据块都完整地绕环一周,并累加了所有节点上对应位置的值。此时,每个节点 ‘i‘ 都拥有一个最终的、全局规约后的数据块(具体来说是第 ‘(i-1+N)

<figure data-latex-placement="H">
<img src="/images/dfee5738f1.png" style="width:80.0%" alt="Ring算法示意图1" />
<figcaption>Ring算法示意图1</figcaption>
</figure>

<figure data-latex-placement="H">
<img src="/images/e01aa6348b.png" style="width:80.0%" alt="Ring算法示意图2" />
<figcaption>Ring算法示意图2</figcaption>
</figure>

<strong class="list-label">阶段二:</strong>全收集 (All-Gather):

1.  <strong class="list-label">再次传递:</strong>现在每个节点都持有一个“最终版”的数据块,目标是让所有节点都拥有所有N个“最终版”数据块。

2.  <strong class="list-label">简单传递:</strong>在接下来的 ‘N-1‘ 步中,节点们只是单纯地传递它们手中的“最终版”数据块,不再进行累加。

3.  <strong class="list-label">最终结果:</strong>经过 ‘N-1‘ 步后,所有最终块都在环上完整地跑了一圈,每个节点都收集到了所有N个最终块,从而得到了完整的全局规约结果。

<figure data-latex-placement="H">
<img src="/images/13b8dd3352.png" style="width:50.0%" alt="Ring算法示意图3" />
<figcaption>Ring算法示意图3</figcaption>
</figure>

<strong class="list-label">性能特点</strong>:

1.  <strong class="list-label">延迟:</strong>总共需要 ‘2 \* (N-1)‘ 次通信步骤,延迟与节点数 ‘N‘ 成正比。

2.  <strong class="list-label">带宽:</strong>在任何时刻,每个节点只发送和接收一小块数据(大小为 ‘M/N‘)。算法可以使链路带宽持续被占满,因此带宽利用效率非常高。它被认为是<strong class="key-term">带宽最优 (Bandwidth Optimal)</strong>的。现代的pytorch的DDP并行的底层计算也是靠Ring算法。

## 8.3 DDP

<span id="sec-9-3"></span>

[参考资料 8.3](/part-2/chapter-8#read-9-3)

### 8.3.1 DDP和DP的区别

<span id="sec-9-3-1"></span> DDP(Distribute Data Parallel)是在DP(Data Paralle)的基础上的进一步优化。两者的主要特性的对比:

| <strong>特性</strong> | <strong>DP (DataParallel)</strong> | <strong>DDP (DistributedDataParallel)</strong> |
|:---|:---|:---|
| <strong>底层实现</strong> | 单进程多线程 | 多进程(通常一个 GPU 一个进程) |
| <strong>模型复制</strong> | 每个前向传播,主 GPU 上的模型会被复制到所有辅助 GPU。 | 每个进程维护一个独立的模型副本。 |
| <strong>数据处理</strong> | 主 GPU 将输入数据分割,并分发给所有 GPU。 | 每个进程独立地加载数据,或从数据加载器获取其应处理的数据分片。 |
| <strong>前向、反向传播</strong> | 每次传播前,主GPU将模型分给其他GPU,计算完了之后其他GPU再把结果返回给主GPU,不保存模型。 | 每个GPU都有保有独立的模型,独立进行计算。 |
| <strong>计算负载</strong> | <strong>不均衡</strong>。主 GPU 承担额外的任务(如收集输出、计算 Loss、计算梯度、更新参数)。 | <strong>均衡</strong>。每个进程独立完成前向、反向传播和参数更新。 |
| <strong>梯度同步</strong> | 反向传播在主 GPU 上完成,主 GPU 更新模型。 | 每个进程计算本地梯度,通过 All-Reduce 在所有进程间同步并平均梯度。 |
| <strong>通信效率</strong> | 效率较低,模型和数据在主 GPU 和辅助 GPU 之间频繁传输。每个前向传播都需要复制模型。 | 效率较高,只在反向传播时同步梯度(相对较小的数据量)。利用高效的通信库。 |
| <strong>GIL 影响</strong> | 受 Python 全局解释器锁(GIL)的潜在影响。 | 由于是多进程,不易受 GIL 影响。 |
| <strong>适用场景</strong> | 单机多卡,代码简单,快速验证小模型。 | 单机多卡、多机多卡,大规模模型训练,追求高性能和更好的可伸缩性。 |
| <strong>pytorch代码</strong> | `torch.nn.DataParallel` |  |

DDP的工作流程可以概括为以下几个步骤:

1.  <strong class="note-label">初始化</strong>

    - <strong class="list-label">多进程启动</strong>:DDP采用多进程模型,为每个GPU分配一个独立的进程。这与Python的<strong class="key-term">全局解释器锁(GIL)</strong>完美契合,因为不同进程拥有独立的内存空间和Python解释器,可以实现真正的并行计算。

    - <strong class="list-label">进程组建立</strong>:通过‘torch.distributed.ini_process_group‘函数初始化一个进程组(Process Group),组内的所有进程共同参与训练。每个进程被分配一个唯一的排名(rank),从0到N-1(N是总GPU数)。通常,rank 0被指定为主进程(master)。

    - <strong class="list-label">模型复制</strong>:使用广播集合通信操作(broadcast)将模型参数从rank 0 的设备发送到所有其他设备。在训练开始时,每个设备都持有相同的模型参数和优化器状态副本(例如Adam优化器中累积的梯度统计量)。

2.  <strong class="note-label">数据分发</strong>

    - <strong class="list-label">数据分发</strong>:给定一个包含n个样本的批次时,该批次会被分片处理,每个设备接收 n/d 个互不重叠的样本(其中d是用于数据并行训练的设备数量)。n必须能被d整除,因为训练时间受限于最慢进程的速度(如果不能整除,那么就会有某一/几个设备的数量大于其他设备,导致不同设备之间时间不一致,产生<strong class="key-term">木桶效应</strong>)

3.  <strong class="note-label">计算与梯度同步(DDP的精髓)</strong>

    - <strong class="list-label">前向传播</strong>:每个进程独立地在其分配到的数据批次上执行模型的前向传播,计算损失。

    - <strong class="list-label">反向传播与梯度同步</strong>:当调用‘loss.backward()‘时,梯度从模型的输出层向输入层逐层计算(这里会有<strong class="key-term">梯度分桶、计算通信重叠</strong>等优化操作,后面会细说),每个反向传播进程计算得到的梯度都会保存。

    - <strong class="list-label">All-Reduce操作</strong>:这是去中心化的梯度同步算法。在All-Reduce操作中,每个进程都会贡献出自己计算出的梯度,并最终从操作中获得所有进程梯度的总和(或平均值)。最常用的高效实现是<strong class="key-term">Ring-AllReduce</strong>。

4.  <strong class="note-label">模型更新</strong>

    - <strong class="list-label">同步更新</strong>:All-Reduce操作完成后,每个进程都拥有了完全相同的平均梯度。

    - <strong class="list-label">本地更新</strong>:每个进程独立调用优化器(如‘optimizer.step()‘)来更新其本地模型副本的权重。因为初始权重相同,更新的梯度也相同,所以更新后的模型权重在所有进程间依然保持严格一致。

### 8.3.2 DDP算法优化

<span id="sec-9-3-2"></span> 主要涉及了两种重要的方法,<strong class="key-term">梯度分桶</strong>和<strong class="key-term">计算通信重叠</strong>。

之前朴素实现DDP的性能瓶颈:

1.  <strong class="list-label">通信开销大</strong>:它为模型中的每一个参数张量都单独执行一次‘all-reduce‘(全局规约)通信操作。频繁的通信调用会带来显著的性能开销。

2.  <strong class="list-label">通信与计算未重叠</strong>:它需要等待整个反向传播过程计算完所有梯度后,才开始进行通信,这浪费了宝贵的计算资源和时间。

针对第一个瓶颈,容易想到为了避免用循环每次都要全规约进行通信,我们可以将循环里的所有的梯度张量先存起来,等循环结束后再一起发送。这样做的好处是减少了通信的开销,但缺点是没能充分发挥<strong class="key-term">“并行”</strong>和<strong class="key-term">“异步”</strong>的力量。

具体而言,由于反向传播不止计算一个参数的梯度,我们完全可以等一个参数梯度计算好了之后,让它<strong class="key-term">立刻异步执行all_reduce</strong>,然后再去算下一个参数的梯度,这样类似流水线的做法充分发挥了<strong class="key-term">“并行”</strong>的能力,理论上更省时间。

<figure data-latex-placement="H">
<img src="/images/f7c28d19ea.png" style="width:50.0%" alt="计算通信重叠" />
<figcaption>计算通信重叠</figcaption>
</figure>

在具体实现里面,反向传播的代码loss.backward()比较固定,但pytorch也配备了hook(钩子)操作。<strong class="key-term">钩子</strong>是一种<strong class="key-term">允许你“挂载”自定义代码到现有程序执行流程中的一种机制</strong>。它使得你可以在不修改程序或框架源代码的情况下,对程序的行为进行<strong class="key-term">监视、修改或增强</strong>。它本质上是一种<strong class="key-term">回调 (callback)</strong>思想的体现:你预先注册一个函数,然后由系统在某个特定事件发生时“回头调用”它。

因此,我们可以在反向传播完一个参数之后利用hook操作异步进行 <strong class="key-term">all_reduce</strong>,从而实现计算和通信的流水线重叠。

```python
    for param in reversed(list(self.module.parameters())):#这里由于是反向传播,梯度从后往前计算,所以加一个reverse
      if param.requires_grad:
          param.register_post_accumulate_grad_hook(self._create_hook(param))
```

接下来介绍分桶排序优化的方法,它更好结合了两种方法缓解两个瓶颈。

相比于之前的每一个参数一计算完就进行通信的方式,利用桶排序对计算好的参数<strong class="key-term">分批进行通信</strong>的方式能显著减少通信次数,从而减少网络压力,而且利用桶的方式有更好地灵活性。比方说某个任务一定要追求极低的时延,那就可以把桶设置成一个桶只装一个参数的形式,这就变成了之前的这种形式,如果某个任务所处的网络环境不好,不方便经常通信,那就把桶设置的大一些,这样就减少了通信次数。

<figure data-latex-placement="H">
<img src="/images/a05977b54a.png" style="width:80.0%" alt="桶排序流程" />
<figcaption>桶排序流程</figcaption>
</figure>

下面介绍整体的流程:

1.  <strong class="note-label">初始化阶段:</strong>桶的构建 (Bucket Construction)

    - <strong class="list-label">参数排序</strong>:DDP首先会拿到模型的所有可训练参数 (‘model.parameters()‘)。它会按照这些参数在反向传播中梯度被计算的<strong class="key-term">逆序</strong>来排列它们。(逆序至关重要,因为反向传播是从模型的最后一层开始,逐层向前计算梯度的。因此,排在最前面的参数(例如模型最后一层全连接层的权重和偏置)是反向传播时最先准备好梯度的。)

    - <strong class="list-label">参数入桶</strong>:DDP会遍历这个逆序的参数列表,依次将参数装入桶中。每个桶有一个预设的<strong class="key-term">容量上限</strong>,由参数‘bucket_cap_mb‘控制(默认为25MB)。DDP会持续向当前桶中添加参数,直到加入下一个参数会导致桶的总大小超过容量上限为止。此时,当前桶构建完毕,DDP会新建一个空桶,并将刚才那个放不下的参数作为新桶的第一个成员,然后继续该过程,直到所有参数都被分配到桶中。

    - <strong class="list-label">结果</strong>:初始化完成后,DDP内部就维护了一个桶的列表,每个桶里包含了一组参数。例如,对于一个典型的Transformer模型,最后一个‘Linear‘层和‘LayerNorm‘层的参数可能在同一个桶里(桶0),倒数第二个注意力块的参数可能在桶1,以此类推。

2.  <strong class="note-label">反向传播阶段:</strong>Autograd Hook的触发

    - <strong class="list-label">注册钩子 (Registering Hooks)</strong>:在初始化阶段,DDP会遍历模型的所有参数,并为每一个参数的梯度(‘.grad‘属性)注册一个回调函数,即‘autograd‘钩子。

    - <strong class="list-label">触发机制</strong>:当你调用‘loss.backward()‘时,PyTorch的‘autograd‘引擎会负责计算每个参数的梯度。每当一个参数的梯度被完全计算出来后,‘autograd‘引擎会<strong class="key-term">立即自动调用</strong>之前注册在它上面的那个钩子函数。

3.  <strong class="note-label">核心逻辑:</strong>标记、检查与异步通信

    - <strong class="list-label">标记就绪</strong>:钩子函数首先会找到该梯度所属的桶,并在桶内部的计数器或状态中,将这个梯度标记为“已就绪”(Ready)。

    - <strong class="list-label">检查桶满</strong>:接下来,钩子会检查这个桶内是否所有参数的梯度都已处于“已就绪”状态。

    - <strong class="list-label">触发通信</strong>:

      - 如果桶内还有其他参数的梯度没算好,钩子函数不做任何事,直接返回。继续进行下一轮的梯度计算。

      - 如果桶内所有参数的梯度都算好了,DDP会执行以下操作:

        - <strong class="list-label">梯度合并 (Flatten)</strong>:为了提高通信效率,DDP会将这个桶里所有零散的梯度张量复制并拼接成一个巨大、扁平且在内存上连续的单一张量。这被称为‘flatten‘操作,因为单次传输一个大Tensor远比多次传输多个小Tensor要快。

        - <strong class="list-label">启动异步All-Reduce</strong>:DDP会立即对这个合并后的大Tensor调用‘torch.distributed.all_reduce(..., async_op=True)‘。告诉NCLL开启异步通信,这边不阻塞继续下一个桶的梯度计算。

4.  <strong class="note-label">等待与梯度回写</strong>

    - <strong class="list-label">等待与同步</strong>:在整个‘loss.backward()‘的末尾,DDP会确保所有桶的All-Reduce操作都已完成。它通常只需要等待最后一个被触发的桶的通信完成即可,因为前面的通信已经和计算重叠了。

    - <strong class="list-label">梯度解包与回写 (Unflatten)</strong>:当一个桶的All-Reduce操作完成后,其对应的扁平化Tensor中存储的是该桶所有梯度拼接后的结果。DDP会将其除以分布式环境的‘world_size‘(即GPU总数)来得到平均梯度。然后,再将这个扁平Tensor中的值“解包”,按顺序写回到该桶内每个原始参数的‘.grad‘属性中。

    - <strong class="list-label">优化器更新</strong>:至此,所有参数的‘.grad‘属性都包含了同步后的全局平均梯度,优化器(‘optimizer.step()‘)便可以安全地使用它们来更新模型权重了。

## 8.4 混合并行(4D 并行)

[参考资料 8.4](/part-2/chapter-8#read-9-4)

混合并行技术是指同时使用多种并行技术,比如数据并行(DP)和张量并行(TP),或者数据并行和流水线并行(PP)。

### 8.4.1 每种并行技术简介

#### 8.4.1.1 <strong>数据并行(Data Parallelism, DP)</strong>

最常见的一种并行方式。它将同一个模型完整地复制到多块GPU上。然后,将一大批训练数据切分成多份小数据,每块GPU拿到一份小数据独立进行计算。在每个训练步骤结束时,所有GPU会同步一次计算结果(通常是梯度),以确保所有模型副本的参数保持一致。

<strong class="list-label">DP工作流程:</strong>

1.  <strong class="list-label">数据切分:</strong>

    - 在一个训练循环中,你从数据加载器中取出一个大的‘batch‘。

    - 主GPU(通常是‘cuda:0‘)会将这个大‘batch‘沿着批次维度进行切分,分成N个子批次,其中N是参与训练的GPU数量。

    - 然后,主GPU将这些子批次分发到其他各个GPU上(包括它自己)。

2.  <strong class="list-label">模型复制:</strong>

    - 在每次前向传播开始之前,主GPU(‘cuda:0‘)上的最新模型参数会被复制并广播(broadcast)到所有其他的从属GPU上,确保每个GPU都使用完全相同的模型。

3.  <strong class="list-label">并行计算:</strong>

    - 每个GPU拿到自己的子批次数据和复制过来的模型。

    - 所有GPU<strong class="key-term">并行地、独立地</strong>执行前向传播,计算出各自的损失。

    - 接着,每个GPU根据自己的损失,并行地、独立地执行后向传播,计算出模型参数的梯度。

4.  <strong class="list-label">梯度规约 (Gather & Reduce):</strong>

    - 所有从属GPU将它们计算出的梯度全部发送回<strong class="key-term">主GPU</strong> (‘cuda:0‘)。

    - 主GPU负责收集所有GPU传来的梯度,并将它们进行<strong class="key-term">Reduce(取平均)</strong>操作。

5.  <strong class="list-label">模型参数更新:</strong>

    - 主GPU使用规约后的总梯度,通过优化器来更新其自身的模型参数。

    - <strong class="list-label">请注意:</strong><strong class="key-term">只有主GPU上的模型参数得到了更新</strong>。其他GPU上的模型在此刻还是旧的。

6.  <strong class="list-label">模型广播 (Broadcast):</strong>

    - 为了开始下一次迭代,主GPU会将刚刚更新完毕的、最新的模型参数再次广播给所有从属GPU。

    - 循环回到第1步,处理下一个‘batch‘的数据。

<figure data-latex-placement="H">
<img src="/images/a07dc8c671.png" style="width:50.0%" alt="DP并行示意图" />
<figcaption>DP并行示意图</figcaption>
</figure>

<strong class="critical-term">张量并行 (TensorParallel, TP):</strong>

这种方式用于解决<strong class="key-term">模型本身过大、单块GPU无法容纳</strong>的问题。它不对模型进行复制,而是将模型中的某个大张量(例如一个巨大的权重矩阵)切分成多个小块(Shard),每一块分别存放在不同的GPU上。计算时,每块GPU只处理自己负责的那一小块张量,最后再将结果同步汇总。

举个例子,对于一个神经网络计算单元: ‘Y = X \* W‘

- ‘X‘ 是输入激活(Input Activation)

- ‘W‘是权重矩阵(Weight Matrix)

- ‘Y‘ 是输出激活(Output Activation)

如果权重矩阵 ‘A‘ 太大,无法放入单个GPU,我们就需要将 ‘A‘ 切分到多个GPU上,并设计一种方法让这些GPU协同完成 ‘X \* W‘ 这个运算。

<strong class="list-label">张量并行的具体实现方式:</strong>

对于张量的切分也是有讲究的,如果随便切分很难实现‘X\*W‘的同步运算,因此‘W‘的切分往往是按照行或者按照列

<strong class="note-label">列并行 (Column Parallelism):</strong>

我们将权重矩阵 ‘W‘ <strong class="key-term">按列</strong>切分。假设我们有2个GPU:

‘W = \[W₁ \| W₂\]‘

其中 ‘W₁‘ 和 ‘W₂‘ 分别是 ‘W‘ 的左半部分和右半部分,它们被分发到 GPU 1 和 GPU 2。

1.  <strong class="list-label">前向传播 (Forward Pass):</strong>

    输入 ‘X‘ 被复制到两个GPU上。GPU 1 计算 ‘Y₁ = X \* W₁‘ ,GPU 2 计算 ‘Y₂ = X \* W₂‘ 。计算完成后,我们将结果拼接起来:‘Y = \[Y₁ \| Y₂\]‘。注意,在这个过程中,‘Y‘ 的不同部分自然地分布在不同的GPU上。

2.  <strong class="list-label">通信:</strong>在这一步的前向传播中,<strong class="key-term">不需要</strong>任何通信。每个GPU独立计算自己那一部分。

3.  <strong class="list-label">反向传播 (Backward Pass):</strong>

    - 在计算输入的梯度 ‘dL/dX‘ 时,根据链式法则,它等于 ‘(dL/dY) \* Wᵀ‘。

    - 由于 ‘Y‘ 和 ‘W‘ 都是分片的,‘dL/dX‘ 的计算会变成 ‘(dL/dY₁) \* W₁ᵀ + (dL/dY₂) \* W₂ᵀ‘。

    - GPU 1 计算 ‘(dL/dY₁) \* W₁ᵀ‘,GPU 2 计算 ‘(dL/dY₂) \* W₂ᵀ‘。

    - 为了得到最终完整的 ‘dL/dX‘,两个GPU需要将各自计算出的梯度相加。

4.  <strong class="list-label">通信:</strong>这里需要一次 <strong class="key-term">All-Reduce</strong> 操作,将所有GPU上的梯度累加,并同步回每个GPU。

<figure data-latex-placement="H">
<img src="/images/cbd6c931fb.png" style="width:80.0%" alt="列并行示意图" />
<figcaption>列并行示意图</figcaption>
</figure>

<strong class="note-label">行并行 (Row Parallelism):</strong>

现在,我们将权重矩阵 ‘W‘ <strong class="key-term">按行</strong>切分:

‘W = \[W₁; W₂\]‘ (这里用分号表示按行堆叠)

‘W₁‘ 和 ‘W₂‘ 分别是 ‘W‘ 的上半部分和下半部分,被分发到 GPU 1 和 GPU 2。

1.  <strong class="list-label">前向传播 (Forward Pass):</strong>

    此时,输入 ‘X‘ 必须是已经被分片的(例如,它是上一个列并行层的输出)。假设 ‘X = \[X₁ \| X₂\]‘。

    完整的计算是 ‘Y = X \* W = \[X₁ \| X₂\] \* \[W₁; W₂\] = X₁\*W₁ + X₂\*W₂‘。GPU 1 计算 ‘Y₁ = X₁ \* W₁‘。GPU 2 计算 ‘Y₂ = X₂ \* W₂‘。为了得到最终的 ‘Y‘,我们需要将 ‘Y₁‘ 和 ‘Y₂‘ 相加。

2.  <strong class="list-label">通信:</strong>这里<strong class="key-term">需要一次 All-Reduce 操作</strong>,将各个GPU上的部分结果相加,得到最终的完整输出 ‘Y‘。

3.  <strong class="list-label">反向传播 (Backward Pass):</strong> 在计算输入的梯度 ‘dL/dX‘ 时,它的一部分 ‘dL/dX₁‘ 等于 ‘(dL/dY) \* W₁ᵀ‘,这可以在GPU 1上独立完成。

4.  <strong class="list-label">通信:</strong>在这一步的反向传播中,<strong class="key-term">不需要</strong> All-Reduce 通信。

<figure data-latex-placement="H">
<img src="/images/5d20ffd63c.png" style="width:80.0%" alt="行并行示意图" />
<figcaption>行并行示意图</figcaption>
</figure>

<strong class="note-label">行列结合:</strong>

以FFN为例,它通常由两个线性层和一个激活函数组成:‘Y = GELU(X\*A) \* B‘

1.  <strong class="list-label">第一个线性层 (‘X\*A‘):</strong>使用列并行。

    - ‘A‘ 按列切分 ‘\[A₁, A₂\]‘。

    - 输出 ‘GELU(X\*A)‘ 也是按列切分的 ‘\[Y’₁, Y’₂\]‘,分布在不同GPU上。

    - 前向传播<strong class="key-term">无通信</strong>。反向传播需要一次 <strong class="key-term">All-Reduce</strong>。

2.  <strong class="list-label">第二个线性层 (‘... \* B‘):</strong>使用行并行。

    - ‘B‘ 按行切分 ‘\[B₁; B₂\]‘。

    - 它的输入 ‘\[Y’₁, Y’₂\]‘ 恰好是上一步的分布式输出。

    - 前向传播需要一次 <strong class="key-term">All-Reduce</strong> 来合并最终结果。反向传播<strong class="key-term">无通信</strong>

<figure data-latex-placement="H">
<img src="/images/7ed0f53a9f.png" style="width:80.0%" alt="行列结合示意图" />
<figcaption>行列结合示意图</figcaption>
</figure>

#### 8.4.1.2 <strong>流水线并行 (PipelineParallel, PP)</strong>

这种方式也是为了解决模型过大的问题,但它的切分维度不同。它将整个模型<strong class="key-term">按层(Layer)进行切分</strong>,类似于工厂里的流水线。例如,一个30层的模型,可以把前10层放在GPU 1上,中间10层放在GPU 2上,最后10层放在GPU 3上。数据像在流水线上传送带一样,依次流过这几块GPU,完成一次完整的计算。

<strong class="list-label">基本原理:</strong>

假设我们将一个模型切分成4个部分,分别放在GPU 0, 1, 2, 3上。

<strong class="list-label">前向传播 (Forward Pass):</strong>

1.  ‘Batch 1‘ 进入 ‘GPU 0‘ 进行计算。

2.  ‘GPU 0‘ 计算完成后,将其输出(称为激活值)发送给 ‘GPU 1‘。

3.  ‘GPU 1‘ 收到后开始计算,计算完成后将激活值发送给 ‘GPU 2‘。

4.  ...以此类推,直到 ‘GPU 3‘ 完成计算。

<strong class="list-label">后向传播 (Backward Pass):</strong>

1.  ‘GPU 3‘ 首先计算梯度。

2.  ‘GPU 3‘ 将其计算的梯度发送回 ‘GPU 2‘。

3.  ‘GPU 2‘ 接收到梯度后,计算自己的梯度,再发送回 ‘GPU 1‘。

4.  ...以此类推,直到 ‘GPU 0‘ 完成梯度计算。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 流水线并行的缺陷——巨大的“气泡”(Bubble)</p>

在上述流程中,当“GPU 1”在工作时,“GPU 0”在做什么？它在空闲等待。当“GPU 2”在工作时,“GPU 0”和“GPU 1”都在空闲等待。在任何一个时间点,只有一个GPU在忙碌,其他的GPU都在闲置。这种GPU利用率的巨大浪费,我们称之为流水线气泡 (Pipeline Bubble)。

</div>

<strong class="list-label">优化:</strong>微批次流水线

为了解决“气泡”问题,现代流水线并行引入了一项关键技术:<strong class="key-term">将一个大的训练批次(mini-batch)进一步切分成若干个更小的微批次(micro-batch)</strong>。

通过这种方式,虽然在开始和结束阶段仍有无法避免的“气泡”,但中间大部分时间GPU的利用率被极大地提高了。微批次数量越多,稳态阶段占比就越大,“气泡”的相对开销就越小。

<strong class="note-label">梯度累积:</strong>在后向传播阶段,每个GPU会计算出自己所负责模型部分的梯度。它会累积所有微批次的梯度,直到整个mini-batch处理完毕,然后用累积的总梯度进行一次参数更新

<figure data-latex-placement="H">
<img src="/images/c62bde6fed.png" style="width:50.0%" alt="流水线并行示意图" />
<figcaption>流水线并行示意图</figcaption>
</figure>

#### 8.4.1.3 <strong>ZeRO (Zero Redundancy Optimizer)</strong>

这是一种<strong class="key-term">旨在消除数据并行中的冗余、极致节省显存</strong>的技术。它类似于张量并行,也会<strong class="key-term">对模型的参数、梯度和优化器状态进行分片</strong>。但它的核心特点是:只有在计算需要时,才会动态地将完整的张量临时重建出来,计算一完成就立即丢弃,从而大大降低了显存峰值。它还支持将暂时用不到的数据“卸载(Offload)”到CPU内存或硬盘上,进一步节省宝贵的GPU显存。

<strong class="list-label">DDP的内存冗余问题</strong>

在标准的数据并行训练中,每个GPU都需要存储三份主要的模型相关数据:

1.  <strong class="list-label">模型参数 (Model Parameters)</strong>:模型的权重,例如‘Float16‘或‘Float32‘。

2.  <strong class="list-label">梯度 (Gradients)</strong>:反向传播计算出的参数梯度,大小与参数相同。

3.  <strong class="list-label">优化器状态 (Optimizer States)</strong>:例如Adam优化器需要为每个参数存储其<strong class="key-term">一阶矩(Momentum)和二阶矩(Variance)</strong>,通常是参数量的2倍。

假设模型参数量为 M,使用Adam优化器和FP16混合精度训练,那么在‘N‘个GPU上进行数据并行时,每个GPU的内存占用大约是:

‘Memory_per_GPU ≈ (2M for FP16 Params) + (2M for FP16 Grads) + (4M for FP32 Momentum) + (4M for FP32 Variance) = 12M‘ (这还不包括激活值等其他开销)

总的内存占用是 ‘N \* 12M‘。你会发现,模型参数、梯度和优化器状态在所有‘N‘个GPU上都存在一份完整的副本,这是巨大的<strong class="key-term">内存冗余</strong>。ZeRO的使命就是消除这种冗余。

<strong class="critical-term">ZeRO的核心思想:</strong>分区 (Partitioning)

ZeRO通过将这三部分数据(P, G, O)在所有参与数据并行的GPU之间进行<strong class="key-term">分区</strong>,而<strong class="key-term">不是复制</strong>,来解决内存冗余问题。每个GPU只负责存储和更新完整数据的一个分片(Shard)。

ZeRO分三个递进的阶段来实现这一目标,每个阶段优化的程度都比前一个更深。

<strong class="note-label">Stage 1:优化器状态分区 (Partition Optimizer States)</strong>

优化器状态通常是内存的大头(在Adam中是参数量的2倍,且通常用FP32存储,总计8M)。这是最容易优化的部分。

<strong class="list-label">工作流程:</strong>

1.  <strong class="list-label">分区</strong>:将优化器状态(如Momentum和Variance)平均切分成‘N‘份,每个GPU只保存其中的 ‘1/N‘。

2.  <strong class="list-label">保留</strong>:每个GPU依然保留完整的模型参数和梯度。

3.  <strong class="list-label">更新过程 (‘optimizer.step()‘)</strong>:

    - 在反向传播结束后,所有GPU上的梯度会通过一次‘All-Reduce‘操作进行同步,确保每个GPU都有完整的、求和后的梯度。

    - 在参数更新时,每个GPU只使用它本地存储的那一小部分优化器状态,来更新对应的那一小部分模型参数。

    - 更新完成后,每个GPU只拥有了模型参数的一部分更新。

    - 最后,通过一次‘All-gather‘通信操作,所有GPU广播自己更新好的那部分参数,从而在每个GPU上重新组合出完整的、更新后的模型。

<strong class="list-label">效果:</strong>

- <strong class="list-label">内存节省</strong>: 大幅减少了优化器状态的内存占用。

- <strong class="list-label">通信开销</strong>: 与标准DDP相比,增加了一次‘All-gather‘操作来同步更新后的参数。

<strong class="note-label">Stage 2:梯度和优化器状态分区 (Partition Gradients & Optimizer States)</strong>

在Stage 1的基础上,进一步消除梯度的冗余。

<strong class="list-label">工作流程:</strong>

1.  <strong class="list-label">分区</strong>:将优化器状态<strong class="key-term">和梯度</strong>都进行‘N‘路分区。每个GPU只负责‘1/N‘的梯度和优化器状态。

2.  <strong class="list-label">保留</strong>:每个GPU依然保留完整的模型参数。

3.  <strong class="list-label">更新过程</strong>:

    - 在反向传播过程中,当梯度计算出来后,不再执行‘All-Reduce‘。取而代之的是一个更高效的‘Reduce-Scatter‘操作。这个操作会一边计算梯度总和,一边将结果直接分发给对应的GPU。例如,属于分区1的梯度会被计算总和后直接发送到GPU 1,以此类推。

    - 这样,反向传播结束后,每个GPU上只存储了它所负责的那部分梯度总和。

    - 后续的优化器更新和参数‘All-gather‘过程与Stage 1类似。

<strong class="list-label">效果:</strong>

- <strong class="list-label">内存节省</strong>: 进一步节省了梯度存储。

- <strong class="list-label">通信效率</strong>: 将DDP中的‘All-Reduce‘替换为‘Reduce-Scatter‘,通常通信量减半,效率更高。

<strong class="note-label">Stage 3: 参数、梯度和优化器状态全部分区 (Partition Everything)</strong>

这是最极致的优化,消除了所有冗余,包括模型参数本身。

<strong class="list-label">工作流程:</strong>

1.  <strong class="list-label">分区</strong>:将模型参数、梯度、优化器状态<strong class="key-term">全部</strong>进行‘N‘路分区。

2.  <strong class="list-label">保留</strong>:原则上,每个GPU在任何时刻都只拥有模型的一个分片。

3.  <strong class="list-label">计算过程 (前向/后向)</strong>:

    - 当一个GPU需要执行某一层的前向或后向计算时,如果该层所需的参数不在本地,它需要通过一次‘All-gather‘操作从其他GPU那里动态地获取这些参数。

    - 计算完成后,为了节省内存,这些临时的、非本地的参数可以被立即丢弃。

    - 这个过程需要在每一层计算前后动态地进行,对通信的要求非常高。

<strong class="list-label">效果:</strong>

- <strong class="list-label">内存节省</strong>: 达到理论最优。理论上可以将一个巨大的模型(如1万亿参数)分散到足够多的GPU上进行训练。

- <strong class="list-label">通信开销</strong>: 通信量巨大,因为它在计算过程中穿插了大量细粒度的‘All-gather‘操作。

<strong class="list-label">ZeRO的扩展:</strong>ZeRO-Offload & ZeRO-Infinity

为了应对GPU内存仍然不足或通信成本过高的情况,ZeRO还发展出了更高级的形态:

- <strong class="list-label">ZeRO-Offload</strong>: 将ZeRO分区后的数据(特别是那些不常被访问的,如优化器状态)进一步从GPU内存<strong class="key-term">卸载(Offload)</strong>到CPU内存。这极大地扩展了可训练模型的大小,代价是引入了GPU和CPU之间的PCIe通信延迟,速度会慢一些。

- <strong class="list-label">ZeRO-Infinity</strong>: ZeRO-Offload的终极版本,不仅利用CPU内存,还利用NVMe固态硬盘作为海量的内存池。这使得在有限的硬件上训练万亿级参数模型成为可能,尽管训练时间会非常长。

## 8.5 混合数据并行

### 8.5.1 DP+PP(数据并行+流水线并行)

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么需要 DP+PP？</p>

- <strong class="list-label">单独使用数据并行 (DP)</strong>:要求<strong class="key-term">每个GPU</strong>都能装下完整的模型。当模型增长到百亿甚至千亿参数时,例如一个175B的GPT-3模型(FP16下需要350GB内存),远超任何单个GPU的容量,DP就失效了。

- <strong class="list-label">单独使用流水线并行 (PP)</strong>:解决了模型过大的问题,可以将模型切分到多个GPU上。但它的扩展性受限于模型的层数和“<strong class="key-term">流水线气泡</strong>”的开销。如果你只有60层模型,使用超过60个GPU做流水线并行是没有意义的,而且气泡开销会变得极其巨大。它无法利用更多的机器来处理更多的数据。

</div>

<strong class="list-label">DP+PP的核心思想</strong>:用PP来解决单个模型过大的问题,用DP来将这个“切分后的大模型”复制多份,以扩展到更多的计算设备上,实现规模化加速。

#### 8.5.1.1 DP+PP 的工作流程

1.  <strong class="list-label">数据分发</strong>:

    一个大的mini-batch被分成‘data_parallel_size‘(例如4)份,我们称之为 ‘D1, D2, D3, D4‘。

    - <strong class="list-label">第一条流水线(节点0)</strong>获得数据 ‘D1‘。

    - <strong class="list-label">第二条流水线(节点1)</strong>获得数据 ‘D2‘。

    - 以此类推。

2.  <strong class="list-label">前向传播 & 后向传播</strong>:

    <strong class="key-term">所有流水线并行地、独立地开始工作。</strong>

    - 在第一条流水线内部,数据‘D1‘被切分成微批次,按照PP流水线并行的方式,流过‘GPU0 -\> GPU1 -\> GPU2 -\> GPU3‘,并完成前向和后向传播。

    - 同时,在第二条流水线内部,数据‘D2‘也以同样的方式流过它的四个GPU。

    - 这个阶段,不同流水线之间<strong class="key-term">没有任何通信</strong>。

3.  <strong class="list-label">梯度同步 (关键步骤)</strong>:

    - 当一条流水线完成了一个完整mini-batch(例如‘D1‘)的后向传播后,其上的每个GPU都计算出了对应模型分片的<strong class="key-term">本地梯度</strong>。

    - 此时,<strong class="key-term">数据并行</strong>的同步机制启动。

    - <strong class="key-term">DP Group 0</strong>(包含节点0的GPU0,节点1的GPU0,)中的所有GPU执行一次‘All-Reduce‘操作,将它们各自计算出的关于模型第一部分的梯度进行平均。

    - <strong class="key-term">DP Group 1</strong>(包含所有GPU1)也执行‘All-Reduce‘,同步模型第二部分的梯度。

    - 以此类推。

    <strong class="key-term">这个‘All-Reduce‘操作是在不同节点之间进行的,通常通过高速网络(如InfiniBand)完成。</strong>

4.  <strong class="list-label">参数更新</strong>:

    - 梯度同步完成后,每个GPU上都有了<strong class="key-term">全局平均梯度</strong>。

    - 每个GPU使用这个梯度来更新它本地存储的那部分模型参数。

<figure data-latex-placement="H">
<img src="/images/1f5deaf043.png" style="width:80.0%" alt="DP+PP示意图" />
<figcaption>DP+PP示意图</figcaption>
</figure>

### 8.5.2 3D并行 (DP+PP+TP)

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · 为什么需要3D并行？2D并行的瓶颈</p>

在我们讨论DP+PP时,我们已经解决了一个核心问题:如何训练一个<strong class="key-term">大到单个节点都装不下的模型</strong>,并将其扩展到多个节点。但这还不够,因为DP+PP组合本身也存在瓶颈:

1.  <strong class="list-label">流水线气泡依然存在</strong>: 即使结合了DP,每条流水线(PP)内部的“气泡”开销依然是其固有效率损失。为了减少气泡,我们需要增加微批次数量,但这会影响模型的收敛性(小批量噪声问题)和硬件效率(小计算量的GEMM操作效率低)。

2.  <strong class="list-label">单层计算/内存瓶颈</strong>: 想象一个流水线阶段(Stage)。虽然整个模型被切分了,但这个阶段本身可能仍然非常巨大。特别是对于某些Transformer层,<strong class="key-term">单个自注意力(Self-Attention)或前馈网络(FFN)层的权重矩阵和激活值</strong>,可能就大到无法放入单个GPU。<strong class="key-term">PP是按层(或层块)切分,它无法解决单层内部的瓶颈。</strong>

</div>

这就是<strong class="key-term">张量并行 (TP)</strong> 发挥关键作用的地方。TP可以在一个操作(如矩阵乘法)的内部进行并行计算,从而解决单层过大的问题。

#### 8.5.2.1 3D并行的架构:一个三维计算立方体

3D并行架构可以用”立方体”结构来直观想象:每个GPU都有一个‘(dp_rank, pp_rank, tp_rank)‘的坐标。

<figure data-latex-placement="H">
<img src="/images/6f94037b26.png" style="width:50.0%" alt="3D并行示意图" />
<figcaption>3D并行示意图</figcaption>
</figure>

1.  <strong class="list-label">维度1:</strong> 流水线并行 (PP) - 模型的“深度”

    - 它将模型的层(layers)沿着这个维度进行切分。

    - GPU ‘(x, 0, z)‘ 执行模型的第1个阶段,‘(x, 1, z)‘ 执行第2个阶段,以此类推。

    - 通信发生在<strong class="key-term">PP组</strong>内,即 ‘pp_rank‘ 不同的GPU之间(例如,从‘(x,0,z)‘到‘(x,1,z)‘)。

2.  <strong class="list-label">维度2:</strong> 张量并行 (TP) - 模型的“宽度”

    - 它将模型中<strong class="key-term">每一层的计算</strong>(特别是巨大的权重矩阵)沿着这个维度进行切分。

    - GPU ‘(0, y, z)‘ 和 ‘(1, y, z)‘ 协同工作,共同完成一个层的计算。

    - 通信发生在<strong class="key-term">TP组</strong>内,即 ‘tp_rank‘ 不同的GPU之间。这种通信非常频繁,<strong class="key-term">必须使用最高速的互联技术</strong>,因此TP通常被限制在单个节点内部。

3.  <strong class="list-label">维度3:</strong> 数据并行 (DP) - 模型的“数量”

    - 它将整个“PP+TP”组合出的模型复制多份,每份处理不同的数据。

    - GPU ‘(x, y, 0)‘ 和 ‘(x, y, 1)‘ 持有完全相同的模型分片,但处理不同的数据批次。

    - 通信发生在<strong class="key-term">DP组</strong>内,即 ‘dp_rank‘ 不同的GPU之间。这种通信是梯度同步,通常跨节点。

<strong class="key-term">总GPU数量 = ‘data_parallel_size‘ \* ‘pipeline_parallel_size‘ \* ‘tensor_parallel_size‘</strong>

#### 8.5.2.2 一个具体的例子:用64个GPU进行3D并行

假设我们有8个节点,每个节点4个GPU,总共32个GPU。我们可以这样配置:

- ‘tensor_parallel_size = 4‘: 使用每个节点内的4个GPU进行张量并行。

- ‘pipeline_parallel_size = 4‘: 将模型切分为4个流水线阶段。

- ‘data_parallel_size = 2‘: 复制出2个数据并行的副本。

<strong class="list-label">一个GPU的角色:</strong> 让我们看看‘GPU9‘(全局排名第9的GPU)在做什么。

- 它位于<strong class="key-term">节点1</strong>上(‘GPU8‘到‘GPU11‘在节点1)。

- 它的<strong class="key-term">TP组</strong>是节点1上的所有4个GPU (‘GPU8‘ 到 ‘GPU11‘)。它们共同负责计算模型中的每一层。

- 它的<strong class="key-term">PP阶段</strong>是什么？假设我们将4个PP阶段分配给4组节点,那么节点0和1可能构成一个数据并行副本,节点2和3构成另一个。那么‘GPU9‘位于PP阶段2。

- 它的<strong class="key-term">DP组</strong>是哪个？它会和另一个数据副本中处于相同TP和PP位置的GPU构成DP组。

<strong class="list-label">工作流程一览:</strong>

1.  <strong class="list-label">输入数据 ‘D‘</strong>被分成两部分 ‘D1‘ 和 ‘D2‘,分别送往两个数据并行副本。

2.  <strong class="list-label">在数据副本1中:</strong>

    - ‘D1‘被切成微批次。

    - <strong class="list-label">阶段0(例如,由节点0的4个GPU负责)</strong>: 这4个GPU通过<strong class="key-term">张量并行</strong>协同计算模型的前$\frac{1}{4}$的层。计算完成后,将激活值发送给下一阶段。

    - <strong class="list-label">阶段1(例如,由节点3的4个GPU负责)</strong>: 接收来自阶段0的激活值,通过<strong class="key-term">张量并行</strong>计算模型的第二个$\frac{1}{4}$的层。

    - 以此类推,完成整个流水线的前向和后向传播。

3.  在<strong class="key-term">数据副本2</strong>中,完全相同的过程用数据‘D2‘并行地发生。

4.  <strong class="list-label">梯度同步</strong>:

    - 在每个副本完成后,处于相同‘pp_rank‘和‘tp_rank‘的GPU之间进行‘All-Reduce‘。

    - 例如,节点0上的‘GPU0‘与节点1上的‘GPU4‘进行梯度同步。

### 8.5.3 4D并行(DP+PP+TP+ZeRO

#### 8.5.3.1 为什么需要这个组合？

我们已经知道3D并行(DP+PP+TP)非常强大,但它仍然有一个根本性的“浪费”:在数据并行的维度上,不同的副本(Data Parallel Replicas)之间存在着<strong class="key-term">完全相同的模型状态冗余</strong>。

例如,在一个 ‘dp_size=8, pp_size=4, tp_size=8‘ 的配置中:

- 由PP和TP组合后,每个GPU只负责整个模型 ‘1/(4\*8) = 1/32‘ 的计算和参数。

- 但是,这个 ‘1/32‘ 的模型分片,在8个数据并行副本中是<strong class="key-term">一模一样</strong>的。这意味着,我们依然为参数、梯度和优化器状态多占用了7倍的内存。

对于一个万亿参数的模型,即使是 ‘1/32‘ 的分片也可能非常巨大。消除这最后的冗余,就是ZeRO在这个组合中要完成的使命。

#### 8.5.3.2 架构和工作流程:ZeRO如何融入3D并行

ZeRO的操作完全发生在数据并行组(DP Group)内部。因此一样用立方体看待。

<figure data-latex-placement="H">
<img src="/images/112378145e.png" style="width:50.0%" alt="4D并行示意图" />
<figcaption>4D并行示意图</figcaption>
</figure>

<strong class="list-label">状态分区</strong>

考虑一个特定的模型块,它已经被PP和TP切分好了,位于坐标 ‘(pp_rank=p, tp_rank=t)‘ 的位置上。这个模型块的状态(参数、梯度、优化器状态)需要被存储。在传统的3D并行中,所有‘dp_rank‘从0到‘dp_size-1‘的GPU,在 ‘(p, t)‘ 这个位置上,都存储着一份<strong class="key-term">完整的、相同的</strong>模型块状态。

<strong class="key-term">引入ZeRO后</strong>:这个模型块的状态被进一步<strong class="key-term">切分成 ‘dp_size‘ 份</strong>,分散存储在整个DP组中。

- GPU ‘(dp=0, pp=p, tp=t)‘ 只存储这份状态的第1片。

- GPU ‘(dp=1, pp=p, tp=t)‘ 只存储这份状态的第2片。

- 以此类推。

<strong class="list-label">训练步骤的变化</strong>

这个改变深刻地影响了训练流程,尤其是在DP组内的通信模式:

- <strong class="list-label">前向传播 (ZeRO-Stage 3)</strong>:

  1.  在一个GPU ‘(d, p, t)‘ 开始计算它的模型块之前,它只拥有该模型块 ‘1/dp_size‘ 的参数。

  2.  它必须先在它的<strong class="key-term">DP组</strong>内发起一次 ‘All-gather‘ 操作。

  3.  通过这次通信,它从DP组的其他伙伴那里收集齐了完整的模型块参数。

  4.  然后,它和它的<strong class="key-term">TP组</strong>内的伙伴们一起,协同完成这一层的前向计算。

  5.  计算完成后,为了节省内存,它可以立即丢弃刚刚收集来的那部分不属于自己的参数。

- <strong class="list-label">后向传播</strong>:

  1.  与前向传播类似,在计算梯度前,需要通过 ‘All-gather‘ 确保拥有完整的参数。

  2.  计算完成后,每个GPU都得到了它负责的模型块的<strong class="key-term">部分梯度</strong>(因为TP)。

  3.  TP组内部会进行一次‘All-Reduce‘,使得TP组内每个GPU都拥有了关于这个模型块的<strong class="key-term">完整梯度</strong>。

- <strong class="list-label">梯度同步与分区 (取代传统DP的All-Reduce)</strong>:

  1.  现在,DP组内的每个GPU都有一份完整的梯度。

  2.  它们不再执行简单的‘All-Reduce‘来求平均。

  3.  取而代之,它们执行一次 ‘Reduce-Scatter‘。这个操作会一边将所有DP副本的梯度相加求平均,一边将结果<strong class="key-term">直接分区</strong>,并发送给对应的GPU。

  4.  操作结束后,GPU ‘(d, p, t)‘ 只会收到它所负责的、平均后的<strong class="key-term">梯度的一小片</strong>。

- <strong class="list-label">参数更新</strong>:

  1.  每个GPU使用它本地存储的一小片优化器状态和刚刚收到的一小片梯度,来更新它本地拥有的一小片模型参数。

### 8.5.4 专家并行/MOE并行

和前面的并行几乎完全不同,专家并行只针对推理阶段,它的思想是只利用和输入token最相关的几个专家模型(top-k)来进行推理最后加权输出。

但模型本身会有很多个专家来“顾问”,每次只用几个,来这样实现所谓”并行”。

<figure data-latex-placement="H">
<img src="/images/77ad80ca96.png" style="width:80.0%" alt="专家并行示意图" />
<figcaption>专家并行示意图</figcaption>
</figure>

CUDA 异构计算、主机与设备通信及设备内部通信见[CUDA 通信机制知识补充](/appendix#app-gpu-cuda)。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment2-systems)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 8.1 · 分布式训练与并行策略

<span id="read-9-1"></span>

分布式并行教程 <https://github.com/LambdaLabsML/distributed-training-guide?tab=readme-ov-file>

PyTorch分布式官方文档 <https://docs.pytorch.org/tutorials/beginner/dist_overview.html>

### 8.2 · AllReduce算法

<span id="read-9-2"></span>

AllReduce算法原理 <https://zhuanlan.zhihu.com/p/79030485>

AllReduce算法原理 <https://juejin.cn/post/7084135971687497759>

### 8.3 · DDP

<span id="read-9-3"></span>

DDP原理 <https://blog.csdn.net/my_name_is_learn/article/details/146468992>

桶排序算法 <https://www.runoob.com/w3cnote/bucket-sort.html>

### 8.4 · 混合并行(4D 并行)

<span id="read-9-4"></span>

DeepSeed原文 <https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/>

DeepSeed文档 <https://www.deepspeed.ai/tutorials/pipeline/>

混合并行教程 <https://blog.csdn.net/qq_25295605/article/details/143671968>

混合并行教程 <https://huggingface.co/docs/transformers/v4.15.0/en/parallelism>

张量模型并行 <https://zhuanlan.zhihu.com/p/622212228>
