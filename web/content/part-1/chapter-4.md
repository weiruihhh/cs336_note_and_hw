---
outline: [2, 3]
---

# 第 4 章 · 训练流程与实验管理

<span id="guide-ch-4"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>完成[模型实现](/part-1/chapter-2#guide-ch-2)与[训练组件](/part-1/chapter-3#guide-ch-3)；熟悉文件路径、配置和基本日志记录。

<strong>准备工作：</strong>准备第一篇的 token 文件、模型与优化器；实验记录字段可查[实验复现](/appendix#app-repro)。

<strong>本章任务：</strong>实现数据批次采样、checkpoint 保存和恢复、脚本化训练；开展归一化、位置编码与前馈网络的消融实验。

</div>

一般的深度学习训练流程就包括:

1.  <strong class="critical-term">数据集加载(DataLoader)</strong>

    代表<strong class="key-term">输入环节</strong>,在数据集进入模型之前,需要对数据集进行处理,比如归一化,分批,打乱等。

2.  <strong class="critical-term">模型保存(Checkpoint)</strong>

    代表<strong class="key-term">输出环节</strong>,在模型训练过程中,需要定期保存模型参数,以便避免在训练过程中因为各种原因导致模型参数丢失。

3.  <strong class="critical-term">训练循环(TrainLoop)</strong>

    这是<strong class="key-term">最终的脚本</strong>。它负责初始化所有组件,并按照正确的顺序组织整个训练流程,包括数据加载、模型计算、反向传播、参数更新、性能记录和模型保存等。

## 4.1 数据集加载(DataLoader)

<span id="sec-4-1"></span>

[参考资料 4.1](/part-1/chapter-4#read-4-1)

PyTorch提供了两个数据原语:torch.utils.data.DataLoader和torch.utils.data.Dataset,它们允许你使用预加载的数据集和你自己的数据。Dataset存储了样本及其相应的标签,DataLoader在Dataset周围包装了一个可迭代对象,以便于访问样本。

<strong class="critical-term">一个完整、符合 PyTorch 习惯的流程图</strong>:

<strong class="key-term">原始数据</strong>$\dashrightarrow$ <strong class="key-term">Dataset</strong> (读取样本 + 数据增强 transforms)$\dashrightarrow$ <strong class="key-term">DataLoader</strong> (batching, shuffle, num_workers, pin_memory...)$\dashrightarrow$ <strong class="key-term">训练循环</strong> for batch in DataLoader

### 4.1.1 数据增强的两种方式

#### 4.1.1.1 <strong>在线增强(Online Augmentation)</strong>

放在 Dataset 里,训练过程中每次取样本时随机增强,特点：

- 每次训练加载一个样本时才进行随机增强(如随机裁剪、翻转),不需要提前生成大量数据副本

- 每个 epoch 都可能看新的增强结,提升模型泛化

- 训练期间使用 CPU 对每个样本执行 transform,DataLoader 多进程可<strong class="key-term">并行加速</strong>

```python
class MyDataset(Dataset):
    def __init__(self, data_paths, transform=None):
            self.paths = data_paths
            self.transform = transform  # 数据增强在这里
    def __getitem__(self, idx):
        img = read_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)  # → 在 Dataset 做 augmentation
        return img
```

#### 4.1.1.2 <strong>离线增强 (Offline Augmentation)</strong>

提前对原始数据增强好，特点：

- Dataset 简单、<strong class="key-term">读取速度快</strong>

- 没有训练时增强的 CPU 开销 → <strong class="key-term">DataLoader 更轻松、GPU 利用率更高</strong>

- <strong class="list-label">非常适合:</strong>

  - 大规模数据

  - 训练频繁、增强重复多次

  - 多机训练、分布式训练(offline augmentation 避免重复增强)

  - 图像增强开销特别大的情况(如大图、复杂变换)

实际工程中通常会采用:

#### 4.1.1.3 <strong>混合模式(推荐)</strong>

- 离线做一些“耗时重、固定不变”的增强

  - 图像标准化

  - 统一尺寸 resize

  - 去噪、直方图均衡(如果需要)

  - 硬件加速无法轻易做的操作

- 在线做一些“轻量、随机”的增强

  - RandomCrop

  - RandomFlip

  - ColorJitter

  - RandomRotation

### 4.1.2 自定义 Dataset

torch.utils.data.Dataset 是一个抽象类,允许你从自己的数据源中创建数据集。

我们需要继承该类并实现以下两个方法:

- \_\_len\_\_(self):返回数据集中的样本数量。

- \_\_getitem\_\_(self, idx):通过索引返回一个样本。(有了这个函数才可以使用索引,比如dataset\[6\])

假设我们有一个简单的 CSV 文件或一些列表数据,我们可以通过继承 Dataset 类来创建自己的数据集。

```python
import torch
from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        """初始化数据集,path是数据集的路径,transform代表数据增强的手段"""
        self.paths = data_paths
        self.transform = transform  # 数据增强在这里
        self.X_data = read(path) #读取数据集
    self.Y_data = ...

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y
```

### 4.1.3 使用 DataLoader 加载数据

DataLoader是对Dataset的进一步封装,它主要是为了提升训练效率,常用的手段有 <strong class="key-term">batch 打包(batch_size)</strong>、<strong class="key-term">打乱顺序(shuffle)</strong>、<strong class="key-term">多进程加速(num_workers)</strong>、<strong class="key-term">内存优化(pin_memory)</strong>、<strong class="key-term">collate 规则(collate_fn)</strong>、<strong class="key-term">drop_last</strong>: 如果数据集中的样本数不能被 ‘batch_size‘ 整除,设置为 ‘True‘ 时,丢弃最后一个不完整的 batch。其中前两个最常用。

```python
dataloader = DataLoader(
dataset,
batch_size=32,
shuffle=True,
num_workers=4,
pin_memory=True
)

# 打印加载的数据
for epoch in range(1):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs: {inputs}')
        print(f'Labels: {labels}')
```

### 4.1.4 如果数据集过大无法载入内存怎么办？

我们可以使用名为mmap的Unix系统调用,它能将磁盘文件映射到虚拟内存,并在访问该内存位置时惰性加载文件内容。这样就能 “假装” 整个数据集已载入内存。

NumPy通过np.memmap(或使用np.load时设置mmap_mode=“r”标志,前提是数组最初通过np.save保存)实现该功能,它会返回一个类似numpy数组的对象,在访问时按需加载数据项。在训练期间从数据集(即numpy数组)采样时,务必以内存映射模式加载数据集(通过np.memmap或np.load的mmap_mode=’r’标志,具体取决于数组的保存方式)。

### 4.1.5 内存映射

数据并不是在磁盘上被直接访问的。CPU的核心原则是:<strong class="key-term">它只能直接处理位于物理内存(RAM)中的数据。 任何在磁盘上的数据,如果想被CPU计算、读取或修改,最终都必须被加载到物理内存中</strong>。

内存映射的快体现在它<strong class="key-term">极大地优化了数据从磁盘进入物理内存,并最终呈现给用户进程的这个过程</strong>。

<strong class="critical-term">传统 I/O (read()) 的数据流:</strong>两次copy

假设程序需要从磁盘读取1GB的文件到一块内存中。

1.  <strong class="list-label">第一次拷贝:</strong>从 <strong class="key-term">磁盘 -\> 内核缓冲区</strong>

    - 程序发起“read()”系统调用,导致程序从<strong class="key-term">用户态</strong>切换到<strong class="key-term">内核态</strong>。

    - 内核向磁盘控制器发出指令。

    - 磁盘控制器通过 <strong class="key-term">DMA(Direct Memory Access)</strong>,将文件数据从磁盘直接拷贝到内核地址空间的一块缓冲区里。这个缓冲区通常被称为<strong class="critical-term">页缓存(Page Cache)</strong>。

    - <strong class="note-label">关键</strong>:到目前为止,数据在内核里,你的用户程序还访问不到它。DMA完成了这次拷贝,CPU没有参与搬运数据,但它需要等待。

2.  <strong class="list-label">第二次拷贝:</strong>从 <strong class="key-term">内核缓冲区 -\> 用户缓冲区</strong>

    - 现在数据已经在内核的页缓存里了,“read()”系统调用的下一步,就是把这些数据从<strong class="key-term">内核空间</strong>拷贝到你程序在<strong class="key-term">用户空间</strong>指定的缓冲区(就是你传给“read()”函数的那个buffer指针)。

    - <strong class="list-label">这是性能瓶颈所在</strong>:这次拷贝是由<strong class="key-term">CPU</strong>亲自执行的。对于1GB的数据,CPU需要执行大量的‘mov‘指令,逐字节地把数据从一块内存搬到另一块内存。这会消耗大量的CPU周期,并且会严重污染CPU的缓存(Cache)。

    - 拷贝完成后,“read()”系统调用返回,程序从内核态切换回用户态。现在你的程序终于可以访问这1GB数据了。

<strong class="note-label">总结传统I/O</strong>:数据走了‘<strong class="key-term">磁盘 -\> 内核内存 -\> 用户内存</strong>‘的路径。发生了两次拷贝,其中一次是由CPU完成的、非常昂贵的内存拷贝。

<strong class="critical-term">内存映射(Memory Mapping)</strong>,

就是操作系统提供的一种机制,它允许一个进程将自己的<strong class="key-term">虚拟地址空间</strong>中的一部分,直接与一个文件或者设备进行<strong class="key-term">“链接”或“映射”</strong>。

<strong class="list-label">核心工作原理</strong>：

1.  <strong class="list-label">建立映射 (“mmap”系统调用)</strong>

    - 当一个进程调用“mmap”系统调用请求建立内存映射时,内核执行以下操作:

    - 在进程的虚拟地址空间中找到一段<strong class="key-term">连续的、未被使用</strong>的区域。

    - 为这段地址区域在进程的内存描述符(如Linux中的“vm_area_struct”中创建一个新的<strong class="key-term">条目</strong>,记录下映射的起始地址、长度、权限(读/写/执行)以及映射所关联的文件和偏移量。

    - <strong class="note-label">关键点</strong>:此时,内核<strong class="key-term">并不会</strong>立即从磁盘加载任何文件数据到物理内存中。它仅仅是建立了逻辑上的关联,即配置了相关的内核数据结构。这是一个<strong class="key-term">非常轻量级</strong>的操作。

2.  <strong class="list-label">首次访问与缺页中断 (Page Fault)</strong>

    - 当进程首次尝试访问(读或写)这段被映射的虚拟内存地址时,会发生以下一系列事件:

    - <strong class="key-term">MMU(内存管理单元)</strong> 硬件在转换虚拟地址时,会查询进程的页表。由于数据尚未加载,对应的PTE会标记为“不存在”(Absent)。

    - MMU无法完成地址翻译,于是触发一个硬件异常,即<strong class="key-term">缺页中断</strong>,并将控制权交回给操作系统内核。

    - 内核的缺页中断处理程序被激活。它会检查导致中断的虚拟地址,并查询进程的内存描述符,确认该地址属于一个合法的内存映射区域。

3.  <strong class="list-label">数据加载与页表更新</strong>

    - 确认是合法访问后,内核会执行真正的I/O操作:

    - 在物理内存(RAM)中分配一个<strong class="key-term">空闲的物理页帧</strong>。

    - 根据‘mmap‘时记录的文件信息和偏移量,计算出需要从磁盘加载的数据块位置。

    - 通过<strong class="key-term">磁盘驱动程序</strong>,将对应的一个页(通常为4KB)的数据从文件加载到刚刚分配的物理页帧中。

    - 更新进程的<strong class="key-term">页表</strong>,将触发中断的那个虚拟页的PTE指向这个新加载的物理页帧,并设置好相应的权限位(如“存在”、“可读”、“可写”等)。

    - 中断处理结束,控制权返回给用户进程。

4.  <strong class="list-label">透明的后续访问</strong>

    - 当中断处理程序返回后,导致中断的那条指令会被重新执行。这一次,<strong class="key-term">MMU</strong>能够成功地通过<strong class="key-term">页表</strong>将虚拟地址翻译为物理地址,进程便可以毫无知觉地访问到数据,仿佛它从一开始就在内存里一样。后续对同一页内其他地址的访问将直接通过MMU快速完成,不再触发中断。

<strong class="list-label">总结内存映射</strong>:数据走了‘<strong class="key-term">磁盘 -\> 内核内存(页缓存)</strong>‘的路径。然后,程序被“授权”直接访问这块内核内存。从始至终,数据只从磁盘到内存拷贝了<strong class="key-term">一次</strong>,而且<strong class="key-term">CPU没有参与任何批量数据的搬运工作</strong>。这就是为什么它有时被称为“零拷贝”技术(这里的“零”指的是没有发生CPU参与的内核态到用户态的拷贝)。

所以实际上内存映射的本质,就是通过<strong class="critical-term">虚拟内存机制</strong>,让用户进程能够安全、<strong class="critical-term">直接地访问到本属于内核空间的那块内存(页缓存)</strong>,从而消除了数据从内核空间到用户空间的冗余拷贝。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · <strong class="key-term"> 内存映射这么牛逼,这么省资源,那么干脆所有的读取方式都改成内存映射不就好了？</strong></p>

也不是,内存映射省资源主要省在一次CPU对文件的拷贝上,对付大文件比较合适,但对付小文件,拷贝省的资源可能还不够<strong class="key-term">建立映射的开销</strong>和<strong class="key-term">缺页中断的开销</strong>这两部分大,所以也是一个trade-off的问题。另外这个也会消耗虚拟地址的空间。

</div>

## 4.2 模型保存(Checkpoint)

<span id="sec-4-2"></span>

[参考资料 4.2](/part-1/chapter-4#read-4-2)

试想一下,假如我们在某次深度学习的训练中大概要训练个1000个epoch,得知训练预计完成要花费将近3天时间,我们好不容易训练了两天23个小时了,这时候突然停电了,服务器/电脑关机了,一切努力付诸东流,这下要气死了。

这个时候checkpoint的重要性就体现出来了。

checkpoint 可以看作深度学习训练里面的<strong class="key-term">“存档点”</strong>,它保存了深度学习某一时刻的模型权重。如果我们定时每100个epoch就保存一次模型权重,那么即使训练到第999个epoch的时候就停电了,那么我们起码还可以从第900个epoch上重新开始续训,而不用从0开始。

### 4.2.1 <strong>为什么 Checkpoint 至关重要？</strong>

Checkpoint 的重要性体现在以下几个关键场景:

- <strong class="note-label">训练中断与恢复</strong>

  - <strong class="list-label">场景:</strong> 深度学习训练非常耗时,服务器可能会意外重启、程序可能崩溃、GPU 资源可能被抢占。

  - <strong class="list-label">作用:</strong> 如果没有 Checkpoint,你需要从头开始训练,浪费大量时间和计算资源。有了 Checkpoint,你可以从最近的保存点继续训练,就像从游戏存档点复活一样,几乎无缝衔接。

- <strong class="note-label">保存最佳模型</strong>

  - <strong class="list-label">场景:</strong> 模型在训练过程中性能会波动。通常,我们关心的是在<strong class="key-term">验证集(Validation Set)</strong>上表现最好的那个模型,而不是训练到最后时刻的模型(后者可能已经开始过拟合)。

  - <strong class="list-label">作用:</strong> 通过在每个 Epoch(或固定步数)结束后评估模型在验证集上的性能(如准确率、损失值),我们可以只保存那个性能指标最好的模型状态。这是实际部署时最常用的策略。

- <strong class="note-label">迁移学习与微调</strong>

  - <strong class="list-label">场景:</strong> 你想在一个新的、数据量较小的任务上训练模型。从零开始训练一个大模型(如 ResNet, BERT)既困难又低效。

  - <strong class="list-label">作用:</strong> 我们可以加载一个在大型数据集(如 ImageNet, Wikipedia)上预训练好的模型 Checkpoint,然后在这个基础上针对我们的新任务进行<strong class="key-term">微调</strong>。这极大地加速了收敛速度并提升了模型性能。

- <strong class="note-label">模型部署与推理</strong>

  - <strong class="list-label">场景:</strong> 模型训练完成后,你需要将它部署到生产环境中提供服务(例如,用于图像识别、文本生成)。

  - <strong class="list-label">作用:</strong> 部署时加载的就是最终选定的最佳模型的 Checkpoint 文件,用它的权重来进行预测和推理。

### 4.2.2 <strong>一个 Checkpoint 文件里通常包含什么？</strong>

一个完备的 Checkpoint 文件不仅仅是模型的权重,它通常包含一个字典或对象,其中含有:

- <strong class="note-label">模型参数或权重</strong>:这是最核心的部分,即模型学习到的所有权重(weights)和偏置(biases)。在 PyTorch 中,这通常是 ‘model.state_dict()‘。

- <strong class="note-label">优化器状态</strong>:这对于恢复训练至关重要。像 Adam、SGD with Momentum 这类优化器,它们内部会维护一些状态(如动量、学习率适应性调整的梯度平方均值等)。如果不保存优化器状态,恢复训练时优化器会重置,相当于重新开始,会影响训练的连续性。在 PyTorch 中,这是 ‘optimizer.state_dict()‘。

- <strong class="note-label">训练元数据</strong>:

  - <strong class="list-label">当前轮次 (Epoch)</strong>:记录训练进行到了第几轮。

  - <strong class="list-label">迭代步数</strong>:记录总共迭代了多少步。

  - <strong class="list-label">损失值</strong>:当前或历史最佳的训练/验证损失。

  - <strong class="list-label">其他指标</strong>:如准确率、F1 分数等。

如果说训练之后的结果只需要用来<strong class="key-term">推理(inference)或部署</strong>,而不再做“从某个中断点继续训练”的话,那么只保存模型的 <strong class="key-term">state_dict</strong>(即参数权重)就够了。例如:

```python
torch.save(model.state_dict(), "model_weights.pth")
```

然后在部署或加载时:

```python
model = MyModel()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
```

但如果担心停电导致前功尽弃,需要断电续训的话,就最好把 模型权重(<strong class="key-term">model.state_dict()</strong>)、优化器状态(<strong class="key-term">optimizer.state_dict()</strong>) —— 包括动量、Adam 的 m/v、学习率状态等 以及当前训练轮次或步数(epoch 或 global_step)都保存下来。

```python
checkpoint = {
'epoch': epoch,
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'loss': some_loss_value,
# 可能也保存 scheduler 和其他东西
}
torch.save(checkpoint, "checkpoint.pth")
```

加载恢复:

```python
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 4.3 脚本化训练

<span id="sec-4-3"></span>

等到把所有的模块都整理完毕了,到最后我们进行实验时的变量其实主要就是超参数+不同数据集了,这个时候使用脚本化的训练能大大方便我们的实验。

因此我们的目标:<strong class="key-term">一个 train.py,改少量参数就能换模型 / 换数据 / 换实验</strong>。

我的经验一般是一个config文件(可以是<strong class="key-term">yaml或者json格式</strong>用来保存所有的超参数,方便修改,而不是在代码里面写死导致修改极不方便。然后用<strong class="key-term">argparse</strong>配合直接在输入命令的时候就能控制变量:

比如

```python
def get_args():
    parser = argparse.ArgumentParser()
    
    # 基本配置
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 数据 & 模型
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')  # 预留给后面多模型选择
    
    # 设备 & 保存
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--resume', type=str, default='')  # 断点续训
    
    args = parser.parse_args()
    return args
```

## 4.4 消融实验 Ablation Studies

<span id="sec-4-4"></span>

消融实验是一种非常常用的实验方法,通过逐步移除模型中的某些组件或修改某些超参数,来研究这些组件或超参数对模型性能的影响。

消融实验的目的是为了证明模型中某个组件或超参数的重要性,或者证明模型中某个组件或超参数的无效性。绝大部分有关修改模型组件的文章里面都会有一章关于消融实验的讨论,以此来证明自己修改的组件或超参数是有用的。

一个严谨的消融实验通常遵循以下步骤:

1.  <strong class="list-label">第一步:</strong>建立一个强大的基线(Baseline)

    - 首先,你需要有一个完整、性能表现最好的模型。这个模型包含了你想要研究的所有组件和技术。你需要在一个标准的评估数据集上测试它,并记录下关键性能指标(如准确率、精确率、召回率、mAP等),这将作为后续所有比较的基准。

2.  <strong class="list-label">第二步:</strong>确定要“消融”的组件

    - 列出你想要研究其贡献度的所有组件。这些组件可以是:

      - <strong class="list-label">网络架构的一部分:</strong>如注意力模块、一个特定的卷积层块(Bottleneck Block)、残差连接(Skip Connection)。

      - <strong class="list-label">数据增强技术:</strong>如 Mixup, CutMix, Mosaic 等。

      - <strong class="list-label">损失函数的某个部分:</strong>如在一个复合损失函数中去掉某个特定的损失项。

      - <strong class="list-label">特定的训练策略:</strong>如学习率预热(Warm-up)。

      - <strong class="list-label">预训练模型的应用:</strong>比如比较“使用预训练权重”和“从零开始训练”的性能差异。

3.  <strong class="list-label">第三步:</strong>系统性地进行实验

    - 从基线模型开始,每次只移除或替换一个组件,然后重新训练模型,并在相同的测试集上评估性能。

4.  <strong class="list-label">第四步:</strong>结果分析与呈现

    - 将所有实验结果整理成一个清晰的表格,这是学术论文中最常见的形式。表格会直观地展示移除不同组件后性能的下降情况。

在我们这个作业里,它要求的消融实验是3部分:

1.  <strong class="list-label">消融实验1:</strong>层归一化

    通常认为层归一化对Transformer训练的稳定性至关重要。讲义要求从每个Transformer模块中移除RMSNorm看看会发生什么。

    另一个关于层归一化的实验是前归一化与后归一化的对比,讲义要求实现前归一化与后归一化的Transformer模块,并比较两者的性能。

2.  <strong class="list-label">消融实验2:</strong>位置嵌入对模型性能的影响

    我们接下来将研究位置嵌入对模型性能的影响。具体而言,我们将对比基础模型(使用RoPE)与完全不使用位置嵌入(NoPE)的模型。研究发现,仅使用解码器的Transformer模型(即我们实现的因果掩码模型)理论上无需显式提供位置嵌入信息,即可推断相对或绝对位置信息。接下来我们将通过实证测试NoPE与RoPE的性能差异。

3.  <strong class="list-label">消融实验3:</strong>SwiGLU与SiLU对比

    讲义要求通过比较使用SiLU激活函数但未采用门控线性单元(GLU)的前馈网络与使用SiLU激活函数但未采用门控线性单元的前馈网络的性能,验证门控机制在前馈网络中的重要性。需要说明的是,在我们的SwiGLU实现中,将内部前馈层的维度设定为约$d_{ff}=\frac{3}{8}d_{model}$(同时确保$d_{ff}$模64=0,以便充分利用GPU张量核心)。在您的FFNSiLU实现中,应将$d_{ff}$设置为 $4 \times d_{model}$,以大致匹配SwiGLU前馈网络的参数数量(该网络具有三个而非两个权重矩阵)。

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 4.1 · 数据集加载(DataLoader)

<span id="read-4-1"></span>

<https://www.runoob.com/pytorch/pytorch-dataset-dataloader.html?utm_source=chatgpt.com>

<https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html?utm_source=chatgpt.com>

### 4.2 · 模型保存(Checkpoint)

<span id="read-4-2"></span>

<https://blog.csdn.net/PolarisRisingWar/article/details/145529106>

<https://juejin.cn/post/7435680251878015016>
