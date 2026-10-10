---
outline: [2, 3]
---

# 第 6 章 · 性能分析与基准测试

<span id="guide-ch-7"></span>

<div class="custom-block info">

<p class="custom-block-title">本章学习导航</p>

<strong>前置知识：</strong>完成第一篇训练流程；区分计算量、时间与吞吐量，参见[计算与存储单位](/appendix#app-units)；精度知识见[混合精度训练](/appendix#app-gpu-precision)。

<strong>准备工作：</strong>准备可运行的模型与固定形状输入、GPU 和性能分析工具；记录设备型号、精度与软件版本。

<strong>本章任务：</strong>测量前向、反向及端到端耗时；使用 Nsight/NVTX 定位瓶颈；分析显存使用情况。

</div>

在我们进行优化之前,我们需要弄清楚程序或模型哪个地方还不够好,哪个地方还有可提升的空间。一般来讲可优化的点就是时间和空间(对应我们老生常谈的<strong class="key-term">时间复杂度和空间复杂度</strong>)。那么我们该如何观察到某一个部分的时间和空间的消耗程度呢？

讲义提供了三种性能评估方式：

1.  使用Python标准库进行简单的<strong class="key-term">端到端基准测试</strong>,以测量前向和后向传递的时间;

2.  使用 <strong class="key-term">NVIDIA Nsight Systems</strong>工具进行计算分析,以了解训练时间在CPU和GPU操作上的分布;

3.  分析<strong class="key-term">显存使用情况</strong>。

## 6.1 端到端简单性能评估测试

<span id="sec-7-1"></span>

[参考资料 6.1](/part-2/chapter-6#read-7-1)

为了找到程序的瓶颈点,实现快速迭代优化。我们最好以<strong class="key-term">脚本化</strong>的方式进行性能评估测试,例如我们可以把需要调节的超参数设置成命令行参数,然后通过脚本自动化地进行性能评估测试。

讲义强烈推荐使用Sbatch或Slurm平台上的<strong class="key-term">submitit</strong>工具来实现批量扫描。

### 6.1.1 Slurm平台上的submitit工具简要介绍

<span id="sec-7-1-1"></span> submitit工具也是一种批量调度脚本化的工具，和使用shell脚本不同，submitit和python绑定程度更高，它甚至是直接无缝衔接在python代码中的。

举个例子，比方说我们要实现一个简单的加法函数。

```python
import sys
import time
def add(a, b):
    time.sleep(10) # 模拟耗时
    return a + b
if __name__ == '__main__':
    # 从命令行读取参数
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    
    result = add(a, b)
    
    # 打印结果，Slurm会把这个输出保存到文件中
    print(f"计算结果是: {result}")
```

之后我们配合shell脚本实现。但如果我们使用了submitit工具，我们就可以直接在python代码中实现。

```python
import submitit
import time
# 我们的“菜谱”函数，和之前完全一样，甚至更纯粹
def add(a, b):
    time.sleep(10) # 模拟耗时
    return a + b
# 1. 实例化一个“助理”（Executor）
# AutoExecutor会自动检测到你正在Slurm环境下，并进行配置
executor = submitit.AutoExecutor(folder="slurm_logs") 
# 2. 设置“厨房”要求 (等同于 #SBATCH 参数)
executor.update_parameters(
    timeout_min=5,  # 任务最长运行5分钟
    cpus_per_task=1,
    mem_gb=1
)
print("准备向集群提交任务...")
# 3. 直接让“助理”把你的Python函数和参数送去做
job = executor.submit(add, 5, 7) 
print(f"任务已提交，任务ID是: {job.job_id}")
# 4. 直接在Python里等待并获取结果
result = job.result() # 这行代码会等待任务完成，然后把返回值给你
print(f"从集群拿回了结果: {result}")
```

在需要做<strong class="key-term">大量重复实验</strong>时（比如机器学习调参）。

假设我们要计算三组不同的加法：(5,7), (10,20), (100,200)。

```python
params = [(5, 7), (10, 20), (100, 200)]# 准备多组参数
# 一句话提交一个任务数组！
jobs = executor.map_array(add, *zip(*params)) # zip(*params) 会把 [(5,7),...] 变成 ([5,10,100], [7,20,200])
print(f"一次性提交了 {len(jobs)} 个任务！")
# 等待所有任务完成，并一次性取回所有结果
results = [job.result() for job in jobs]
print(f"所有任务都完成了，结果是: {results}")
# 输出将会是: 所有任务都完成了，结果是: [12, 30, 300]
```

submitit的一大核心优势是<strong class="key-term">控制器与工作任务的解耦</strong>，当执行 job = executor.submit(...) 时，发生了两件独立的事情：

- <strong class="list-label">控制器（你的本地脚本）</strong>: 它的任务是打包你的函数和参数，生成一个 Slurm 脚本，通过 sbatch 命令把它发送给 Slurm 调度器。一旦 Slurm 确认收到（返回一个任务ID），控制器的主要任务就完成了。之后它做的 job.result() 只是一个可选的“等待和接收”动作。

- <strong class="list-label">工作任务（集群上的脚本）</strong>: Slurm 收到请求后，会在某个计算节点上启动一个全新的、独立的进程。这个进程负责执行你的深度学习函数。它的生死只由 Slurm 和它自己决定，和你的“控制器”脚本没有任何关系。

### 6.1.2 前向和反向传播性能评估

<span id="sec-7-1-2"></span> 首先,通过计时前向和反向传递来对模型进行最基础的性能分析。由于仅需测量时间和内存消耗,因此将采用随机权重和数据。

<div class="custom-block info">

<p class="custom-block-title">延伸阅读 · CUDA异步调用</p>

当GPU去进行矩阵计算的时候,由于CUDA默认是<strong class="key-term">异步执行</strong>,CPU也会继续执行下面的任务,不会等待GPU执行完毕之后再去。这样做充分利用了计算资源,但在测量的时候由于是异步,所以不能够通过time.time()这种方式直接测量GPU计算时间(因为CPU会直接跳过GPU运算的步骤),要测量需要<strong class="key-term">torch.cuda.synchronize()</strong> 强制同步。

</div>

评估流程:

1.  给定<strong class="key-term">超参数</strong>(例如层数),初始化模型。

2.  生成随机数据批次。

3.  运行w个<strong class="key-term">预热步骤</strong>(在开始测量时间之前),然后记录n个步骤的执行时间(根据参数选择仅前向或前向和反向传递)。

    <div class="custom-block info">

    <p class="custom-block-title">延伸阅读 · 预热</p>

    端到端基准测试里面的预热并非学习率预热,而是希望机器在训练了几个步骤达到稳定之后再去测试时间,否则由于刚开始的随机权重和数据,导致损失值非常大,梯度也非常大,导致训练不稳定,时间测量不准确。

    </div>

4.  对于计时,可以使用Python的timeit模块(例如使用timeit函数,或者使用 timeit.default_timer(),它提供系统最高分辨率的时钟,因此比time()更适合作为基准测试的默认选项)。

5.  每个步骤后调用 torch.cuda.synchronize()。

## 6.2 Nsight Systems Profiler

<span id="sec-7-2"></span>

[参考资料 6.2](/part-2/chapter-6#read-7-2)

端到端基准测试无法揭示模型在前向传播和反向传播过程中具体消耗时间与内存的环节，因而难以发现特定组件的优化空间。为准确掌握程序<strong class="key-term">各组件（如函数）</strong>的运行耗时，可采用性能分析工具。执行分析器通过在函数开始和结束时插入监控点来检测代码，从而提供函数级别的详细执行统计（包括调用次数、该函数累计耗时等指标）。

像CProfile这样的Python性能分析器无法对 CUDA 内核进行性能分析，因为CUDA内核在GPU上是异步执行的。因此我们将利用 NVIDIA 提供的一个可以通过命令行工具nsys使用的性能分析器。

### 6.2.1 nsys 简介

<span id="sec-7-2-1"></span> nsys 全称 <strong class="key-term">NVIDIA Nsight Systems</strong>, 是一个专门记录 CPU、GPU 具体工作的工具，输出的结果类似甘特图、时序图这种，会配合 nvtx 使用清楚标注每一个事件的起始时间。

<strong class="critical-term">nsys 的工作流程（The Workflow）</strong>

1.  <strong class="note-label">命令行分析 (Profile on the Command Line)</strong>

    讲义中给出的命令是：

    uv run nsys profile -o result.nsys.rep python benchmark.py

    - <strong class="list-label">nsys profile:</strong> 这是核心命令，告诉 Nsight Systems 开始记录。

    - -o result.nsys.rep: -o 代表 output。它告诉 nsys 将所有记录下来的性能数据保存到一个名为 result.nsys.rep 的文件中。

    - python benchmark.py: 这是你想要分析的目标程序。

2.  <strong class="note-label">查看报告文件 (The .nsys-rep File)</strong>

3.  <strong class="note-label">图形界面分析 (Analyze in the GUI)</strong>

### 6.2.2 nvtx

NVTX API 本身是一套<strong class="key-term">“空接口”、“占位符”</strong>。默认情况下，代码中调用的 NVTX 函数<strong class="key-term">不会产生任何实际操作，也不会带来性能开销</strong>。

它的作用只在当程序被一个专门的<strong class="key-term">“开发者工具”（如性能分析器）</strong>启动时才会被激活。这时，这些 NVTX 调用就会被该工具拦截，并转交（重定向）给工具内部的相应功能来处理。

<strong class="note-label">比如</strong>一个运动员在跑道上跑步，而教练在赛道上某个位置（比如100米、400米）打个标记方便后续分析，但这些标记不会对运动员跑步产生直接影响。

由于 NVTX 只定义了调用的规范（比如“标记一个时间范围的开始”），而没有规定具体如何响应，因此不同的开发者工具可以根据自己的需要，自由决定如何利用这些调用信息。例如，有的工具可能用它来计时，有的则用它来触发日志记录。

一些常见工具配合 NVTX 使用的例子

- <strong class="key-term">打印一条消息到控制台</strong>

- <strong class="key-term">工具精确记录下每一次 NVTX 调用发生的具体时间点</strong>。当程序运行结束后，工具会将所有这些带有时间戳的事件收集起来，并以时间轴的形式图形化地展示出来。这样可以直观地看到代码中不同标记阶段的执行顺序、持续时间以及它们之间是否存在重叠。

- <strong class="key-term">统计 NVTX 记录下的时间</strong>

- <strong class="key-term">作为开关，在 NVTX 调用限定的范围内启用/禁用工具功能</strong>

- <strong class="key-term">作为一个中转站，将数据转发到其他工具出处理</strong>

#### 6.2.2.1 NVTX 提供的注解类型

- <strong class="list-label">Markers</strong>

  基础注解类型，主要传递消息，提供了一些选择<strong class="key-term">颜色、类别</strong>的参数(在 Ranges 里也有这些可选参数)，配合Nsight System这种可视化的性能分析工具使用。

  ```python
      nvtx.**mark**(
      message: str | None = None,
      color: str | int | None = 'blue',
      domain: str | None = None,
      category: str | int | None = None,
      payload: int | float | None = None,)
      
  ```

- <strong class="list-label">Ranges</strong>

  Ranges 用于标记程序在<strong class="key-term">一段时间内</strong>的活动，就像一对相关联的标记点（一个开始，一个结束）。它主要用来衡量一个函数或一段代码块的执行耗时。

  这也是 cs336 希望我们使用的。

  NVTX 提供了两种不同机制的范围，以适应不同的应用场景：

  1.  <strong class="critical-term">推入/弹出式范围 (Push/Pop Ranges):</strong>

      - <strong class="list-label">工作机制：</strong> 这种范围像一个<strong class="key-term">栈（Stack）</strong>一样工作，遵循“后进先出”（Last-In, First-Out）的原则。当你 ‘Push‘ 一个新范围时，它就被推入栈顶；当你 ‘Pop‘ 时，最顶端的范围被弹出。

      - <strong class="list-label">核心特点：</strong> 它们必须是<strong class="key-term">严格嵌套</strong>的，不能交叉重叠。‘Pop‘ 操作总是自动与同一线程上最近一次的 ‘Push‘ 操作配对。

      - <strong class="list-label">适用场景：</strong> 非常适合标记结构清晰、层层调用的函数或代码块。例如，‘main‘ 函数调用 ‘function_A‘，‘function_A‘ 又调用 ‘function_B‘。

      ```python
              nvtx.push_range("数据处理总流程")
                  // 步骤1: 加载数据
                  nvtx.push_range("加载数据")
                  ...加载数据的代码...
                  nvtx.pop_range() // "加载数据"范围结束
                  
                  // 步骤2: 处理数据
                  nvtx.push_range("处理数据")
                  ...处理数据的代码...
                  nvtx.pop_range() // "处理数据"范围结束
                  
                  // 步骤3: 保存结果
                  nvtx.push_range("保存结果")
                  ...保存结果的代码...
                  nvtx.pop_range() // "保存结果"范围结束
              nvtx.pop_range() // "数据处理总流程"范围结束
              主程序结束
              
      ```

      在 Nsight system 上的结果就是:

      \- 一个最长的、名为 “数据处理总流程” 的范围。 - 在它<strong class="key-term">内部</strong>，依次排列着三个互不重叠的短范围：“加载数据”、“处理数据”和“保存结果”。 - 你永远不会看到“加载数据”还没结束，“处理数据”就开始的情况。这就是严格嵌套。

  2.  <strong class="critical-term">开始/结束式范围 (Start/End Ranges):</strong> - <strong class="list-label">工作机制：</strong> ‘Start‘ 操作会返回一个唯一的<strong class="key-term">句柄（Handle）</strong>，这个句柄就像一个凭证,唯一对应。你必须在未来的某个时刻调用 ‘End‘，并将这个凭证传递给它，才能正确地关闭对应的范围。 - <strong class="list-label">核心特点：</strong> 它们可以<strong class="key-term">任意重叠</strong>，并且一个范围的开始和结束可以发生在<strong class="key-term">不同的线程</strong>上。 - <strong class="list-label">适用场景：</strong> 用于标记复杂的、异步的或并行的操作，这些操作的生命周期不是简单的嵌套关系。

      ```python
              主程序线程:
              ...
              // 步骤1: 加载数据
              // 步骤2: 并行处理数据
              // 为工作线程1的任务创建一个范围,并拿到凭证 handle1
              handle1 = nvtx.start_range("并行任务1")
              // 启动工作线程1,把 handle1 交给它
      
              // 为工作线程2的任务创建一个范围,并拿到凭证 handle2
              handle2 = nvtx.start_range("并行任务2")
              // 启动工作线程2,把 handle2 交给它
      
              // 主线程可以继续做其他事，或者等待线程结束...
      
              - ---------- 时间流逝 -----------
      
              工作线程1 (在另一个CPU核心上):
              ...执行任务1的代码...
              // 任务完成，用凭证 handle1 关闭对应的范围
              nvtx.end_range(handle1)
      
              工作线程2 (在另一个CPU核心上):
              ...执行任务2的代码...
              // 任务完成，用凭证 handle2 关闭对应的范围
              nvtx.end_range(handle2)
              
      ```

      在性能分析工具的时间轴上，你会看到：

      \- “并行任务1” 和 “并行任务2” 这两个范围是在主线程上<strong class="key-term">几乎同时开始</strong>的。 - 它们的执行过程在时间上是<strong class="key-term">重叠</strong>的。 - 它们的结束点发生在各自的工作线程上，并且结束时间<strong class="key-term">可能不同</strong>。

  3.  <strong class="critical-term">Resources</strong>

      <strong class="list-label">资源命名：</strong>

      对于新的线程，在Nsight里面分析的结果默认显示 python (TID: 54321) 这种线程名，资源命名就是给这个线程起一个名字方便观测处理。

      <strong class="list-label">资源追踪：</strong>

      是命名的扩展，对于一些标准的比如加锁、解锁的过程，NVTX只需命名，其他工具可以自动识别并给予、释放资源；

      对于一些不标准的，比如自己写的自旋锁，NVTX除了命名之外，还需要明确指出哪里加锁了，哪里解锁了。

<strong class="list-label">我的结果示例</strong>

<figure data-latex-placement="H">
<img src="/images/01cf358dcd.png" style="width:80.0%" alt="nvtx 结果示例" />
<figcaption>nvtx 结果示例</figcaption>
</figure>

从结果上来看，没想到裁剪梯度这一步居然这么耗时。。。

混合精度训练的原理、浮点数表示与实现方法见[混合精度训练](/appendix#app-gpu-precision)。

## 6.3 显存分析

[参考资料 6.3](/part-2/chapter-6#read-7-3)

此前我们主要探讨了计算的时间性能，接下来将聚焦于<strong class="key-term">显存</strong>这一语言模型训练与推理中的关键资源。PyTorch自带强大的内存分析工具Memory Profiler，可实时追踪内存分配动态。

对于训练和推理大语言模型（LLM）来说，显存（GPU Memory）和计算的时间性能一样，都是极其宝贵的资源。显存瓶颈常常会导致 “Out of Memory” (OOM) 错误，限制了我们能使用的模型大小和批量大小（batch size）。因此，精确地分析显存都用在了哪里，是性能优化的关键一步。

Memory Profiler的核心流程是：

1.  通过代码启动内存历史记录，运行需要分析的目标代码，

    ```python
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        
    ```

2.  然后将记录到的内存分配快照（snapshot）保存成一个 “pickle” 文件。

    ```python
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    ```

3.  停止记录。

    ```python
        torch.cuda.memory._record_memory_history(enabled=None)
        
    ```

4.  可视化工具分析。

    - <strong class="list-label">使用官方工具：</strong>访问图片中提到的在线工具 “<https://pytorch.org/memory_viz>”。

    - <strong class="list-label">加载快照文件：</strong>将你本地生成的 “memory_snapshot.pickle” 文件拖拽到这个网页上。

    - <strong class="list-label">解读分析结果：</strong>

      - <strong class="list-label">内存时间线 (Timeline)：</strong>你会看到一个图表，显示了总的预留显存（Reserved Memory）和已使用显存（Allocated Memory）随时间的变化。这可以帮助你快速定位显存峰值出现在哪个阶段。

      - 内存分配详情 (Allocations)：工具会列出每一次具体的内存分配事件。最关键的是，它会提供每一次分配的<strong class="key-term">大小（Size）</strong>和<strong class="key-term">代码堆栈追踪（Stack Trace）</strong>。通过堆栈追踪，你可以精确地知道是你的代码中的哪一行（例如，‘model.forward()‘ 里的某个特定操作）导致了这次内存分配，从而实现精确优化。

示例代码：

```python
import torch
import torch.nn as nn

# 确认你的环境支持 CUDA

if not torch.cuda.is_available():
print("CUDA is not available. This script requires a GPU.")
else:
device = torch.device("cuda")
# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1024, out_features=2048)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=2048, out_features=512)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# 2. 实例化模型和输入数据，并移动到 GPU
model = SimpleModel().to(device)
# 创建一个需要梯度的输入张量
input_tensor = torch.randn(256, 1024, device=device, requires_grad=True)

# 3. 开始记录内存历史
print("======== 开始内存性能分析 ========")
torch.cuda.memory._record_memory_history(max_entries=1000000)

# --- 我们想要分析的核心代码段 ---
# 执行一次前向传播
output = model(input_tensor)
# 计算一个标量损失
loss = output.sum()
# 执行一次反向传播
loss.backward()
# ---------------------------------

# 4. 保存内存快照到文件
snapshot_filename = "simple_model_snapshot.pickle"
torch.cuda.memory._dump_snapshot(snapshot_filename)

# 5. 停止记录
torch.cuda.memory._record_memory_history(enabled=None)

print(f"======== 性能分析结束, 快照已保存至 '{snapshot_filename}' ========")
print("请访问 <https://pytorch.org/memory_viz> 并上传该文件进行分析。")
```

## 参考文献与延伸阅读

<strong class="note-label">作业依据：</strong>[课程作业仓库与讲义](https://github.com/stanford-cs336/assignment2-systems)。使用前核对课程年份与仓库版本。

<strong class="note-label">延伸阅读：</strong>以下保留原稿收录的教程、文档与阅读说明；编号与正文中的阅读入口对应。

### 6.1 · 端到端简单性能评估测试

<span id="read-7-1"></span>

- <https://blog.csdn.net/gitblog_00002/article/details/148993434>

- <https://blog.csdn.net/weixin_40198079/article/details/129726302>

### 6.2 · Nsight Systems Profiler

<span id="read-7-2"></span>

<https://github.com/NVIDIA/NVTX>

<https://nvidia.github.io/NVTX/python/reference.html#nvtx.start_range>

### 6.3 · 显存分析

<span id="read-7-3"></span>

<https://hugging-face.cn/blog/train_memory>
