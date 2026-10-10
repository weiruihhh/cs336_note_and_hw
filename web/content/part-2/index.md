---
title: 第二篇 · 训练系统与性能优化
---

<p class="eyebrow">PART TWO · SYSTEMS AND PARALLELISM</p>

# 第二篇：训练系统与性能优化

从测量训练瓶颈开始，理解 GPU 存储、混合精度、FlashAttention 与 Triton，再扩展到分布式通信和并行训练。

<div class="learning-guide">

**贯穿本篇的路线**

基准测量 → 单卡算子优化 → 多进程与多卡扩展。

</div>

## 按实验顺序阅读

| 章节 | 学习内容 |
| :--- | :--- |
| [06 · 性能分析与基准测试](./chapter-6) | 建立计时基线，使用性能分析工具定位瓶颈，记录显存开销。 |
| [07 · FlashAttention 与 Triton 优化](./chapter-7) | 理解注意力计算的访存瓶颈、分块与重计算，编写 Triton 内核。 |
| [08 · 分布式训练与并行策略](./chapter-8) | 理解集合通信、DDP、状态分片与混合并行，将训练扩展到多卡。 |

## 贯穿全篇的作业

Assignment 2：建立基准测试，优化注意力实现，实践梯度同步与状态分片；用正确性、耗时和显存共同评价结果。

算子测试使用固定形状的随机输入；端到端实验沿用第一篇数据。工具准备与原有任务清单见[实验任务与资源入口](./resources)。

## 随用随查的基础知识

- [混合精度训练](../appendix#app-gpu-precision)：浮点格式、权重备份与累加精度。
- [GPU 内存架构](../appendix#app-gpu-memory)：物理内存架构与逻辑内存层次。
- [CUDA 通信机制](../appendix#app-gpu-cuda)：主机与设备通信、设备内部通信。
- [计算量、显存与单位](../appendix#app-units)及[实验复现与结果记录](../appendix#app-repro)。

::: tip 阅读建议
先确认[第一篇的训练流程](../part-1/chapter-4)可以运行，再测量基线、修改实现并进行对照。比较性能时，保持输入形状、精度与测量范围一致，同时核对输出误差。
:::
