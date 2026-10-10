---
outline: [2, 3]
---

# 实验任务与资源入口

<strong class="list-label">作业入口：</strong>[Assignment 2 · Systems](https://github.com/stanford-cs336/assignment2-systems)。沿用第一篇的模型实现，按讲义准备基准测试、注意力优化与分布式训练代码。

<strong class="list-label">测试输入：</strong>算子正确性和性能测试可使用固定种子的随机张量，记录形状、dtype、设备与掩码。端到端训练比较沿用第一篇的 token 数据，无需为每种优化重新选择语料。

<strong class="list-label">工具准备：</strong>根据对应版本说明准备 PyTorch、CUDA 与 Triton；性能分析使用 Nsight Systems 和 NVTX。先确认单卡基线可运行，再增加多进程和多卡。

<strong class="note-label">应保存的文件：</strong>计时结果、误差检查、设备与环境记录、性能时间线、显存快照；各方法使用一致的测量范围。

## 实验目标与任务

<span id="sec-6-1"></span>

实验目标: 学习如何从<strong class="critical-term">系统底层</strong>优化提升<strong class="critical-term">单GPU的训练速度</strong>以及如何将训练扩展到<strong class="critical-term">多GPU</strong>。

<strong class="critical-term">每一部分的实验任务</strong>

1.  基准测试与性能分析框架

2.  Flash Attention 2 Triton内核编写

3.  分布式数据并行训练

4.  优化器状态分片
