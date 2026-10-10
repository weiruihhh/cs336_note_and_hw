---
title: 第三篇 · Scaling Laws
---

<p class="eyebrow">PART THREE · SCALING LAWS</p>

# 第三篇：Scaling Laws

理解参数量、训练 token 数和计算预算的关系，比较幂律拟合与 IsoFLOPs 分析如何支持规模选择。

<div class="learning-guide">

**贯穿本篇的路线**

明确规模变量 → 理解损失与计算约束 → 阅读 IsoFLOPs 分析。

</div>

## 本篇阅读路线

本篇对应[第 9 章 · Scaling Law](./chapter-9)，依次介绍：

| 主题 | 阅读重点 |
| :--- | :--- |
| 幂律与规模变量 | 参数量 N、训练 token 数 D、计算预算 C。 |
| Chinchilla | 损失公式中的各项，以及模型规模、数据量与计算预算之间的关系。 |
| Kaplan 单变量公式 | 分别从参数量、数据量和计算量观察损失的变化。 |
| IsoFLOPs | 在固定计算预算下比较配置，理解曲线最低点及幂律拟合的用途。 |

## 作业与当前进度

本篇对应 Assignment 3。由于缺乏对应资源，原稿尚未展开实验，当前以理论分析为主。

作业仓库、实验记录要求与后续练习建议见[作业与阅读入口](./resources)。目前没有完整作业实现、拟合结果或实验日志。

## 阅读前可以回看

- [语言模型的训练](../part-1/chapter-3)：损失函数与训练组件。
- [性能分析与基准测试](../part-2/chapter-6)：理解测量口径与计算成本。
- [计算量、显存与单位](../appendix#app-units)：区分参数量、token 数和 FLOPs。
- [实验复现与结果记录](../appendix#app-repro)：后续整理实验点时统一配置与评估口径。
