---
title: 第六篇 · 指令微调与 RLHF
---

<p class="eyebrow">PART SIX · INSTRUCTION TUNING AND RLHF</p>

# 第六篇：指令微调与 RLHF

把 SFT 与偏好优化应用到通用指令任务。从序列组织、梯度累积和优化目标出发，完成基座评估、指令微调与 DPO，再比较三个阶段的结果。

<div class="learning-guide">

**贯穿本篇的路线**

理论：SFT、梯度累积与偏好优化 → 实验：统一评估、SFT 训练、DPO 训练与结果对照。

</div>

## 先理解理论，再对照实验

| 章节 | 阅读重点 |
| :--- | :--- |
| [第 14 章 · 指令微调与偏好对齐理论](./chapter-14) | Padding 与 Packing、梯度累积、奖励模型、RLHF、DPO 推导和方法比较，以及 IPO、KTO。 |
| [第 15 章 · 指令微调与 DPO 实验](./chapter-15) | 四类任务基线、SFT 数据加载与训练、梯度累积实现、DPO 训练和最终评估。 |

## 贯穿全篇的作业

本书将原 Assignment 5 的 Safety/RLHF 选做补充独立编号为 **Assignment 6**。实验使用 Llama-3.1-8B，依次建立基座评估、进行 SFT，再使用偏好数据训练 DPO。

- **评估任务：**MMLU、GSM8K、AlpacaEval、SimpleSafetyTests。
- **SFT 数据：**UltraChat-200K、SafetyTunedLlamas。
- **偏好数据：**Anthropic HH-RLHF。

模型、数据与工具的入口，以及划分和格式检查事项，统一见[模型、数据与评估入口](./resources)。

## 阅读实验记录

第 15 章保留了原稿中的训练配置、损失曲线、学习率曲线、DPO 训练曲线，以及基座、SFT、DPO 三个阶段的最终评估表。

::: tip 比较结果时保持口径一致
原稿注明最终评估更换了硬件，因此吞吐量变化不能直接归因于训练方法。阅读准确率时也要一起查看解析失败数量和数据划分；AlpacaEval、SimpleSafetyTests 的表格记录了样本数与吞吐量，不据此推断质量或安全性提升。
:::

## 按需查阅

- [梯度累积的理论依据](./chapter-14#part6-gradient-theory)。
- [IPO：让偏好差距有一个有限目标](./chapter-14#sec-alignment-ipo)。
- [KTO：从成对比较到单条好坏反馈](./chapter-14#sec-alignment-kto)。
- [第五篇 · 策略优化基础](../part-5/chapter-12)：衔接策略梯度、优势估计与 PPO。
- [训练流程与实验管理](../part-1/chapter-4)：回顾 checkpoint、日志与验证。

完成本篇后，可回到[全书首页](/)按六篇路线查阅，或使用搜索定位概念。
