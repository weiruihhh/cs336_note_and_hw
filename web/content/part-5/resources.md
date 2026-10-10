---
outline: [2, 3]
---

# 模型、数据与实验入口

<strong class="list-label">作业入口：</strong>[Assignment 5 · Alignment](https://github.com/stanford-cs336/assignment5-alignment)。确认使用版本的讲义、评分函数、数据获取说明和测试接口，再准备训练文件。

<strong class="list-label">基座模型：</strong>[Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)。保存模型与 tokenizer 的版本，使用与实验匹配的提示模板；基座模型与 Instruct 版本应明确区分。

<strong class="list-label">数学任务数据：</strong>原稿的训练与验证分别读取 `train.jsonl` 和 `validation.jsonl`。通过所用讲义的数据入口获取后，检查问题、答案字段及评分接口，分别用于 SFT/rollout 和固定验证。

<strong class="note-label">来源记录：</strong>原稿尚未记录这两个 JSONL 文件的准确下载地址、数据版本和划分过程，不能仅凭文件名认定它们与某个公开数据集相同。复现实验前先补齐这些信息；保留原始文件与转换脚本。

<strong class="note-label">应保存的文件：</strong>原始样本、提示模板、生成结果、答案解析结果、奖励统计、训练配置与 checkpoint。先完成零样本评估，再进入 SFT、专家迭代和 GRPO，参见[数学任务、SFT 与专家迭代](/part-5/chapter-11#guide-ch-12)及[GRPO 原理与训练实验](/part-5/chapter-13#guide-ch-12-grpo)。
