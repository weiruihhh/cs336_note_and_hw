---
outline: [2, 3]
---

# 模型、数据与评估入口

<strong class="list-label">作业入口：</strong>[Assignment 5 的 Safety/RLHF 补充讲义](https://github.com/stanford-cs336/assignment5-alignment)。本书将这一选做补充独立编号为 Assignment 6。

<strong class="list-label">基座模型：</strong>[Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)。按模型页面完成访问申请与下载，保存模型和 tokenizer 版本。

<strong class="list-label">评估数据与工具</strong>

- [MMLU](https://huggingface.co/datasets/cais/mmlu)：知识选择题。核对学科配置、划分以及选项与答案字段，再适配正文评估格式。

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)：数学问答。保留原始答案，并单独实现最终答案提取与评分。

- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)：从项目说明获取评估输入、参考输出与评估配置；不同评估版本应分别记录。

- [SimpleSafetyTests](https://github.com/bertiev/SimpleSafetyTests)：从原项目获取测试提示，保留类别信息并按正文流程记录模型回复。

<strong class="list-label">训练数据</strong>

- [UltraChat-200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) 与[SafetyTunedLlamas](https://github.com/vinid/safety-tuned-llamas)：读取原始字段后，按讲义转换成所需的指令与回答样本；原始数据不必然已经是正文使用的单轮格式。

- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)：DPO 偏好数据。检查 chosen/rejected 的共同上下文，按讲义准备训练、验证划分。

<strong class="note-label">处理顺序：</strong>固定评估基线，准备 SFT 数据，微调并评估，再准备偏好数据做 DPO；全过程保持提示与评分口径一致。理论见[指令微调与偏好对齐理论](/part-6/chapter-14#guide-ch-13)，实验见[指令微调与 DPO 实验](/part-6/chapter-15#guide-ch-13-experiments)。
