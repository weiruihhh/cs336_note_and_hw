---
outline: [2, 3]
---

# 数据与实验入口

<strong class="list-label">作业入口：</strong>[Assignment 4 · Data](https://github.com/stanford-cs336/assignment4-data)。下载与训练配置以所用讲义版本为准，正文已有一条脱离课程集群的获取路径。

Common Crawl · 待过滤文本  
从[Common Crawl 官方获取说明](https://commoncrawl.org/get-started)选择 crawl 批次，再获取相应文件清单和 WET 文件。正文的综合实验针对 WET 文本；如要实验 HTML 正文提取，需要对应的原始 HTML/WARC 数据，不能把两者混用。

Paloma · 验证语料  
从[AllenAI 的 Paloma 数据仓库](https://huggingface.co/datasets/allenai/paloma)核对配置、划分和字段，读取验证文本，再按训练代码要求使用配套 tokenizer 编码为验证 token 文件。

<strong class="note-label">处理顺序：</strong>获取原始文件、抽样检查、过滤与去重、编码、训练、固定验证集评估。记录 crawl 批次、过滤阈值、保留比例及 tokenizer 版本。

<strong class="note-label">复现边界：</strong>原稿使用自行处理的 Paloma 验证数据；它与课程提供的预处理文件不应直接视为完全一致。实验细节见[数据处理综合实验](/part-4/chapter-10#guide-ch-11)。
