---
outline: [2, 3]
---

# 数据与作业入口

<strong class="list-label">作业入口：</strong>[Assignment 1 · Basics](https://github.com/stanford-cs336/assignment1-basics)。先获取所用课程版本的仓库和讲义，按仓库说明准备环境、连接测试接口，再逐步实现各模块。

TinyStories · 分词与小规模训练  
从[原作者的数据仓库](https://huggingface.co/datasets/roneneldan/TinyStories)进入 Files，选择与讲义相符的训练、验证文本。先保留一份小样本供调试，再训练 tokenizer、保存词表与 merges，并将文本编码为 token 文件。

OpenWebText · 语料比较与训练  
从[项目下载页](https://skylion007.github.io/OpenWebTextCorpus/)获取语料，记录下载版本、解压路径和划分方法。使用同一套 tokenizer 编码训练与验证数据，避免改变评估口径。

<strong class="note-label">应保存的文件：</strong>原始文本、训练/验证划分记录、词表、merges、token 文件、配置及模型 checkpoint。

<strong class="note-label">对应正文：</strong>[下载与 BPE](/part-1/chapter-1#guide-ch-1)、[数据加载与训练](/part-1/chapter-4#guide-ch-4)、[生成与采样](/part-1/chapter-5#guide-ch-5)。
