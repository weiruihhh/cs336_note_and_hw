<script setup lang="ts">
import { withBase } from 'vitepress'
const chapters = [
  { title: '分词器与 BPE', stage: '文本 → Token', detail: '从字节与词表开始，理解并实现 BPE。' },
  { title: 'Transformer 架构', stage: 'Token → 模型', detail: '连接嵌入、注意力与前馈网络。' },
  { title: '语言模型的训练', stage: '模型 → 学习', detail: '理解损失、优化器与训练组件。' },
  { title: '训练流程与实验管理', stage: '学习 → 实验', detail: '组织训练与验证，记录并保存模型。' },
  { title: '推理与采样', stage: '模型 → 文本', detail: '加载模型，用采样策略生成文本。' }
]
const systemsChapters = [
  { title: '性能分析与基准测试', stage: '测量 → 瓶颈', detail: '建立可信的计时基线，分析计算与显存开销。' },
  { title: 'FlashAttention 与 Triton 优化', stage: '瓶颈 → 算子', detail: '理解分块与重计算，用 Triton 实现注意力优化。' },
  { title: '分布式训练与并行策略', stage: '单卡 → 多卡', detail: '从集合通信到 DDP，理解分片与混合并行。' }
]
const alignmentChapters = [
  { title: '数学任务、SFT 与专家迭代', stage: '基线 → 监督微调', detail: '固定数学评估口径，连接 SFT 组件、采样筛选与专家迭代。' },
  { title: '策略优化基础：从策略梯度到 PPO', stage: '奖励 → 策略更新', detail: '从策略梯度与优势估计出发，理解 GAE、TRPO 和 PPO。' },
  { title: 'GRPO 原理与训练实验', stage: '组内比较 → 训练实验', detail: '梳理组内优势与训练循环，结合原稿的奖励统计和实验记录。' }
]
const preferenceChapters = [
  { title: '指令微调与偏好对齐理论', stage: '序列组织 → 偏好目标', detail: '从 Padding、Packing 与梯度累积，走到奖励模型、RLHF、DPO，以及 IPO、KTO。' },
  { title: '指令微调与 DPO 实验', stage: '统一基线 → 三阶段比较', detail: '建立四类任务基线，完成 SFT 与 DPO 训练，结合曲线和结果表复盘实验。' }
]
</script>

<template>
  <main class="textbook-home">
    <section class="book-hero" aria-labelledby="book-title">
      <div class="hero-copy">
        <p class="eyebrow">STANFORD CS336 · 中文学习笔记</p>
        <h1 id="book-title">从文本出发，<br>构建语言模型。</h1>
        <p class="hero-description">从分词、Transformer 与训练出发，走过系统优化、规模规律和数据处理，再进入推理强化学习与偏好对齐。把课堂里的概念，连接成可以运行的系统。</p>
        <div class="hero-actions">
          <a class="primary-link" :href="withBase('/part-1/chapter-1')">开始阅读 <span aria-hidden="true">↗</span></a>
          <a class="secondary-link" :href="withBase('/part-1/')">查看第一篇路线 <span aria-hidden="true">→</span></a>
        </div>
        <p class="edition-note">喂喂薇 · 第二版 · 全六篇 · 15 章 · Assignments 1–6</p>
      </div>
      <div class="hero-diagram" aria-label="从文本到生成的模型流程">
        <span class="diagram-caption">一段文本的旅程</span>
        <div class="text-sample">“语言模型如何学习？”</div>
        <span class="flow-line" aria-hidden="true">↓</span>
        <div class="token-strip"><span>语言</span><span>模型</span><span>如何</span><span>学习</span><span>？</span></div>
        <span class="flow-label">分词 · 嵌入 · 注意力</span>
        <div class="model-block"><span>Transformer</span><small>在上下文中，预测下一个 token</small></div>
        <span class="flow-line" aria-hidden="true">↓</span>
        <div class="output-sample">理解结构，然后亲手实现。<span class="text-cursor" aria-hidden="true"></span></div>
        <p class="diagram-note">流程示意 · token 划分与输出仅作展示</p>
      </div>
    </section>

    <section class="pathway-section" aria-labelledby="pathway-title">
      <div class="section-heading"><div><p class="eyebrow">PART ONE</p><h2 id="pathway-title">一条连续的学习路线</h2></div><p>分词器 → 模型 → 训练组件 → 训练管理 → 推理采样</p></div>
      <div class="chapter-pathway">
        <a v-for="(chapter, index) in chapters" :key="chapter.title" class="chapter-card" :href="withBase(`/part-1/chapter-${index + 1}`)">
          <div class="card-top"><span class="chapter-number">0{{ index + 1 }}</span><span class="card-arrow" aria-hidden="true">↗</span></div>
          <p class="chapter-stage">{{ chapter.stage }}</p><h3>{{ chapter.title }}</h3><p class="chapter-detail">{{ chapter.detail }}</p>
        </a>
      </div>
    </section>

    <section class="pathway-section" aria-labelledby="systems-title">
      <div class="section-heading"><div><p class="eyebrow">PART TWO</p><h2 id="systems-title">让训练系统跑得更快</h2></div><p>基准测量 → 单卡算子优化 → 多进程与多卡扩展</p></div>
      <div class="chapter-pathway systems-pathway">
        <a v-for="(chapter, index) in systemsChapters" :key="chapter.title" class="chapter-card" :href="withBase(`/part-2/chapter-${index + 6}`)">
          <div class="card-top"><span class="chapter-number">0{{ index + 6 }}</span><span class="card-arrow" aria-hidden="true">↗</span></div>
          <p class="chapter-stage">{{ chapter.stage }}</p><h3>{{ chapter.title }}</h3><p class="chapter-detail">{{ chapter.detail }}</p>
        </a>
      </div>
      <p class="systems-links"><a :href="withBase('/part-2/')">查看第二篇路线 →</a><a :href="withBase('/part-2/resources')">Assignment 2 · 实验任务与工具 →</a></p>
    </section>

    <section class="pathway-section" aria-labelledby="scaling-title">
      <div class="section-heading"><div><p class="eyebrow">PART THREE</p><h2 id="scaling-title">理解模型规模与计算预算</h2></div><p>规模变量 → 损失与计算约束 → IsoFLOPs 分析</p></div>
      <a class="chapter-card" :href="withBase('/part-3/chapter-9')">
        <div class="card-top"><span class="chapter-number">09</span><span class="card-arrow" aria-hidden="true">↗</span></div>
        <p class="chapter-stage">预算 → 规模选择</p><h3>Scaling Law</h3><p class="chapter-detail">理解幂律、Chinchilla 与 IsoFLOPs，梳理参数量、训练数据量和计算预算的关系。本篇以理论分析为主。</p>
      </a>
      <p class="systems-links"><a :href="withBase('/part-3/')">查看第三篇路线 →</a><a :href="withBase('/part-3/resources')">Assignment 3 · 作业与阅读 →</a></p>
    </section>

    <section class="pathway-section" aria-labelledby="data-title">
      <div class="section-heading"><div><p class="eyebrow">PART FOUR</p><h2 id="data-title">把网页文本变成训练数据</h2></div><p>理解数据 → 过滤与去重 → 编码训练 → 评估质量</p></div>
      <a class="chapter-card" :href="withBase('/part-4/chapter-10')">
        <div class="card-top"><span class="chapter-number">10</span><span class="card-arrow" aria-hidden="true">↗</span></div>
        <p class="chapter-stage">原始文本 → 训练语料</p><h3>数据处理与质量控制</h3><p class="chapter-detail">从网页提取、语种识别和隐私处理，走到 MinHash、去重与数据处理综合实验，连接过滤结果与训练评估。</p>
      </a>
      <p class="systems-links"><a :href="withBase('/part-4/')">查看第四篇路线 →</a><a :href="withBase('/part-4/resources')">Assignment 4 · 数据与实验 →</a></p>
    </section>

    <section class="pathway-section" aria-labelledby="alignment-title">
      <div class="section-heading"><div><p class="eyebrow">PART FIVE</p><h2 id="alignment-title">从数学任务走向推理强化学习</h2></div><p>任务与基线 → SFT 与专家迭代 → PPO → GRPO</p></div>
      <div class="chapter-pathway systems-pathway">
        <a v-for="(chapter, index) in alignmentChapters" :key="chapter.title" class="chapter-card" :href="withBase(`/part-5/chapter-${index + 11}`)">
          <div class="card-top"><span class="chapter-number">{{ index + 11 }}</span><span class="card-arrow" aria-hidden="true">↗</span></div>
          <p class="chapter-stage">{{ chapter.stage }}</p><h3>{{ chapter.title }}</h3><p class="chapter-detail">{{ chapter.detail }}</p>
        </a>
      </div>
      <p class="systems-links"><a :href="withBase('/part-5/')">查看第五篇路线 →</a><a :href="withBase('/part-5/resources')">Assignment 5 · 模型、数据与实验 →</a></p>
    </section>

    <section class="pathway-section" aria-labelledby="preference-title">
      <div class="section-heading"><div><p class="eyebrow">PART SIX</p><h2 id="preference-title">从指令微调到偏好对齐</h2></div><p>训练机制与目标 → 基座评估 → SFT → DPO → 结果对照</p></div>
      <div class="chapter-pathway two-chapter-pathway">
        <a v-for="(chapter, index) in preferenceChapters" :key="chapter.title" class="chapter-card" :href="withBase(`/part-6/chapter-${index + 14}`)">
          <div class="card-top"><span class="chapter-number">{{ index + 14 }}</span><span class="card-arrow" aria-hidden="true">↗</span></div>
          <p class="chapter-stage">{{ chapter.stage }}</p><h3>{{ chapter.title }}</h3><p class="chapter-detail">{{ chapter.detail }}</p>
        </a>
      </div>
      <p class="systems-links"><a :href="withBase('/part-6/')">查看第六篇路线 →</a><a :href="withBase('/part-6/resources')">Assignment 6 · 模型、数据与评估 →</a></p>
    </section>

    <section class="practice-section" aria-labelledby="practice-title">
      <div><p class="eyebrow">READ → BUILD → EXPERIMENT</p><h2 id="practice-title">读懂之后，动手实现。</h2><p>Assignment 1：实现分词器、模型和训练组件；完成训练与消融，保存可加载的 checkpoint，并生成文本。</p></div>
      <div class="resource-note"><h3>准备数据与工具</h3><p>TinyStories、OpenWebText、作业仓库及其处理顺序见资源入口；后续章节共用同一套分词与数据记录。</p><a :href="withBase('/part-1/resources')">打开资源入口 <span aria-hidden="true">→</span></a></div>
    </section>
  </main>
</template>
