<script setup lang="ts">
import { withBase } from 'vitepress'
const chapters = [
  { title: '分词器与 BPE', stage: '文本 → Token', detail: '从字节与词表开始，理解并实现 BPE。' },
  { title: 'Transformer 架构', stage: 'Token → 模型', detail: '连接嵌入、注意力与前馈网络。' },
  { title: '语言模型的训练', stage: '模型 → 学习', detail: '理解损失、优化器与训练组件。' },
  { title: '训练流程与实验管理', stage: '学习 → 实验', detail: '组织训练与验证，记录并保存模型。' },
  { title: '推理与采样', stage: '模型 → 文本', detail: '加载模型，用采样策略生成文本。' }
]
</script>

<template>
  <main class="textbook-home">
    <section class="book-hero" aria-labelledby="book-title">
      <div class="hero-copy">
        <p class="eyebrow">STANFORD CS336 · 中文学习笔记</p>
        <h1 id="book-title">从文本出发，<br>构建语言模型。</h1>
        <p class="hero-description">沿着一条完整的实现路线，理解分词、Transformer、训练与生成。把课堂里的概念，连接成可以运行的系统。</p>
        <div class="hero-actions">
          <a class="primary-link" :href="withBase('/part-1/chapter-1')">开始阅读 <span aria-hidden="true">↗</span></a>
          <a class="secondary-link" :href="withBase('/part-1/')">查看第一篇路线 <span aria-hidden="true">→</span></a>
        </div>
        <p class="edition-note">喂喂薇 · 第二版 · 第一篇试读 · Assignment 1</p>
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

    <section class="practice-section" aria-labelledby="practice-title">
      <div><p class="eyebrow">READ → BUILD → EXPERIMENT</p><h2 id="practice-title">读懂之后，动手实现。</h2><p>Assignment 1：实现分词器、模型和训练组件；完成训练与消融，保存可加载的 checkpoint，并生成文本。</p></div>
      <div class="resource-note"><h3>准备数据与工具</h3><p>TinyStories、OpenWebText、作业仓库及其处理顺序见资源入口；后续章节共用同一套分词与数据记录。</p><a :href="withBase('/part-1/resources')">打开资源入口 <span aria-hidden="true">→</span></a></div>
    </section>
  </main>
</template>
