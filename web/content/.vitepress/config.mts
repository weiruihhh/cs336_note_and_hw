import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const chapters = [
  '分词器与 BPE', 'Transformer 架构', '语言模型的训练', '训练流程与实验管理', '推理与采样'
].map((text, i) => ({ text: `${i + 1}. ${text}`, link: `/part-1/chapter-${i + 1}` }))
const systemsChapters = [
  '性能分析与基准测试', 'FlashAttention 与 Triton 优化', '分布式训练与并行策略'
].map((text, i) => ({ text: `${i + 6}. ${text}`, link: `/part-2/chapter-${i + 6}` }))

export default defineConfig({
  lang: 'zh-CN',
  title: 'CS336 学习笔记',
  description: '从文本到生成，逐步实现语言模型。CS336 中文学习笔记与作业实践。',
  base: process.env.SITE_BASE || '/cs336_note_and_hw/',
  cleanUrls: true,
  appearance: 'light',
  lastUpdated: false,
  vue: { template: { compilerOptions: { isCustomElement: tag => tag.startsWith('mjx-') } } },
  head: [['meta', { name: 'theme-color', content: '#faf9f6' }]],
  markdown: { math: false, config(md) { md.use(mathjax3, { tex: { macros: { KeyTerm: ['\\textbf{#1}', 1] } } }) } },
  themeConfig: {
    siteTitle: 'CS336 / 学习笔记',
    nav: [
      { text: '阅读目录', activeMatch: '/part-[1-6]/', items: [
        { text: '第一篇 · 从文本到生成', link: '/part-1/' },
        { text: '第二篇 · 训练系统与性能优化', link: '/part-2/' },
        { text: '第三篇 · Scaling Laws', link: '/part-3/' },
        { text: '第四篇 · 数据处理与质量控制', link: '/part-4/' },
        { text: '第五篇 · 对齐与推理强化学习', link: '/part-5/' },
        { text: '第六篇 · 指令微调与 RLHF', link: '/part-6/' }
      ] },
      { text: '资源入口', items: [
        { text: 'Assignment 1 · 数据与作业', link: '/part-1/resources' },
        { text: 'Assignment 2 · 实验与工具', link: '/part-2/resources' },
        { text: 'Assignment 3 · 作业与阅读', link: '/part-3/resources' },
        { text: 'Assignment 4 · 数据与实验', link: '/part-4/resources' },
        { text: 'Assignment 5 · 模型与实验', link: '/part-5/resources' },
        { text: 'Assignment 6 · 模型与评估', link: '/part-6/resources' }
      ] },
      { text: '基础附录', link: '/appendix' }
    ],
    sidebar: [{ text: '第一篇 · 从文本到生成', items: [
      { text: '阅读路线', link: '/part-1/' }, ...chapters,
      { text: '数据与作业资源', link: '/part-1/resources' }
    ] }, { text: '第二篇 · 训练系统与性能优化', items: [
      { text: '阅读路线', link: '/part-2/' }, ...systemsChapters,
      { text: '实验任务与资源', link: '/part-2/resources' }
    ] }, { text: '第三篇 · Scaling Laws', items: [
      { text: '阅读路线', link: '/part-3/' },
      { text: '9. Scaling Law', link: '/part-3/chapter-9' },
      { text: '作业与阅读入口', link: '/part-3/resources' }
    ] }, { text: '第四篇 · 数据处理与质量控制', items: [
      { text: '阅读路线', link: '/part-4/' },
      { text: '10. 数据处理与质量控制', link: '/part-4/chapter-10' },
      { text: '数据与实验入口', link: '/part-4/resources' }
    ] }, { text: '第五篇 · 对齐与推理强化学习', items: [
      { text: '阅读路线', link: '/part-5/' },
      { text: '11. 数学任务、SFT 与专家迭代', link: '/part-5/chapter-11' },
      { text: '12. 策略优化基础：从策略梯度到 PPO', link: '/part-5/chapter-12' },
      { text: '13. GRPO 原理与训练实验', link: '/part-5/chapter-13' },
      { text: '模型、数据与实验入口', link: '/part-5/resources' }
    ] }, { text: '第六篇 · 指令微调与 RLHF', items: [
      { text: '阅读路线', link: '/part-6/' },
      { text: '14. 指令微调与偏好对齐理论', link: '/part-6/chapter-14' },
      { text: '15. 指令微调与 DPO 实验', link: '/part-6/chapter-15' },
      { text: '模型、数据与评估入口', link: '/part-6/resources' }
    ] }, { text: '基础附录', items: [
      { text: '附录总览', link: '/appendix' },
      { text: '数学基础', link: '/appendix#app-math' },
      { text: '深度学习基础', link: '/appendix#app-deep-learning' },
      { text: 'AI Infra 基础', link: '/appendix#app-infra' },
      { text: '信息论基础', link: '/appendix#app-information' },
      { text: '正则表达式与文本匹配', link: '/appendix#app-regex' }
    ] }],
    socialLinks: [{ icon: 'github', link: 'https://github.com/weiruihhh/cs336_note_and_hw' }],
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一节', next: '下一节' },
    sidebarMenuLabel: '章节', returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '主题', lightModeSwitchTitle: '切换到浅色', darkModeSwitchTitle: '切换到深色',
    search: { provider: 'local', options: {
      miniSearch: { options: { tokenize: (text: string) => Array.from(new Intl.Segmenter('zh-CN', { granularity: 'word' }).segment(text), item => item.segment).filter(word => /[\p{L}\p{N}]/u.test(word)) } },
      translations: { button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' }, modal: { displayDetails: '显示详细结果', resetButtonTitle: '清除搜索', backButtonTitle: '关闭搜索', noResultsText: '没有找到相关内容', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } } }
    } },
    footer: { message: '喂喂薇 · CS336 中文学习笔记 · 第二版', copyright: '仅供学习交流 · <a href="https://github.com/weiruihhh/cs336_note_and_hw/blob/main/LICENSE">CC BY-NC-SA 4.0</a>' }
  }
})
