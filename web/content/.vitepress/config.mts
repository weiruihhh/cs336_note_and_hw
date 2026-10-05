import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const chapters = [
  '分词器与 BPE', 'Transformer 架构', '语言模型的训练', '训练流程与实验管理', '推理与采样'
].map((text, i) => ({ text: `${i + 1}. ${text}`, link: `/part-1/chapter-${i + 1}` }))

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
      { text: '第一篇', link: '/part-1/', activeMatch: '/part-1/' },
      { text: '资源入口', link: '/part-1/resources' },
      { text: '基础附录', link: '/appendix' }
    ],
    sidebar: [{ text: '第一篇 · 从文本到生成', items: [
      { text: '阅读路线', link: '/part-1/' }, ...chapters,
      { text: '数据与作业资源', link: '/part-1/resources' }
    ] }, { text: '参考', items: [{ text: '基础附录', link: '/appendix' }] }],
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
