import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// VitePress can build a successful 404-only site when Markdown was not committed.
const root = fileURLToPath(new URL('../content/', import.meta.url))
const required = ['index.md', 'appendix.md',
  ...[1, 2, 3, 4, 5, 6].flatMap(part => [`part-${part}/index.md`, `part-${part}/resources.md`]),
  ...Array.from({ length: 5 }, (_, i) => `part-1/chapter-${i + 1}.md`),
  ...Array.from({ length: 3 }, (_, i) => `part-2/chapter-${i + 6}.md`),
  'part-3/chapter-9.md', 'part-4/chapter-10.md',
  ...Array.from({ length: 3 }, (_, i) => `part-5/chapter-${i + 11}.md`),
  'part-6/chapter-14.md', 'part-6/chapter-15.md']
const missing = required.filter(page => !existsSync(resolve(root, page)))
if (missing.length) {
  console.error('缺少网页正文，停止构建。请确认 Markdown 文件已复制并提交到 Git：\n' + missing.join('\n'))
  process.exit(1)
}
console.log(`网页源码检查通过：${required.length} 个必需页面均存在。`)
