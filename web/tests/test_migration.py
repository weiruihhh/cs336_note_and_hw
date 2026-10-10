import json
import os
import re
import unittest
import tempfile
from pathlib import Path
from scripts.migrate import Migration, link, CHAPTERS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('CS336_TEX_SOURCE', ROOT.parent / 'cs336第二版笔记')).resolve()

class ContentMigrationTests(unittest.TestCase):
    def test_appendix_keeps_legacy_repro_link_after_reorganization(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)/'appendix.md'
            Migration('appendix.tex').convert(r'\chapter{数学基础}\label{app:math}新版内容。', output)
            text = output.read_text(encoding='utf-8')
            self.assertIn('id="app-repro"', text)
            self.assertIn('/part-1/chapter-4', text)
            self.assertIn('新版内容', text)

    def test_partial_table_rules_do_not_become_cell_content(self):
        source = r"""\chapter{测试}
\begin{tabular}{lll}
任务 & 指标 & 分数 \\
GSM8K & 数量 & 1319 \\
\cline{2-3}
 & 准确率 & 18.95\% \\
\end{tabular}
"""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)/'table.md'
            Migration('16.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
            self.assertNotIn('2-3', text)
            self.assertIn('18.95%', text)
            self.assertIn('1319', text)

    def test_sixth_part_split_and_gradient_reference(self):
        self.assertEqual(link('guide:ch:13'), '/part-6/chapter-14#guide-ch-13')
        self.assertEqual(link('guide:ch:13-experiments'), '/part-6/chapter-15#guide-ch-13-experiments')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)/'links.md'
            Migration('16.tex').convert(r'\chapter{测试}理论依据见第\ref{part6:gradient-theory}节。入口见第\pageref{resources:part:6}页。', output)
            text = output.read_text(encoding='utf-8')
            self.assertIn('/part-6/chapter-14#part6-gradient-theory', text)
            self.assertIn('/part-6/resources', text)
            self.assertNotIn('pageref', text)

    def test_sft_prompt_and_special_tokens_are_literal(self):
        prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nInstruction:{prompt}\n\nResponse:{response}'
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)/'prompt.md'
            Migration('16.tex').convert('\\chapter{测试}\n<pad> <eos>\n\n'+prompt+'\n', output)
            text = output.read_text(encoding='utf-8')
            self.assertIn('```text\n'+prompt+'\n```', text)
            self.assertIn('&lt;pad&gt;', text)
            self.assertIn('&lt;eos&gt;', text)

    def test_small_reference_paragraph_keeps_link_text(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / 'reference.md'
            Migration('13.tex').convert(r'\chapter{测试}\par\noindent{\small\NoteLabel{延伸阅读：}\hyperref[read:12:3]{见本篇末资料 [3]（第\pageref{read:12:3}页）。}}\par', output)
            text = output.read_text(encoding='utf-8')
            self.assertIn('见本篇末资料', text)
            self.assertIn('/part-5/chapter-13#read-12-3', text)
            self.assertNotIn('pageref', text)

    def test_inline_math_cannot_become_setext_heading(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / 'formula.md'
            Migration('14.tex').convert('\\chapter{测试}\n$a\n=\nb$', output)
            self.assertIn('$a = b$', output.read_text(encoding='utf-8'))

    def test_fifth_part_split_chapters_and_supplement_links(self):
        for label, number in [('12', 11), ('12-policy', 12), ('12-grpo', 13)]:
            self.assertEqual(link('guide:ch:'+label), f'/part-5/chapter-{number}#guide-ch-{label}')
            page = ROOT/f'content/part-5/chapter-{number}.md'
            self.assertTrue(page.is_file())
            self.assertIn(f'# 第 {number} 章', page.read_text(encoding='utf-8'))
        self.assertEqual(link('supp:trpo'), '/part-5/chapter-13#supp-trpo')
        self.assertEqual(link('read:12:3'), '/part-5/chapter-13#read-12-3')
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'fifth-links.md'
            Migration('12.tex').convert(r'\chapter{测试}\ChapLink{12-policy}{策略优化}\ChapLink{13}{第六篇}，见第\ref{supp:gae}节。', output)
            text = output.read_text(encoding='utf-8')
        self.assertIn('/part-5/chapter-12#guide-ch-12-policy', text)
        self.assertIn('/part-5/chapter-13#supp-gae', text)
        self.assertIn('/part-6/chapter-14#guide-ch-13', text)

    def test_prompt_template_preserves_literal_tags_and_placeholder(self):
        prompt = 'A conversation between User and Assistant. Use <think> and <answer>.\nUser: {question}\nAssistant: <think>'
        source = '\\chapter{测试}\n格式 <think> </answer>\n\n' + prompt + '\n'
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'prompt.md'
            report = Migration('12.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        self.assertIn('```text\n'+prompt+'\n```', text)
        self.assertIn('&lt;think&gt;', text)
        self.assertIn('&lt;/answer&gt;', text)
        self.assertEqual(report['unresolved_raw_tex'], [])

    def test_inline_math_before_digits_remains_renderable(self):
        source = r'''\chapter{测试}
100万行$\times$600列，100$\sim$200。
'''
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'numeric-math.md'
            report = Migration('11.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        self.assertIn(r'$\times$ 600', text)
        self.assertIn(r'$\sim$ 200', text)
        self.assertEqual(report['math_count'], report['restored_math_count'])

    def test_fourth_part_chapter_resources_and_experiment(self):
        self.assertEqual(link('guide:ch:11'), '/part-4/chapter-10#guide-ch-11')
        self.assertEqual(link('read:11:1'), '/part-4/chapter-10#read-11-1')
        page = ROOT/'content/part-4/chapter-10.md'
        self.assertTrue(page.is_file())
        text = page.read_text(encoding='utf-8')
        for phrase in ['# 第 10 章', 'MinHash', '数据处理综合实验', '训练实验结果', '(/part-4/resources)']:
            self.assertIn(phrase, text)
        self.assertNotIn('pageref', text)
        resources = (ROOT/'content/part-4/resources.md').read_text(encoding='utf-8')
        for phrase in ['Common Crawl', 'Paloma', 'assignment4-data', '/part-4/chapter-10#guide-ch-11']:
            self.assertIn(phrase, resources)

    def test_scaling_law_box_forms_and_resource_page_reference(self):
        source = r'''\chapter{Scaling Law}
作业入口见第\pageref{resources:part:3}页。
\begin{ext}{幂律分布}{规模变化说明。}\end{ext}
\ext[]{}{关键发现 $N$ 与 $D$。}
'''
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'scaling.md'
            report = Migration('10.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        self.assertIn('# 第 9 章', text)
        self.assertIn('规模变化说明', text)
        self.assertIn('关键发现', text)
        self.assertIn('(/part-3/resources)', text)
        self.assertNotIn('pageref', text)
        self.assertEqual(text.count('custom-block info'), 2)
        self.assertEqual(report['unresolved_raw_tex'], [])

    def test_third_part_chapter_and_links(self):
        self.assertEqual(link('guide:ch:10'), '/part-3/chapter-9#guide-ch-10')
        self.assertEqual(link('sec:11.1'), '/part-3/chapter-9#sec-11-1')
        page = ROOT/'content/part-3/chapter-9.md'
        self.assertTrue(page.is_file())
        self.assertIn('IsoFLOPs', page.read_text(encoding='utf-8'))

    def test_spanning_table_cells_allow_markdown_math(self):
        source = r'''\chapter{测试}
\begin{tabular}{ll}
\multicolumn{2}{c}{总量 $N$} \\
$Q$ & $K$ \\
\end{tabular}
'''
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'table.md'
            Migration('8.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        self.assertRegex(text, r'<td[^>]*>\n\n[^<]*\$N\$\n\n</td>')

    def test_nested_layout_table_becomes_cell_text(self):
        source = r'''\chapter{测试}
\begin{tabular}{ll}
\multicolumn{2}{c}{总量 $N$} \\
$Q$ & \begin{tabular}{l}第一行\\第二行\end{tabular} \\
\end{tabular}
'''
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'nested-table.md'
            Migration('8.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        self.assertEqual(text.count('<table>'), 1)
        self.assertIn('第一行', text)
        self.assertIn('第二行', text)

    def test_tex_aliases_and_extension_keep_content(self):
        source = r'''\chapter{测试}
对于矩阵，假设其\term{奇异值分解}为已知。
\begin{Extension}{\KeyTerm{系数的来源}}{muon-coefficients}
\bluebf{计算梯度}，同时完成\term{权重衰减}和\term{梯度更新}。
\end{Extension}
'''
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as temporary:
            output = Path(temporary)/'fixture.md'
            Migration('3.tex').convert(source, output)
            text = output.read_text(encoding='utf-8')
        for phrase in ['奇异值分解', '系数的来源', '计算梯度', '权重衰减', '梯度更新']:
            self.assertIn(phrase, text)
        self.assertIn('custom-block info', text)
        self.assertNotIn('muon-coefficients', text)

    def test_all_five_chapters_are_present(self):
        for number in range(1, 6):
            page = ROOT / f'content/part-1/chapter-{number}.md'
            self.assertTrue(page.exists(), f'Chapter {number} has not been migrated')

    def test_second_part_chapters_and_resource_tasks(self):
        for number in range(6, 9):
            page = ROOT / f'content/part-2/chapter-{number}.md'
            self.assertTrue(page.exists(), f'Chapter {number} has not been migrated')
            self.assertIn(f'# 第 {number} 章', page.read_text(encoding='utf-8'))
        resources = (ROOT/'content/part-2/resources.md').read_text(encoding='utf-8')
        self.assertIn('实验目标与任务', resources)
        self.assertIn('优化器状态分片', resources)

    def test_cross_part_links_follow_actual_label_location(self):
        self.assertEqual(link('guide:ch:7'), '/part-2/chapter-6#guide-ch-7')
        self.assertEqual(link('sec:9.3'), '/part-2/chapter-8#sec-9-3')
        self.assertEqual(link('sec:8.1'), '/appendix#sec-8-1')
        self.assertEqual(link('sec:6.1'), '/part-2/resources#sec-6-1')

    def test_every_code_block_is_unchanged(self):
        for number, (part, chapter) in CHAPTERS.items():
            source = (SOURCE / f'{number}.tex').read_text(encoding='utf-8')
            expected = re.findall(r'\\begin\{fancycode\}(?:\[[^]]*\])?[^\S\n]*\n(.*?)\\end\{fancycode\}', source, re.S)
            page = ROOT / f'content/part-{part}/chapter-{chapter}.md'
            self.assertTrue(page.exists(), f'Chapter {number} has not been migrated')
            actual = []
            for match in re.finditer(r'^([ ]*)```python\n(.*?)^\1```[ ]*$', page.read_text(encoding='utf-8'), re.M | re.S):
                indent, body = match.groups()
                actual.append('\n'.join(line[len(indent):] if line.startswith(indent) else line for line in body.rstrip('\n').split('\n')))
            self.assertEqual([x.rstrip('\n') for x in expected], actual, f'Code differs in chapter {number}')

    def test_math_images_and_conversion_are_accounted_for(self):
        report = ROOT / 'reports/migration.json'
        self.assertTrue(report.exists(), 'Migration report does not exist yet')
        data = json.loads(report.read_text(encoding='utf-8'))
        for chapter in data['documents']:
            self.assertEqual(chapter['unresolved_raw_tex'], [], chapter['source'])
            self.assertEqual(chapter['unresolved_tokens'], [], chapter['source'])
            self.assertEqual(chapter['math_count'], chapter['restored_math_count'], chapter['source'])
            for filename in chapter['images']:
                self.assertTrue((ROOT / 'content/public' / filename).is_file(), filename)

    def test_no_placeholder_fragments_or_unconverted_code(self):
        for page in (ROOT / 'content').rglob('*.md'):
            text = page.read_text(encoding='utf-8')
            self.assertNotRegex(text, r'(?:ZZNOTETOKEN|\d{5}ZZ|class="fancycode")', str(page))

if __name__ == '__main__':
    unittest.main()
