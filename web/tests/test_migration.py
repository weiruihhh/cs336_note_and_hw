import json
import os
import re
import unittest
import tempfile
from pathlib import Path
from scripts.migrate import Migration

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('CS336_TEX_SOURCE', ROOT.parent / 'cs336第二版笔记')).resolve()

class ContentMigrationTests(unittest.TestCase):
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

    def test_every_code_block_is_unchanged(self):
        for number in range(1, 6):
            source = (SOURCE / f'{number}.tex').read_text(encoding='utf-8')
            expected = re.findall(r'\\begin\{fancycode\}(?:\[[^]]*\])?[^\S\n]*\n(.*?)\\end\{fancycode\}', source, re.S)
            page = ROOT / f'content/part-1/chapter-{number}.md'
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
