"""One-time migration of the second-edition notes. Source files are read-only."""
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('CS336_TEX_SOURCE', ROOT.parent / 'cs336第二版笔记')).resolve()
PANDOC = os.environ.get('PANDOC') or shutil.which('pandoc') or str(ROOT / '.tools/pypandoc/files/pandoc.exe')
SEMANTICS = {'KeyTerm': 'key-term', 'CriticalTerm': 'critical-term',
             'ListLabel': 'list-label', 'NoteLabel': 'note-label',
             'KeyFormula': 'key-formula', 'term': 'key-term',
             'bluebf': 'list-label', 'redbf': 'critical-term',
             'pinkbf': 'key-formula', 'greenbf': 'note-label'}
CHAPTERS = {**{n: (1, n) for n in range(1, 6)}, **{n: (2, n - 1) for n in range(7, 10)}, 10: (3, 9), 11: (4, 10), 12: (5, 11), 13: (5, 12), 14: (5, 13), 15: (6, 14), 16: (6, 15)}
SOURCE_ROUTES = {f'{n}.tex': f'/part-{part}/chapter-{chapter}'
                 for n, (part, chapter) in CHAPTERS.items()}
SOURCE_ROUTES.update({'6.tex': '/part-2/resources', 'appendix.tex': '/appendix'})

def label_routes():
    result = {}
    for filename, route in SOURCE_ROUTES.items():
        source = re.sub(r'(?<!\\)%[^\n]*', '', (SOURCE/filename).read_text(encoding='utf-8'))
        for label in re.findall(r'\\(?:label|ReferenceGroup)\{([^}]+)\}', source):
            result[label] = route
    return result

LABEL_ROUTES = None

def group(text, pos, opening='{', closing='}'):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != opening:
        raise ValueError(f'Expected {opening} near {text[pos:pos+90]!r}')
    start, depth = pos + 1, 1
    pos += 1
    while pos < len(text):
        if text[pos] == '\\':
            pos += 2
            continue
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    raise ValueError(f'Unclosed group near {text[start:start+80]!r}')

def macro(text, name, count, fn):
    pattern = re.compile(r'\\' + re.escape(name) + r'(?![A-Za-z])')
    while True:
        match = pattern.search(text)
        if not match:
            return text
        pos, args = match.end(), []
        for _ in range(count):
            value, pos = group(text, pos)
            args.append(value)
        text = text[:match.start()] + fn(*args) + text[pos:]

def anchor(label):
    return re.sub(r'[^a-zA-Z0-9_-]', '-', label)

def link(label):
    global LABEL_ROUTES
    if LABEL_ROUTES is None:
        LABEL_ROUTES = label_routes()
    if label in LABEL_ROUTES:
        return LABEL_ROUTES[label] + '#' + anchor(label)
    if label.startswith('app:'):
        return '/appendix#' + anchor(label)
    return '#' + anchor(label)

class Migration:
    def __init__(self, filename):
        self.filename = filename
        self.tokens = {}
        self.math = []
        self.code = []
        self.images = []
        self.raw = []

    def token(self, value):
        key = f'ZZNOTETOKEN{len(self.tokens):05d}ZZ'
        self.tokens[key] = value
        return key

    def block(self, value):
        return '\n\n' + self.token(value) + '\n\n'

    def box(self, title, body, kind='info'):
        # HTML boundaries allow nested boxes without ambiguous fence lengths.
        return self.block(f'<div class="custom-block {kind}">') + self.block(
            '<p class="custom-block-title">' + title + '</p>') + body + self.block('</div>')

    def protect_code(self, text):
        def replace(m):
            code = m.group(1).rstrip('\n')
            self.code.append(code)
            return self.block('```python\n' + code + '\n```')
        return re.sub(r'\\begin\{fancycode\}(?:\[[^]]*\])?[^\S\n]*\n(.*?)\\end\{fancycode\}', replace, text, flags=re.S)

    def protect_math(self, text):
        text = macro(text, 'FitDisplayMath', 1, lambda x: '\\[' + x + '\\]')
        pattern = re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\\)\$\$(.*?)\$\$|(?<!\\)\$(.*?)(?<!\\)\$|\\begin\{(equation\*?|align\*?|gather\*?)\}(.*?)\\end\{\5\}', re.S)
        def replace(m):
            raw = m.group(0)
            self.math.append(raw)
            inline = m.group(2) is not None or m.group(4) is not None
            if m.group(5):
                content = m.group(6)
                if m.group(5).startswith('align'):
                    content = '\\begin{aligned}' + content + '\\end{aligned}'
            else:
                content = next(v for v in m.groups()[:4] if v is not None)
            content = content.strip()
            if inline:
                # TeX treats source line breaks as spaces; Markdown may parse them as headings.
                content = re.sub(r'\s*\n\s*', ' ', content)
            rendered = '$' + content + '$' if inline else '$$\n' + content + '\n$$'
            # Markdown math rejects a closing dollar directly followed by a digit
            # (currency heuristic), while TeX allows expressions like 100$\sim$200.
            if inline and text[m.end():m.end()+1].isdigit():
                rendered += ' '
            return self.token(rendered) if inline else self.block(rendered)
        return pattern.sub(replace, text)

    def prepare(self, original):
        text = self.protect_code(original)
        # Keep prompt data literal: Pandoc otherwise consumes {question} as a TeX group.
        text = re.sub(r'(?m)^A conversation between User and Assistant\.[\s\S]*?^Assistant: <think>[^\S\n]*(?=\n|$)',
                      lambda m: self.block('```text\n' + m.group(0).rstrip() + '\n```'), text)
        text = re.sub(r'(?m)^[ \t]*Below is an instruction that describes a task\.[\s\S]*?^[ \t]*Response:\{response\}[^\S\n]*(?=\n|$)',
                      lambda m: self.block('```text\n' + '\n'.join(line.strip() for line in m.group(0).strip().splitlines()) + '\n```'), text)
        text = re.sub(r'</?(?:think|answer|pad|eos|endoftext)>' , lambda m: self.token(html.escape(m.group(0))), text)
        text = re.sub(r'(?<!\\)%[^\n]*', '', text)
        if self.filename == 'appendix.tex':
            text = text[text.index('\\chapter{'):]
        text = self.protect_math(text)
        # Page-layout declarations can make Pandoc swallow their following group.
        text = re.sub(r'\\(?:noindent|small)(?![A-Za-z])', '', text)
        for command, cls in SEMANTICS.items():
            tag = 'span' if cls == 'key-formula' else 'strong'
            def semantic(x, cls=cls, tag=tag):
                is_display = cls == 'key-formula' and any(
                    key in x and value.startswith('$$') for key, value in self.tokens.items())
                if is_display:
                    return self.block(f'<div class="{cls}">') + x + self.block('</div>')
                return self.token(f'<{tag} class="{cls}">') + x + self.token(f'</{tag}>')
            text = macro(text, command, 1, semantic)
        text = macro(text, 'ChapterGuide', 3, lambda a,b,c:
                     self.box('本章学习导航', '\\textbf{前置知识：}' + a + '\n\n\\textbf{准备工作：}' + b + '\n\n\\textbf{本章任务：}' + c))
        text = macro(text, 'ChapLink', 2, lambda n,t:
                     '\\href{' + link('guide:ch:'+n) + '}{' + t + '}' if link('guide:ch:'+n).startswith('/')
                     else t + '（后续篇章，网页版暂未收录）')
        text = macro(text, 'AppLink', 2, lambda n,t: '\\href{' + link('app:'+n) + '}{' + t + '}')
        text = re.sub(r'第\\pageref\{resources:part:([123456])\}页',
                      lambda m: '\\href{/part-' + m.group(1) + '/resources}{本篇资源入口}', text)
        text = re.sub(r'（第\\pageref\{read:[^}]+\}页）', '', text)
        text = re.sub(r'第\\ref\{(supp:[^}]+)\}节',
                      lambda m: '\\href{' + link(m.group(1)) + '}{对应补充节}', text)
        text = re.sub(r'第\\ref\{(part6:[^}]+)\}节',
                      lambda m: '\\href{' + link(m.group(1)) + '}{对应理论节}', text)
        text = macro(text, 'ReferencePointer' , 2, lambda a,b: '\\href{'+link(a)+'}{参考资料 '+b+'}')
        text = macro(text, 'ReferenceGroup', 3, lambda a,b,c: '\n\\subsection*{'+b+' · '+c+'}\n\\label{'+a+'}\n')
        text = macro(text, 'label', 1, lambda x: self.token('<span id="'+anchor(x)+'"></span>'))
        text = re.sub(r'\\ref\*?\{([^}]+)\}', lambda m: m.group(1).split(':')[-1] if not m.group(1).startswith('app:') else '', text)
        text = re.sub(r'\\hyperref\[([^]]+)\]', lambda m: '\\href{'+link(m.group(1))+'}', text)
        text = text.replace('\\begin{ext}', '\\ext').replace('\\end{ext}', '')
        text = re.sub(r'\\ext\s*\[\s*\]', r'\\ext', text)
        text = macro(text, 'ext', 2, lambda a,b: self.box('延伸阅读'+(' · '+a if a.strip() else ''), b))
        def extension(m):
            title, end = group(m.group(1), 0)
            _identifier, end = group(m.group(1), end)
            return self.box('延伸阅读' + (' · ' + title if title.strip() else ''), m.group(1)[end:])
        text = re.sub(r'\\begin\{Extension\}(.*?)\\end\{Extension\}', extension, text, flags=re.S)
        text = macro(text, 'ex', 3, lambda a,b,c: self.box('例子'+(' · '+a if a else ''), c, 'tip'))
        # The source has both the two-argument form and a one-argument shorthand.
        while (m := re.search(r'\\weiyan(?![A-Za-z])', text)):
            first, end = group(text, m.end())
            rest = text[end:].lstrip()
            if rest.startswith('{'):
                second, end = group(text, end)
                title, body = '薇言大义'+(' · '+first if first.strip() else ''), second
            else:
                title, body = '薇言大义', first
            text = text[:m.start()] + self.box(title, body, 'tip') + text[end:]
        for name, title, kind in [('Problem','想一想','info'), ('Keynote','理解与提示','tip')]:
            pattern = re.compile(r'\\begin\{' + name + r'\}(.*?)\\end\{' + name + r'\}', re.S)
            def replace(m, title=title, kind=kind):
                body = m.group(1).lstrip()
                custom_title = ''
                if body.startswith('['):
                    custom_title, pos = group(body, 0, '[', ']')
                    body = body[pos:]
                    if custom_title.startswith('{'):
                        custom_title, _ = group(custom_title, 0)
                elif body.startswith('{'):
                    custom_title, pos = group(body, 0)
                    body = body[pos:]
                # Titles may contain semantics tokens; restore them at the end.
                return self.box(title, '\\textbf{'+custom_title+'}\n\n'+body if custom_title else body, kind)
            text = pattern.sub(replace, text)
        text = text.replace('\\begin{ChapterReferences}', '\n\\section*{参考文献与延伸阅读}\n').replace('\\end{ChapterReferences}', '')
        text = re.sub(r'\\begin\{(itemize|enumerate|description)\}\[[^]]*\]', r'\\begin{\1}', text)
        text = re.sub(r'\\renewcommand\{\\arraystretch\}\{[^}]*\}', '', text)
        # Partial horizontal rules are layout only; Pandoc otherwise emits '2-5' as cell text.
        text = re.sub(r'\\cline\s*\{\s*\d+\s*-\s*\d+\s*\}', '', text)
        text = macro(text, 'textbf' , 1, lambda x: self.token('<strong>') + x + self.token('</strong>'))
        text = re.sub(r'\\([A-Za-z]+?)(?=ZZNOTETOKEN)', r'\\\1 ', text)
        return text

    def convert(self, original, output):
        prepared = self.prepare(original)
        process = subprocess.run([str(PANDOC), '-f', 'latex', '-t', 'json'], input=prepared,
                                 text=True, encoding='utf-8', capture_output=True, check=True)
        ast = json.loads(process.stdout)
        source_n = int(self.filename[:-4]) if self.filename[:-4].isdigit() else None
        chapter_n = CHAPTERS.get(source_n, (None, None))[1]
        section_counts = [0,0,0]
        def walk(node, inside_table=False):
            if isinstance(node, list):
                for x in node:
                    walk(x, inside_table)
            elif isinstance(node, dict):
                if node.get('t') == 'Table':
                    c = node['c']
                    # A nested one-column tabular is a TeX line-wrap device, not
                    # a data table. Keep its text together in the enclosing cell.
                    if inside_table and len(c[2]) == 1 and not c[1][1] and not c[3][1] and not c[5][1]:
                        rows = [row for body in c[4] for row in body[2] + body[3]]
                        cells = [row[1][0] for row in rows if len(row[1]) == 1]
                        blocks = [block for cell in cells for block in cell[4]]
                        if len(cells) == len(rows) and all(cell[2:4] == [1, 1] for cell in cells) and all(block['t'] in ('Plain', 'Para') for block in blocks):
                            content = []
                            for block in blocks:
                                content.extend(block['c'])
                                content.append({'t': 'SoftBreak'})
                            node.update(t='Plain', c=content[:-1])
                    inside_table = True
                if node.get('t') in ('RawInline','RawBlock') and node['c'][0] in ('latex','tex'):
                    self.raw.append(node['c'][1])
                if node.get('t') == 'Header':
                    level = node['c'][0]
                    if chapter_n and level == 1:
                        node['c'][2].insert(0, {'t':'Str','c':f'第 {chapter_n} 章 · '})
                    elif chapter_n and 2 <= level <= 4 and 'unnumbered' not in node['c'][1][1]:
                        idx = level - 2
                        section_counts[idx] += 1
                        section_counts[idx+1:] = [0] * (2-idx)
                        prefix = '.'.join(map(str,[chapter_n]+section_counts[:idx+1]))
                        node['c'][2].insert(0, {'t':'Str','c':prefix+' '})
                    if self.filename == 'appendix.tex':
                        node['c'][0] += 1
                if node.get('t') == 'Image':
                    old = node['c'][-1][0]
                    path = SOURCE / old
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    new = 'images/' + hashlib.sha256(old.encode()).hexdigest()[:10] + path.suffix.lower()
                    dest = ROOT / 'content/public' / new
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    node['c'][-1][0] = '/' + new
                    self.images.append(new)
                for value in node.values():
                    walk(value, inside_table)
        walk(ast)
        process = subprocess.run([str(PANDOC), '-f','json','-t','gfm+raw_html','--wrap=none'],
                                 input=json.dumps(ast), text=True, encoding='utf-8',capture_output=True,check=True)
        markdown = process.stdout
        # Blank lines inside HTML cells let Markdown/MathJax render their contents.
        # Pandoc uses HTML for tables with spans or nested tabular environments.
        markdown = re.sub(r'(<(?:td|th)\b[^>]*>)', r'\1\n\n', markdown)
        markdown = re.sub(r'(</(?:td|th)>)', r'\n\n\1\n', markdown)
        # GFM escapes tokens only if punctuation is present; our placeholders are letters/digits.
        restored = 0
        while True:
            previous = markdown
            for key, value in self.tokens.items():
                if key in markdown:
                    if value.startswith('$'):
                        restored += markdown.count(key)
                        # A literal math bar must not become a Markdown table separator.
                        markdown = re.sub(r'(?m)^\|[^\n]*' + key + r'[^\n]*$',
                                          lambda m: m.group(0).replace(key, value.replace('|', r'\|')), markdown)
                    if '\n' in value:
                        markdown = re.sub(r'(?m)^([ \t]*)' + key + r'[ \t]*$',
                                          lambda m: '\n'.join(m.group(1) + line for line in value.split('\n')), markdown)
                    markdown = markdown.replace(key, value)
            if markdown == previous:
                break
        unresolved = re.findall(r'ZZNOTETOKEN\d+ZZ', markdown)
        # Vue templates would otherwise interpret literal angle-bracket examples.
        markdown = markdown.replace('<endoftext>', '&lt;endoftext&gt;')
        def figure_alt(m):
            caption = re.sub(r'<[^>]*>', '', m.group(2))
            return m.group(1).replace(' />', ' alt="' + html.escape(caption, quote=True) + '" />') + '<figcaption>' + m.group(2) + '</figcaption>'
        markdown = re.sub(r'(<figure[^>]*>\s*<img[^>]*>\s*)<figcaption>(.*?)</figcaption>', figure_alt, markdown, flags=re.S)
        front = '---\noutline: [2, 4]\n---\n\n' if self.filename == '10.tex' else '---\noutline: [2, 3]\n---\n\n'
        if self.filename == 'appendix.tex':
            intro = '# 基础知识速查\n\n'
            intro += '按需查阅：[数学基础](#app-math) · [深度学习基础](#app-deep-learning) · [AI Infra 基础](#app-infra) · [信息论基础](#app-information) · [正则表达式](#app-regex)。\n\n'
            if 'id="app-repro"' not in markdown:
                intro += '<span id="app-repro"></span>\n\n'
                intro += '::: info 实验复现与结果记录\n新版附录已按知识领域重新组织。前文的旧链接保留在此：训练日志与 checkpoint 见[训练流程与实验管理](/part-1/chapter-4)，计时范围与环境记录见[性能分析与基准测试](/part-2/chapter-6)。\n:::\n\n'
            markdown = intro + markdown
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(front + markdown, encoding='utf-8')
        return {'source':self.filename,'sha256':hashlib.sha256(original.encode()).hexdigest(),
                'output':str(output.relative_to(ROOT)), 'math_count':len(self.math),
                'restored_math_count':restored,'code_count':len(self.code), 'images':self.images,
                'unresolved_raw_tex':self.raw,'unresolved_tokens':unresolved}

def main():
    reports = []
    for number, (part, chapter) in CHAPTERS.items():
        name = f'{number}.tex'
        reports.append(Migration(name).convert((SOURCE/name).read_text(encoding='utf-8'),ROOT/f'content/part-{part}/chapter-{chapter}.md'))
    reports.append(Migration('appendix.tex').convert((SOURCE/'appendix.tex').read_text(encoding='utf-8'), ROOT/'content/appendix.md'))
    for part, title in [(1, '数据与作业入口'), (2, '实验任务与资源入口'), (3, '作业与阅读入口'), (4, '数据与实验入口'), (5, '模型、数据与实验入口'), (6, '模型、数据与评估入口')]:
        resources = (SOURCE/'part-resources.tex').read_text(encoding='utf-8').split('\\or')[part]
        if part == 6:
            resources = resources.split('\\fi', 1)[0]
        if part == 2:
            resources = resources.replace('\\input{6.tex}', (SOURCE/'6.tex').read_text(encoding='utf-8').replace('\\subsection*{实验目标与任务}', '\\section*{实验目标与任务}'))
        resources = '\\chapter*{' + title + '}\n' + resources
        if part == 3:
            resources = resources.replace('\\chapter*{' + title + '}', '\\chapter*{' + title + '}\n\\NoteLabel{本篇进度：}对应 Assignment 3。由于缺乏对应资源，原稿尚未展开实验，当前以理论分析为主。以下保留资源文件中的入口与练习建议，供后续准备使用。\n')
        reports.append(Migration('part-resources.tex').convert(resources, ROOT/f'content/part-{part}/resources.md'))
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/migration.json').write_text(json.dumps({'documents':reports},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
