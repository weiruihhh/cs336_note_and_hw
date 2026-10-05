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
    if label.startswith('guide:ch:'):
        number = int(label.split(':')[-1])
        if number <= 5:
            return f'/part-1/chapter-{number}#' + anchor(label)
        return '/part-1/#reading-scope'
    if label.startswith('app:'):
        return '/appendix#' + anchor(label)
    if label.startswith('sec:') or label.startswith('read:'):
        return f'/part-1/chapter-{label.split(":")[1]}#' + anchor(label)
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
            rendered = '$' + content + '$' if inline else '$$\n' + content + '\n$$'
            return self.token(rendered) if inline else self.block(rendered)
        return pattern.sub(replace, text)

    def prepare(self, original):
        text = self.protect_code(original)
        text = re.sub(r'(?<!\\)%[^\n]*', '', text)
        if self.filename == 'appendix.tex':
            text = text[text.index('\\chapter{'):]
        text = self.protect_math(text)
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
                     '\\href{' + link('guide:ch:'+n) + '}{' + t + '}' if int(n) <= 5
                     else t + '（后续篇章，本次试读未收录）')
        text = macro(text, 'AppLink', 2, lambda n,t: '\\href{' + link('app:'+n) + '}{' + t + '}')
        text = macro(text, 'ReferencePointer', 2, lambda a,b: '\\href{'+link(a)+'}{参考资料 '+b+'}')
        text = macro(text, 'ReferenceGroup', 3, lambda a,b,c: '\n\\subsection*{'+b+' · '+c+'}\n\\label{'+a+'}\n')
        text = macro(text, 'label', 1, lambda x: self.token('<span id="'+anchor(x)+'"></span>'))
        text = re.sub(r'\\ref\*?\{([^}]+)\}', lambda m: m.group(1).split(':')[-1] if not m.group(1).startswith('app:') else '', text)
        text = re.sub(r'\\hyperref\[([^]]+)\]', lambda m: '\\href{'+link(m.group(1))+'}', text)
        text = macro(text, 'ext', 2, lambda a,b: self.box('延伸阅读 · '+a, b))
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
        text = macro(text, 'textbf', 1, lambda x: self.token('<strong>') + x + self.token('</strong>'))
        text = re.sub(r'\\([A-Za-z]+?)(?=ZZNOTETOKEN)', r'\\\1 ', text)
        return text

    def convert(self, original, output):
        prepared = self.prepare(original)
        process = subprocess.run([str(PANDOC), '-f', 'latex', '-t', 'json'], input=prepared,
                                 text=True, encoding='utf-8', capture_output=True, check=True)
        ast = json.loads(process.stdout)
        chapter_n = int(self.filename[:-4]) if self.filename[:-4].isdigit() else None
        section_counts = [0,0,0]
        def walk(node):
            if isinstance(node, list):
                for x in node:
                    walk(x)
            elif isinstance(node, dict):
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
                    walk(value)
        walk(ast)
        process = subprocess.run([str(PANDOC), '-f','json','-t','gfm+raw_html','--wrap=none'],
                                 input=json.dumps(ast), text=True, encoding='utf-8',capture_output=True,check=True)
        markdown = process.stdout
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
        front = '---\noutline: [2, 3]\n---\n\n'
        if self.filename == 'appendix.tex':
            markdown = '# 基础知识速查\n\n' + markdown
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(front + markdown, encoding='utf-8')
        return {'source':self.filename,'sha256':hashlib.sha256(original.encode()).hexdigest(),
                'output':str(output.relative_to(ROOT)), 'math_count':len(self.math),
                'restored_math_count':restored,'code_count':len(self.code), 'images':self.images,
                'unresolved_raw_tex':self.raw,'unresolved_tokens':unresolved}

def main():
    reports = []
    for number in range(1,6):
        name = f'{number}.tex'
        reports.append(Migration(name).convert((SOURCE/name).read_text(encoding='utf-8'),ROOT/f'content/part-1/chapter-{number}.md'))
    reports.append(Migration('appendix.tex').convert((SOURCE/'appendix.tex').read_text(encoding='utf-8'), ROOT/'content/appendix.md'))
    resources = (SOURCE/'part-resources.tex').read_text(encoding='utf-8').split('\\or')[1]
    resources = '\\chapter*{数据与作业入口}\n'+resources
    reports.append(Migration('part-resources.tex').convert(resources, ROOT/'content/part-1/resources.md'))
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/migration.json').write_text(json.dumps({'documents':reports},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(reports,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
