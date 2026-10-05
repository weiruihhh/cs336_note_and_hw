"""Independent checks of migrated prose and the built site's internal links."""
import json
import os
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('CS336_TEX_SOURCE', ROOT.parent / 'cs336第二版笔记')).resolve()

class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.links, self.images, self.math_errors = set(), [], [], []
        self.math_count = 0
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'mjx-container':
            self.math_count += 1
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag == 'img' and 'src' in attrs:
            self.images.append(attrs['src'])
        if attrs.get('data-mml-node') == 'merror' or tag == 'mjx-merror' or (tag == 'mtext' and attrs.get('mathcolor') == 'red'):
            self.math_errors.append(attrs)

def audit():
    prose = []
    for n in [1, 2, 3, 4, 5, 'appendix']:
        source = (SOURCE/f'{n}.tex').read_text(encoding='utf-8')
        source = re.sub(r'(?<!\\)%[^\n]*', '', source)
        source = re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]*\}', '', source)
        page = ROOT/'content/appendix.md' if n == 'appendix' else ROOT/f'content/part-1/chapter-{n}.md'
        markdown = page.read_text(encoding='utf-8')
        expected = Counter(re.findall(r'[\u4e00-\u9fff]{4,}', source))
        # Removing a TeX wrapper can join adjacent Chinese phrases in Markdown.
        # Count occurrences rather than requiring identical token boundaries.
        actual = Counter({phrase: markdown.count(phrase) for phrase in expected})
        missing = expected-actual
        prose.append({'chapter':n, 'missing_chinese_passages':dict(missing)})
    dist = ROOT/'content/.vitepress/dist'
    pages = {p.relative_to(dist).as_posix():Page(p.read_text(encoding='utf-8')) for p in dist.rglob('*.html')}
    issues = []
    migration = json.loads((ROOT/'reports/migration.json').read_text(encoding='utf-8'))
    math = []
    for doc in migration['documents']:
        name = Path(doc['output'].replace('\\', '/')).relative_to('content').with_suffix('.html').as_posix()
        actual = pages[name].math_count
        math.append({'page': name, 'expected': doc['math_count'], 'rendered': actual})
        if actual != doc['math_count']:
            issues.append({'page': name, 'issue': 'static math count mismatch'})
    base = '/cs336_note_and_hw/'
    for name, page in pages.items():
        for url in page.links+page.images:
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            path = unquote(parsed.path)
            if path.startswith(base):
                path = path[len(base):]
            elif path.startswith('/'):
                issues.append({'page':name,'url':url,'issue':'missing project base'})
                continue
            elif not path:
                path = name
            else:
                path = (Path(name).parent/path).as_posix()
            if path.endswith('/') or not path:
                path += 'index.html'
            elif not Path(path).suffix:
                path += '.html'
            if not (dist/path).is_file():
                issues.append({'page':name,'url':url,'issue':'missing file'})
            elif parsed.fragment and path in pages and unquote(parsed.fragment) not in pages[path].ids:
                issues.append({'page':name,'url':url,'issue':'missing anchor'})
        for error in page.math_errors:
            issues.append({'page':name,'issue':'math rendering error','detail':error})
    result = {'prose':prose,'math':math,'html_pages':len(pages),'site_issues':issues}
    (ROOT/'reports/content-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if issues or any(item['missing_chinese_passages'] for item in prose):
        raise SystemExit(1)

if __name__ == '__main__':
    audit()
