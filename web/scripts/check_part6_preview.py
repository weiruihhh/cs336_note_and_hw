"""Check a running local preview; never executes note examples."""
import json
import urllib.request
import hashlib
from pathlib import Path
from html.parser import HTMLParser
ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:4178/cs336_note_and_hw/'
class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.counts = {'math': 0, 'tables': 0, 'images': 0, 'code_blocks': 0}
        self.parts, self.images, self.rows = [], [], []
        self.cell, self.row = None, None
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for element, key in [('mjx-container', 'math'), ('table', 'tables'), ('img', 'images'), ('pre', 'code_blocks')]:
            if tag == element: self.counts[key] += 1
        if tag == 'img': self.images.append(attrs['src'])
        if tag == 'tr': self.row = []
        if tag in ['td', 'th']: self.cell = []
    def handle_endtag(self, tag):
        if tag in ['td', 'th'] and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None
    def handle_data(self, data):
        self.parts.append(data)
        if self.cell is not None: self.cell.append(data)
def main():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report = []
    for route in ['', 'part-6/', 'part-6/chapter-14', 'part-6/chapter-15', 'part-6/resources']:
        response = opener.open(BASE + route, timeout=15)
        assert response.status == 200
        page = Page(response.read().decode('utf-8'))
        text = ''.join(page.parts)
        if route.endswith('chapter-14'):
            assert page.counts['math'] == 144 and page.counts['tables'] == 2
            assert '<pad>' in text and '<eos>' in text and 'IPO' in text and 'KTO' in text
        if route.endswith('chapter-15'):
            assert page.counts['math'] == 7 and page.counts['tables'] == 1
            assert 'Instruction:{prompt}' in text and 'Response:{response}' in text
            assert ['', '准确率', '10.462%', '18.423%', '18.95%'] in page.rows
            assert ['', '准确率', '58.375%', '54.878%', '55.00%'] in page.rows
            assert len(page.rows) == 15 and not any(row[0] == '2-5' for row in page.rows)
        for image in page.images:
            response = opener.open('http://127.0.0.1:4178' + image, timeout=15)
            assert response.status == 200 and len(response.read()) > 0
        report.append({'route': route, 'status': 200, **page.counts, 'images_checked': page.images})
    (ROOT/'reports/part-6-http-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
