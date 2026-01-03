import gzip
from warcio.archiveiterator import ArchiveIterator
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import random
def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """你已有的函数"""
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if encoding:
            html_str = html_bytes.decode(encoding, errors='ignore')
        else:
            return None
    
    text = extract_plain_text(html_str)
    return text
def extract_texts_from_warc(warc_path: str, max_samples: int = 30000) -> list[str]:
    """
    从 WARC 文件中提取纯文本
    
    Args:
        warc_path: WARC 文件路径
        max_samples: 最大采样数量
    
    Returns:
        提取的文本列表
    """
    texts = []
    
    with gzip.open(warc_path, 'rb') as f:
        for i, record in enumerate(ArchiveIterator(f)):
            # 只处理 response 类型的记录
            if record.rec_type != 'response':
                continue
            
            # 只处理 HTML 内容
            content_type = record.http_headers.get_header('Content-Type', '')
            if 'text/html' not in content_type:
                continue
            
            # 读取内容
            html_bytes = record.content_stream().read()
            
            # 提取文本
            text = extract_text_from_html_bytes(html_bytes)
            
            # 过滤：文本太短的跳过
            if text and len(text.strip()) > 100:
                # 清理：去掉换行，压缩空格（fastText 要求单行）
                clean_text = ' '.join(text.split())
                texts.append(clean_text)
            
            # 进度提示
            if len(texts) % 1000 == 0 and len(texts) > 0:
                print(f"Extracted {len(texts)} texts...")
            
            # 够了就停
            if len(texts) >= max_samples:
                break
    
    print(f"Total extracted: {len(texts)} texts")
    return texts
def save_fasttext_format(texts: list[str], label: str, output_path: str):
    """
    保存为 fastText 格式
    
    格式: __label__xx 文本内容
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"__label__{label} {text}\n")
    print(f"Saved {len(texts)} samples to {output_path}")
if __name__ == "__main__":
    # 你的 CC WARC 文件路径
    warc_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    
    # 提取负样本
    negative_texts = extract_texts_from_warc(warc_path, max_samples=30000)
    
    # 保存
    save_fasttext_format(negative_texts, label="lq", output_path="negative.txt")