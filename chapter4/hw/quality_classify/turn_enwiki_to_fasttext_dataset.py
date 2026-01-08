"""
将提取的 Wikipedia 数据转换为 FastText 质量分类器的训练正样本

处理流程：
1. 数据清洗pipeline
   - 脱敏（邮箱、电话、IP地址）
   - Gopher质量规则过滤
2. 转换成fasttext的训练集格式
   - 正样本：__label__hq xxxxxxxxx...
"""

import json
import sys
from pathlib import Path
from typing import Generator, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
# 导入已经编写好的模块
from filter_email import run_mask_emails
from filter_ip import run_mask_ips
from filter_phone_number import run_mask_phone_numbers
from quality_rule import run_gopher_quality_filter


def desensitize_text(text: str) -> str:
    """
    对文本进行脱敏处理
    """
    # 使用已有的脱敏函数
    text, _ = run_mask_emails(text)
    text, _ = run_mask_phone_numbers(text)
    text, _ = run_mask_ips(text)
    
    return text


def check_gopher_quality_rules(text: str) -> bool:
    """
    检查文本是否符合Gopher质量规则
    - 词数大于50 小于100000
    - 平均词长3-10
    - 省略号占比小于30%
    """
    return run_gopher_quality_filter(text)


def read_extracted_files(extracted_dir: str) -> Generator[dict, None, None]:
    """
    读取所有提取的JSONL文件
    """
    extracted_path = Path(extracted_dir)
    
    # 遍历所有子目录
    for subdir in sorted(extracted_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 遍历子目录中的所有wiki_*文件
        for wiki_file in sorted(subdir.glob('wiki_*')):
            if not wiki_file.is_file():
                continue
            
            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            article = json.loads(line)
                            yield article #惰性迭代，每次只返回一个article
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error reading {wiki_file}: {e}")
                continue


def process_text(text: str) -> Optional[str]:
    """
    处理单个文本：脱敏 + 质量过滤
    
    Returns:
        处理后的文本，如果不符合质量要求则返回None
    """
    if not text or not text.strip():
        return None
    
    # 1. 脱敏
    text = desensitize_text(text)
    
    # 2. Gopher质量规则过滤
    if not check_gopher_quality_rules(text):
        return None
    
    # 3. 清理：压缩空格，转换为单行（fastText要求）
    text = ' '.join(text.split())
    
    return text


def save_fasttext_format(texts: list[str], label: str, output_path: str):
    """
    保存为 fastText 格式
    
    格式: __label__xx 文本内容
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # FastText格式要求：标签和文本之间用空格分隔，文本中的换行符需要转义
            f.write(f"__label__{label} {text}\n")


def main():
    script_dir = Path(__file__).parent
    extracted_dir = str(script_dir / "extracted")
    output_path = str(script_dir / "wiki_positive_samples.txt")
    
    
    processed_texts = []
    total_articles = 0
    skipped_empty = 0
    skipped_quality = 0
    
    # 读取并处理所有文章
    for article in tqdm(read_extracted_files(extracted_dir), desc="Processing articles"):
        total_articles += 1
        text = article.get('text', '')
        
        if not text or not text.strip():
            skipped_empty += 1
            continue
        
        # 处理文本
        processed_text = process_text(text)
        
        if processed_text is None:
            skipped_quality += 1
            continue
        
        processed_texts.append(processed_text)
        
        # 每处理1000条输出一次进度
        if len(processed_texts) % 1000 == 0:
            print(f"已处理: {len(processed_texts)} 条有效样本")
    
    # 保存为FastText格式，这里是正样本，所以标签是hq
    if processed_texts:
        save_fasttext_format(processed_texts, label="hq", output_path=output_path)


if __name__ == "__main__":
    main()

