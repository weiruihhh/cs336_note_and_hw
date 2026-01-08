#!/usr/bin/env python3
"""
从Common Crawl (CC) WARC文件中提取负样本并转换为FastText格式
"""

import gzip
import sys
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(Path(__file__).parent.parent))
# 导入已经编写好的模块
from filter_email import run_mask_emails
from filter_ip import run_mask_ips
from filter_phone_number import run_mask_phone_numbers
from quality_rule import run_gopher_quality_filter
from extract_text import run_extract_text_from_html_bytes


def desensitize_text(text: str) -> str:
    """
    对文本进行脱敏处理
    """
    # 使用已有的脱敏函数
    text, _ = run_mask_emails(text)
    text, _ = run_mask_phone_numbers(text)
    text, _ = run_mask_ips(text)
    
    return text



def extract_text_from_warc_record(record) -> Optional[str]:
    """
    从WARC记录中提取纯文本
    
    Returns:
        提取的文本，如果提取失败则返回None
    """
    # 只处理 response 类型的记录
    if record.rec_type != 'response':
        return None
    
    # 只处理 HTML 内容
    content_type = record.http_headers.get_header('Content-Type', '')
    if 'text/html' not in content_type.lower():
        return None
    
    try:
        # 读取HTML内容
        html_bytes = record.content_stream().read()
        
        # 提取纯文本
        text = run_extract_text_from_html_bytes(html_bytes)
        
        if text and len(text.strip()) > 100:  # 过滤太短的文本
            return text
    except Exception as e:
        return None
    
    return None


def process_text(text: str) -> Optional[str]:
    """
    处理单个文本：脱敏 
    """
    if not text or not text.strip():
        return None
    
    # 1. 脱敏
    text = desensitize_text(text)
    
    # 2. 清理：压缩空格，转换为单行（fastText要求）
    text = ' '.join(text.split())
    
    return text


def extract_negative_samples_from_warc(
    warc_paths: list[str],
    target_count: int,
    random_seed: int = 42
) -> list[str]:
    """
    从WARC文件中提取负样本（支持多个WARC文件）
    
    Args:
        warc_paths: WARC文件路径列表
        target_count: 目标样本数量，努力和正样本数量一致
        random_seed: 随机种子
    
    Returns:
        提取的文本列表
    """
    random.seed(random_seed)
    
    processed_texts = []
    total_records = 0
    skipped_empty = 0
    skipped_quality = 0
    extracted_count = 0
    

    for warc_idx, warc_path in enumerate(warc_paths, 1):
        if len(processed_texts) >= target_count:
            print(f"\n已达到目标数量，停止处理")
            break
        
        print(f"\n处理文件 {warc_idx}/{len(warc_paths)}: {Path(warc_path).name}")
        
        if not Path(warc_path).exists():
            print(f"警告：文件不存在，跳过: {warc_path}")
            continue
        
        try:
            with gzip.open(warc_path, 'rb') as f:
                for record in tqdm(ArchiveIterator(f), desc=f"Processing WARC {warc_idx}"):
                    total_records += 1
                    
                    # 如果已经达到目标数量，提前退出
                    if len(processed_texts) >= target_count:
                        break
                    
                    # 提取文本
                    text = extract_text_from_warc_record(record)
                    
                    if text is None:
                        skipped_empty += 1
                        continue
                    
                    extracted_count += 1
                    
                    # 处理文本（脱敏）
                    processed_text = process_text(text)
                    
                    if processed_text is None:
                        skipped_quality += 1
                        continue
                    
                    processed_texts.append(processed_text)
                    
                    # 每处理100条输出一次进度
                    if len(processed_texts) % 100 == 0:
                        print(f"已处理: {len(processed_texts)} 条有效样本 (目标: {target_count})")
        
        except Exception as e:
            print(f"错误：读取WARC文件失败: {e}")
            continue
    
    return processed_texts


def count_positive_samples(positive_file: str) -> int:
    """
    统计正样本文件中的样本数量
    """
    try:
        count = 0
        with open(positive_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and line.startswith('__label__hq'):
                    count += 1
        return count
    except FileNotFoundError:
        return 0


def save_fasttext_format(texts: list[str], label: str, output_path: str):
    """
    保存为 fastText 格式
    
    格式: __label__xx 文本内容
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"__label__{label} {text}\n")


def find_warc_files(datasets_dir: Path) -> list[str]:
    """
    查找datasets目录下的所有WARC文件
    """
    warc_files = list(datasets_dir.glob("*.warc.gz"))
    # 排除WET文件（只处理WARC文件）
    warc_files = [f for f in warc_files if not f.name.endswith(".wet.gz")]
    return sorted([str(f) for f in warc_files])


def main():
    script_dir = Path(__file__).parent
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    
    # 查找所有WARC文件
    warc_paths = find_warc_files(datasets_dir)
    
    # 正样本文件路径（用于统计数量）
    positive_file = str(script_dir / "wiki_positive_samples.txt")
    
    # 输出文件路径
    output_path = str(script_dir / "wiki_negative_samples.txt")
    
    # 统计正样本数量
    positive_count = count_positive_samples(positive_file)
    
    if positive_count == 0:
        # 如果正样本数量为0，则使用默认数量10000
        target_count = 10000
    else:
        target_count = positive_count
    
    # 检查WARC文件
    if not warc_paths:
        return
    
    # 提取负样本
    negative_texts = extract_negative_samples_from_warc(
        warc_paths=warc_paths,
        target_count=target_count
    )
    
    # 保存为FastText格式
    if negative_texts:
        save_fasttext_format(negative_texts, label="lq", output_path=output_path)



if __name__ == "__main__":
    main()

