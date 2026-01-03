import gzip
import random

def reservoir_sample(filepath: str, k: int, seed: int = 42) -> list[str]:
    """
    蓄水池采样：从大文件中随机采样 k 行，内存占用 O(k)
    
    Args:
        filepath: 文件路径（支持 .gz 压缩文件）
        k: 采样数量
        seed: 随机种子，保证可复现
    
    Returns:
        采样得到的 k 行文本
    """
    random.seed(seed)
    reservoir = []
    
    # 根据文件后缀选择打开方式
    open_func = gzip.open if filepath.endswith('.gz') else open
    
    with open_func(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                # 以 k/(i+1) 的概率替换
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line
            
            # 进度提示（每 100 万行打印一次）
            if (i + 1) % 1_000_000 == 0:
                print(f"Processed {(i+1) // 1_000_000}M lines...")
    
    print(f"Sampling complete. Total lines scanned: {i+1}")
    return reservoir


def save_urls(urls: list[str], output_path: str) -> None:
    """保存 URL 列表到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(url + '\n')
    print(f"Saved {len(urls)} URLs to {output_path}")


if __name__ == "__main__":
    # Wikipedia 外链 URL 文件路径
    wiki_urls_path = "enwiki-20240420-extracted_urls.txt.gz"
    
    # 采样 30k 条
    sampled_urls = reservoir_sample(wiki_urls_path, k=30000)
    
    # 保存
    save_urls(sampled_urls, "30k_positive_urls.txt")
