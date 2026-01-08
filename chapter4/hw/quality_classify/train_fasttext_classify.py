"""
训练FastText质量分类器

1. 合并正负样本文件
2. 打乱数据
3. 划分训练集和验证集
4. 训练模型
"""

import fasttext
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def merge_and_shuffle_samples(
    positive_file: str,
    negative_file: str,
    output_file: str,
    shuffle: bool = True,
    random_seed: int = 42
) -> int:
    """
    合并正负样本文件并打乱
    
    Args:
        positive_file: 正样本文件路径
        negative_file: 负样本文件路径
        output_file: 输出文件路径
        shuffle: 是否打乱数据
        random_seed: 随机种子
    
    Returns:
        合并后的总样本数
    """
    random.seed(random_seed)
    
    # 读取所有样本
    all_samples = []
    
    # 读取正样本
    positive_count = 0
    try:
        with open(positive_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading positive samples"):
                line = line.strip()
                if line and line.startswith('__label__hq'):
                    all_samples.append(line)
                    positive_count += 1
    except FileNotFoundError:
        return 0
    
    # 读取负样本
    negative_count = 0
    try:
        with open(negative_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading negative samples"):
                line = line.strip()
                if line and line.startswith('__label__lq'):
                    all_samples.append(line)
                    negative_count += 1
    except FileNotFoundError:
        return 0
    
    # 打乱数据
    if shuffle:
        random.shuffle(all_samples)
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(all_samples, desc="Writing merged samples"):
            f.write(line + '\n')
    
    return len(all_samples)


def split_train_val(
    merged_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple[int, int]:
    """
    划分训练集和验证集
    
    Args:
        merged_file: 合并后的文件路径
        train_file: 训练集输出文件路径
        val_file: 验证集输出文件路径
        val_ratio: 验证集比例（默认10%）
        random_seed: 随机种子
    
    Returns:
        (训练集样本数, 验证集样本数)
    """
    random.seed(random_seed)
    
    # 读取所有样本
    all_samples = []
    with open(merged_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading samples"):
            line = line.strip()
            if line:
                all_samples.append(line)
    
    # 打乱
    random.shuffle(all_samples)
    
    # 划分训练集和验证集
    val_size = int(len(all_samples) * val_ratio)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in tqdm(train_samples, desc="Writing train set"):
            f.write(line + '\n')
    
    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in tqdm(val_samples, desc="Writing val set"):
            f.write(line + '\n')
    
    return len(train_samples), len(val_samples)


def train_quality_classifier(
    train_path: str,
    model_path: str,
    val_path: Optional[str] = None,
    **kwargs
) -> fasttext.FastText:
    """
    训练并保存 fastText 分类器
    
    Args:
        train_path: 训练数据路径
        model_path: 模型保存路径
        val_path: 验证集路径（可选）
        **kwargs: 其他训练参数
    
    Returns:
        训练好的模型
    """
    # 默认训练参数
    train_params = {
        'input': train_path,
        'lr': 0.1,
        'epoch': 25,
        'wordNgrams': 2,
        'dim': 100,
        'loss': 'softmax',
        'minCount': 5,
        'bucket': 2000000,
    }
    
    # 更新用户自定义参数
    train_params.update(kwargs)
    
    model = fasttext.train_supervised(**train_params)
    
    # 保存模型
    model.save_model(model_path)
    
    # 如果有验证集，评估模型
    if val_path and Path(val_path).exists():
        result = model.test(val_path)
    
    return model



def main():
    """主函数"""
    script_dir = Path(__file__).parent
    
    # 文件路径
    positive_file = str(script_dir / "wiki_positive_samples.txt")
    negative_file = str(script_dir / "wiki_negative_samples.txt")
    merged_file = str(script_dir / "quality_train_merged.txt")
    train_file = str(script_dir / "quality_train.txt")
    val_file = str(script_dir / "quality_val.txt")
    model_path = str(script_dir / "quality_classifier.bin")
    
    # 步骤1: 合并正负样本
    total_samples = merge_and_shuffle_samples(
        positive_file=positive_file,
        negative_file=negative_file,
        output_file=merged_file,
        shuffle=True
    )
    
    if total_samples == 0:
        return
    
    # 步骤2: 划分训练集和验证集
    train_count, val_count = split_train_val(
        merged_file=merged_file,
        train_file=train_file,
        val_file=val_file,
        val_ratio=0.1  # 10%作为验证集
    )
    
    # 步骤3: 训练模型
    model = train_quality_classifier(
        train_path=train_file,
        model_path=model_path,
        val_path=val_file
    )
    


if __name__ == "__main__":
    main()
