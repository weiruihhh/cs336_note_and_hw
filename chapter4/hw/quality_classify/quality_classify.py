import fasttext

def train_quality_classifier(train_path: str, model_path: str) -> None:
    """训练并保存 fastText 分类器"""
    model = fasttext.train_supervised(
        input=train_path,           # 训练数据路径
        lr=0.1,                      # 学习率
        epoch=25,                    # 训练轮数
        wordNgrams=2,                # 词级别 n-gram (bigram)
        dim=100,                     # embedding 维度
        loss='softmax',              # 损失函数 (二分类也可用 'hs' 或 'ova')
        minCount=5,                  # 词频阈值，低于此的词被忽略
        bucket=2000000,              # n-gram hash 桶数量
    )
    # 保存模型
    model.save_model("quality_classifier.bin")

def run_classify_quality(text: str, model) -> tuple[str, float]:
    """
    对文本进行质量分类
    
    Returns:
        ("high-quality", confidence) 或 ("low-quality", confidence)
    """
    # model = fasttext.load_model(model)
    perdiction = model.predict(text)
    label = perdiction[0][0]
    score = perdiction[1][0]
    return (label, score)

if __name__ == "__main__":
    # 1. 训练模型
    train_quality_classifier("quality_train.txt", "quality_model.bin")
    
    # 2. 加载模型并测试
    model = fasttext.load_model("quality_classifier.bin")
    
    # 3. 验证
    test_cases = [
        "The mitochondria is the powerhouse of the cell.",
        "FREE VIAGRA CLICK HERE NOW!!!",
        "Quantum computing leverages superposition and entanglement for parallel computation.",
        "lol idk man just click the link trust me bro",
    ]
    
    for text in test_cases:
        label, conf = run_classify_quality(text, model)
        print(f"{label} ({conf:.3f}): {text[:50]}...")
