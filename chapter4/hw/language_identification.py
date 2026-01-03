import fasttext
def run_identify_language(text: str) -> tuple[Any, float]:
    # 1. 加载预训练模型（通常在模块级别加载一次，避免重复加载）
    model = fasttext.load_model('lid.176.bin')
    # 2. 预测语言
    # fastText 返回格式: (['__label__en'], array([0.9876]))
    perdiction = model.predict(text.replace('\n', ' '))# 因为一次只能预测一行文本，所以将换行符替换为空格
    print(perdiction)
    
    label = perdiction[0][0]
    score = perdiction[1][0]

    #去掉 __label__前缀
    language_code = label.replace('__label__', '')
    return (language_code, score)
