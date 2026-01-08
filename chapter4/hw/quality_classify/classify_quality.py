import fasttext
from typing import Any

def run_classify_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('final_quality_classifier.bin')
    perdiction = model.predict(text.replace('\n', ' '))

    label = perdiction[0][0]   
    score = perdiction[1][0]

    #去掉 __label__前缀
    quality_detect = label.replace('__label__', '')
    return (quality_detect, score)