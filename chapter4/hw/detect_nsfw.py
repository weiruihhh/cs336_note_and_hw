import fasttext

def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('jigsaw_fasttext_bigrams_nsfw_final.bin')
    perdiction = model.predict(text)

    label = perdiction[0][0]   
    score = perdiction[1][0]

    #去掉 __label__前缀
    nsfw_detect = label.replace('__label__', '')
    return (nsfw_detect, score)
