import fasttext

def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model('jigsaw_fasttext_bigrams_hatespeech_final.bin')
    perdiction = model.predict(text)

    label = perdiction[0][0]   
    score = perdiction[1][0]

    #去掉 __label__前缀
    toxic_speech_detect = label.replace('__label__', '')
    return (toxic_speech_detect, score)
