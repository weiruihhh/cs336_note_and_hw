def check_word_count(words: list[str]) -> bool:
    "检查单词数量是否在50到100000之间"
    count = len(words)
    return 50 <= count <= 100000

def check_mean_word_length(words: list[str]) -> bool:
    "检查单词平均长度是否在3到10之间"
    mean_length = sum(len(word) for word in words) / len(words)
    return 3 < mean_length < 10

def check_ellipsis_ratio(text: str) -> bool:
    "检查文本中省略号数量占比是否小于30%"
    lines = text.split('\n')
    ellipsis_count = sum(1 for line in lines if line.strip().endswith('...'))
    # print(f"ellipisi_count:{ellipsis_count}")
    return ellipsis_count / len(lines) < 0.3

def check_aplhabet_ratio(words: list[str]) -> bool:
    "检查至少80%的单词包含字母"
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    return alpha_words / len(words) > 0.8


def run_gopher_quality_filter(text: str) -> bool:


    #先分词
    # words = nltk.word_tokenize(text)
    words = text.split()

    # result1 = check_word_count(words)
    # result2 = check_mean_word_length(words)
    # result3 = check_ellipsis_ratio(text)
    # result4 = check_aplhabet_ratio(words)   
    
    # print(f"word_count: {result1}, avg_len: {result2}, ellipsis: {result3}, alpha: {result4}")
    return check_aplhabet_ratio(words) and check_ellipsis_ratio(text) and check_word_count(words) and check_mean_word_length(words)