from tokenizer_encode import Tokenizer
import os

import pickle

special_tokens = ["<|endoftext|>"]  
# 从 vocab.pkl 加载词汇表
with open("owt_vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
with open("owt_merges.pkl", "rb") as f:
    # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens)

data_path = "../data/owt_train.txt"
data_path = "../data/owt_valid.txt"
with open(data_path, "r",encoding="utf-8") as f:
    original_data = f.read()
encode_ids = tokenizer.encode(original_data)

# with open("owt_encoded_ids_train.pkl", "wb") as f:
#     pickle.dump(encode_ids, f)
with open("owt_encoded_ids_valid.pkl", "wb") as f:
    pickle.dump(encode_ids, f)