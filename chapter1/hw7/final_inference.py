import torch
from transformermodule import TransformerModule
from inference import decode_token
import pickle
from tokenizer_encode import Tokenizer

# 定义模型结构（参数要和训练时一致）
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
n_layers = 4
n_heads = 16
rope_theta = 10000.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModule(d_model, n_heads, d_ff, context_length, rope_theta, n_layers, vocab_size, device).to(device)

# 加载权重
model.load_state_dict(torch.load("checkpoints/model_final_20250706_221431.pth", map_location=device))

# # 加载中途的checkpoint文件 先加载整个 checkpoint 文件到一个变量中
# checkpoint = torch.load("checkpoints/model_epoch_59_20250706_221431.pth", map_location=device)
# # 然后从 checkpoint 字典中，根据键 'model_state_dict' 取出真正的模型权重
# model_weights = checkpoint['model_state_dict']
# # 最后将取出的模型权重加载到模型中
# model.load_state_dict(model_weights)

# 从 vocab.pkl 加载词汇表
with open("vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
with open("merges.pkl", "rb") as f:
    # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
    merges = pickle.load(f)
special_tokens = ["<|endoftext|>"]  

tokenizer = Tokenizer(vocab, merges, special_tokens)
input_text = "Once upon a time, there was a pretty girl named Lily.One day, Lily’s mom asked her to help cook dinner."
# input_text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of darkness, it was the spring of hope, it was the winter of despair."
# input_text = "baby shark"
input_ids = tokenizer.encode(input_text)
input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)

# 推理
output_ids = decode_token(input_ids,model,max_tokens_to_generate=200)
print(output_ids)

output_ids_list = output_ids[0].cpu().tolist()
output_text = tokenizer.decode(output_ids_list)
print(output_text)




