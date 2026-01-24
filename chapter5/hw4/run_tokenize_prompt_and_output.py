import torch
from torch import Tensor
import torch.nn.functional as F

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:

    #第一步要先拼接,不要把prompt和output分开处理，而是拼接到一起之后再进行掩码
    full_sequence_ids = [] #先用list来存储拼接后的token id
    padding_length = 0
    prompt_length_index = [] 
    output_length_index = []
    for i in range(len(prompt_strs)):
        prompt_ids_i = tokenizer.encode(prompt_strs[i])
        output_ids_i = tokenizer.encode(output_strs[i])
        full_sequence_ids_i = prompt_ids_i + output_ids_i
        full_sequence_ids.append(full_sequence_ids_i)
        #记录prompt和output的长度，方便后续的mask操作
        prompt_length_index.append(len(prompt_ids_i))
        output_length_index.append(len(output_ids_i)+prompt_length_index[i])
        padding_length = max(padding_length, len(full_sequence_ids_i))
    #得到tensor格式的拼接后的结果
    B = len(full_sequence_ids)
    pad_id = tokenizer.pad_token_id  #这个是专门用来补齐padding的token的id
    full = torch.full((B, padding_length), pad_id, dtype=torch.long)
    for i, ids in enumerate(full_sequence_ids):
        full[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    full_sequence_ids = full

    raw_mask = torch.zeros((B,padding_length), dtype=torch.bool)
    for i in range(len(prompt_strs)):
        raw_mask[i, :prompt_length_index[i]] = False #prompt部分为True
        raw_mask[i, prompt_length_index[i]:output_length_index[i]] = True #output部分为True
        raw_mask[i, output_length_index[i]:] = False #padding部分为False

    return {"input_ids": full_sequence_ids[:, :-1], "labels": full_sequence_ids[:, 1:], "response_mask": raw_mask[:, 1:]}
