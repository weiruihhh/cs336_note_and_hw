import random
import torch
import json

class Dataset:
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            item = json.loads(line)
            # 使用 Alpaca 模板转换为标准的指令微调格式
            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {item['prompt']}

            ### Response:
            {item['response']}"""
            data.append(text)

        if shuffle:
            random.shuffle(data)

        all_token_ids = []
        for text in data:
            # 不自动添加特殊 token，手动控制
            token_ids = tokenizer.encode(text, add_special_tokens=False)#不同的tokenizer的特殊 token 不同，所以不自动添加，手动控制。
            # 在开头加 BOS
            if tokenizer.bos_token_id is not None:
                all_token_ids.append(tokenizer.bos_token_id)
            all_token_ids.extend(token_ids)
            # 在结尾加 EOS
            if tokenizer.eos_token_id is not None:
                all_token_ids.append(tokenizer.eos_token_id)
        self.all_token_ids = all_token_ids

    def __len__(self):
        return (len(self.all_token_ids) - 1) // self.seq_length

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Index out of range")
        input_ids = self.all_token_ids[i*self.seq_length:(i+1)*self.seq_length]
        labels = self.all_token_ids[i*self.seq_length+1:(i+1)*self.seq_length+1]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}