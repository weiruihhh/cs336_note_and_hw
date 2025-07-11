import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self, data:List[int], batch_size:int, context_length:int, shuffle=True):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
    # def __iter__(self):
    #     if self.shuffle:
    #         self.data = random.shuffle(self.data)
    #     for i in range(0, len(self.data), self.batch_size):
    #         yield self.data[i:i+self.batch_size]

    def get_train_batch_data(self):
        """
        随机获取一个batch的数据，y 正好是 x 向左移动一个位置的结果。模型在看到 x 的第 i 个位置的输入时，需要努力预测出 y 在第 i 个位置的词元，这正是我们想要的“预测下一个词”的效果。
        """
        idxs = np.random.randint(0,self.data_len-self.context_length-1,size=(self.batch_size,)) # 随机选择一个batch_size大小的索引,最后的括号其实写不写都行，写上了就更明确输出是一个数组
        x = np.stack([self.data[i:i+self.context_length] for i in idxs])
        y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
        return torch.tensor(x),torch.tensor(y)
    
    def get_valid_batch_data_iter(self):
        """
        验证集的区别之处在于，不需要随机选择，而是直接从数据集中按顺序获取所有数据，并用迭代器返回
        """
        start_num = (self.data_len - self.context_length - 1) // self.batch_size # 表示有多少个batch
        for i in range(start_num):
            bias = i * self.batch_size     # 表示每一个batch开始的位置
            x = np.stack([self.data[bias:bias+self.context_length] for i in range(self.batch_size)])
            y = np.stack([self.data[bias+1:bias+self.context_length+1] for i in range(self.batch_size)])
            yield torch.tensor(x),torch.tensor(y)

    def __len__(self):
        """
        返回数据集的batch数量
        """
        return self.data_len // self.batch_size