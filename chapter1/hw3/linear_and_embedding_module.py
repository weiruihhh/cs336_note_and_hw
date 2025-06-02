import torch
from torch import nn

class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        # self.b = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

        # 对权重进行Xavier初始化
        std = 2 / (self.in_features + self.out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
    

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings # 词表vocab_size大小
        self.embedding_dim = embedding_dim # 词向量维度d_model
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids] # 从词表到词向量的映射的神经网络里读取token_ids输出词向量