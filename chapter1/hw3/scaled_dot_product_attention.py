import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #如果mask为0，则将对应位置的score设置为-1e9
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return torch.matmul(attn_weights, V) # 将attn_weights与value相乘得到最终的输出