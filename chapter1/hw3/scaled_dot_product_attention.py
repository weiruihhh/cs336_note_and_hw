import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention 是缩放点积注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    公式是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        Q: (batch_size, seq_len, d_k) 查询向量
        K: (batch_size, seq_len, d_k) 键向量
        V: (batch_size, seq_len, d_k) 值向量
        mask: (batch_size, seq_len, seq_len) 掩码
    output:
        out: (batch_size, seq_len, d_k) 输出的稠密向量
    """
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #如果mask为0，则将对应位置的score设置为-1e9
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return torch.matmul(attn_weights, V) # 将attn_weights与value相乘得到最终的输出