import torch
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    softmax 是归一化技术，它通过将输入除以输入的指数的平均值来稳定训练。
    公式是：
    out = exp(x - x_max) / sum(exp(x - x_max)) 
    Args:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        dim: 归一化的维度
    output:
        out: (batch_size, seq_len, d_model) 归一化后的稠密向量
    """
    x_max = x.max(dim=dim, keepdim=True)[0] # 这里x_max是batch_size, seq_len, 1
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


