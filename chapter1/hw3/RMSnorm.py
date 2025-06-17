import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    RMSNorm 是归一化技术，它通过将输入除以输入的平方根的平均值来稳定训练。
    公式是：
    x_norm = x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_norm = x_norm * weight
    
    Args:
        d_model (int): 经过embedding层之后，每个token的维度
        eps (float): 一个很小的常数，用于避免除以零
        device (torch.device): 设备
        dtype (torch.dtype): 数据类型
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
    output:
        x_norm: (batch_size, seq_len, d_model) 归一化后的稠密向量
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #weight对应缩放参数 gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 题目要求对于不同的精度要先转换为float32再进行归一化，最后再转换回原来的精度
        origin_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        norm_x = x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_norm = norm_x.to(origin_dtype)
        return x_norm * self.weight