import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
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