import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    RoPE 是旋转位置编码，它通过将输入的稠密向量旋转来稳定训练。
    公式是：
    out = x * cos(theta * position) - x * sin(theta * position)
    Args:
        theta (float): 底数超参数
        d_k (int): 输入的维度，也就是d_model
        max_seq_len (int): 最大序列长度
        device (torch.device): 设备
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        token_positions: (batch_size, seq_len) 每个token的位置信息
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta #这个是RoPE的底数超参数，不是直接的角度
        self.d_k = d_k #d_k就是d_model,即嵌入之后的稠密向量，它必须为偶数
        self.max_seq_len = max_seq_len
        self.device = device
        #计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        #记录每个token的位置信息
        positions = torch.arange(self.max_seq_len)
        #计算正弦和余弦
        sinusoids = torch.outer(positions, freqs) #outer是外积，即每个位置都与每个频率相乘 shape: [max_seq_len, d_k//2]
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False) #利用register_buffer表示这是固定的，不需要学习
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 这里的x是输入的稠密向量，token_positions是token的位置信息
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        cos = cos.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]
        sin = sin.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]

        #  这里还是分奇偶数写容易理解
        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin # 偶数位置乘以cos，奇数位置乘以sin
        output2 = x_part1 * sin + x_part2 * cos # 偶数位置乘以sin，奇数位置乘以cos

        # out = torch.cat([output1, output2], dim=-1) # shape: [batch,  max_seq_len, d_k]
        out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2] #用stack能巧妙的把奇数和偶数交叉在一起，cat就不行
        out = out.flatten(-2)  # [batch, seq_len, d_k]
        return out
