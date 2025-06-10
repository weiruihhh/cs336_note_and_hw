import torch
import torch.nn as nn

class RoPE(nn.Module):
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
        sinusoids = torch.outer(positions, freqs) #outer是外积，即每个位置都与每个频率相乘
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False) #利用register_buffer表示这是固定的，不需要学习
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 这里的x是输入的稠密向量，token_positions是token的位置信息
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        x1 = x[...,0:2]
        x2 = x[...,1:2]

        output1 = x1 * cos - x2 * sin
        output2 = x1 * sin + x2 * cos #这里x[:, :, self.d_k//2:] * cos是x2 * cos
        out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2]
        out = out.flatten(-2)  # [batch, seq_len, d_k]
        return out
    

# if __name__ == "__main__":
#     x = torch.randn(1, 10, 128)
#     token_positions = torch.arange(10)
#     rope = RoPE(theta=10000, d_k=128, max_seq_len=10)
#     output = rope(x, token_positions)
#     print(output.shape)