import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False) #注意讲义上是Wx这种列向量的形式出现，简单写法self.w1(x)就是按照行向量了，因此要形状反过来。
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))