import torch
import torch.nn as nn
from causal_multi_head_attention_no_weight import CausalMultiHeadAttentionNoWeight
from RMSnorm import RMSNorm
from SwiGLU import SwiGLU
from rope import RoPE

class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        attn_q_proj_weight (torch.Tensor): 查询的权重
    """
    def __init__(self, d_model:int, n_heads:int, d_ff:int, max_seq_len:int, theta:float,device=None):
        super(TransformerBlock, self).__init__()
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.rms_norm1 = RMSNorm(d_model,eps=1e-5,device=device)
        self.rms_norm2 = RMSNorm(d_model,eps=1e-5,device=device)
        self.swiglu = SwiGLU(d_model,d_ff)
        self.causal_multi_head_attention = CausalMultiHeadAttentionNoWeight(d_model,n_heads,max_seq_len,theta,device)

    def forward(self,in_features:torch.Tensor):
        token_positions = torch.arange(in_features.shape[1],device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1,token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out