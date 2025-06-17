import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from linear_and_embedding_module import LinearModule,EmbeddingModule
from rope import RoPE
from RMSnorm import RMSNorm
from SwiGLU import SwiGLU
from softmax import softmax

class TransformerLM(nn.Module):
    """
    TransformerLM 是整个训练过程的封装，它把包含Embedding、TransformerBlock、RMSNorm、LinearModule等组件包装在一起，形成一个完整的Transformer语言模型。
    Args:
        vocab_size (int): 词表大小
        context_length (int): 上下文长度
        d_model (int): 输入的维度，也就是d_model
        num_layers (int): 层数
        num_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        rope_theta (float): 底数超参数
        weights (dict[str, torch.Tensor]): 权重
    input:
        in_indices (torch.Tensor): 输入的索引
    output:
        out_linear (torch.Tensor): 输出的线性层
    """
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int, num_heads:int, d_ff:int, rope_theta:float, weights:dict[str, torch.Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

    def forward(self, in_indices):
        # print("in_indices.shape",in_indices.shape)
        # print("in_indices",in_indices)
        embedding_module = EmbeddingModule(self.vocab_size,self.d_model,device=None)
        embedding_module.load_state_dict({"embedding_matrix":self.weights["token_embeddings.weight"]})
        embedding = embedding_module(in_indices)
        #对于每一个TransformerBlock，我们都需要从权重中获取对应的权重
        for layer in range(self.num_layers):
            attn_q_proj_weight = self.weights[f"layers.{layer}.attn.q_proj.weight"]
            attn_k_proj_weight = self.weights[f"layers.{layer}.attn.k_proj.weight"]
            attn_v_proj_weight = self.weights[f"layers.{layer}.attn.v_proj.weight"]
            attn_o_proj_weight = self.weights[f"layers.{layer}.attn.output_proj.weight"]
            ln1_weight = self.weights[f"layers.{layer}.ln1.weight"]
            ln2_weight = self.weights[f"layers.{layer}.ln2.weight"]
            ffn_w1_weight = self.weights[f"layers.{layer}.ffn.w1.weight"]
            ffn_w2_weight = self.weights[f"layers.{layer}.ffn.w2.weight"]
            ffn_w3_weight = self.weights[f"layers.{layer}.ffn.w3.weight"]   

            transformer_block = TransformerBlock(self.d_model,self.num_heads,self.d_ff,self.context_length,self.rope_theta,attn_q_proj_weight,attn_k_proj_weight,attn_v_proj_weight,attn_o_proj_weight,ln1_weight,ln2_weight,ffn_w1_weight,ffn_w2_weight,ffn_w3_weight,device=None)
            embedding = transformer_block(embedding)
        # print("transformer_block_out.shape",transformer_block_out.shape)
        transformer_block_out = embedding   
        #最后需要一个RMSNorm来归一化输出，然后通过一个线性层得到最终的输出
        rms_norm = RMSNorm(self.d_model,eps=1e-5,device=None)
        rms_norm.load_state_dict({"weight":self.weights["ln_final.weight"]})
        out_norm = rms_norm(transformer_block_out)

        linear_module = LinearModule(self.d_model,self.vocab_size,device=None)
        linear_module.load_state_dict({"W":self.weights["lm_head.weight"]})
        out_linear = linear_module(out_norm)
        # out = softmax(out_linear,dim=-1) #注意最后作业不需要softmax归一化
        return out_linear







