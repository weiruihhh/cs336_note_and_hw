import torch
import torch.nn as nn
# from transformer_no_weight_block import TransformerBlock
from transformer_block_without_rmsnorm import TransformerBlock
from embedding import EmbeddingModule

class TransformerModuleWithoutRMSNorm(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, max_seq_len:int, theta:float,n_layers:int,vocab_size:int,device=None):
        super(TransformerModuleWithoutRMSNorm, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.device = device

        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, max_seq_len, theta,device) for _ in range(n_layers)])
        self.embedding_module = EmbeddingModule(vocab_size, d_model, device)
        self.linear_module = nn.Linear(d_model, vocab_size, bias=False,device=device)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.embedding_module(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.linear_module(x)
        return x

        