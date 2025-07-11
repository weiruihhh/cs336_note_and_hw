import torch
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, device=None):
        super(EmbeddingModule, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        self.embedding_matrix = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=self.device))
        std = 1
        torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        x: [batch, seq]
        output: [batch, seq, d_model]
        """
        # print('embedding input shape:', x.shape, 'output shape:', self.embedding_matrix[x].shape)
        return self.embedding_matrix[x]