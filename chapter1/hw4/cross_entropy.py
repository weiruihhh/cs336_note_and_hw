from hw3.softmax import softmax
import torch
class CrossEntropyLoss:
    def __init__(self,inputs,targets):
        self.inputs = inputs
        self.targets = targets
        self.vocab_size = inputs.shape[1]
        self.batch_size = inputs.shape[0]
    def forward(self):
        y_pred = softmax(self.inputs,1)
        p = y_pred[range(self.batch_size),self.targets]
        return -torch.sum(torch.log(p))


