# from hw3.softmax import softmax
import torch
class CrossEntropyLoss:
    def __init__(self):
        """
        inputs: [batch, seq, vocab]
        targets: [batch, seq]
        """
        pass
    def logsoftmax(self,inputs): 
        inputs = inputs - torch.max(inputs,-1,keepdim=True)[0] #减去最大值，防止溢出
        # return torch.log(softmax(inputs,1))
        return inputs - torch.log(torch.sum(torch.exp(inputs),-1,keepdim=True)) #logsumexp
    def nll_loss(self,inputs,targets):
        return torch.mean(-inputs[range(inputs.shape[0]),targets]) #花式索引，range(inputs.shape[0])是行索引，targets是列索引，找到对应位置的值也就是概率，这里使用的是one-hot编码，所以直接找到对应位置的值
    
    def forward(self,inputs,targets):
        batch, seq, vocab = inputs.shape
        #先把原始输入展平为二维，然后计算log_softmax，最后计算nll_loss
        inputs = inputs.view(batch*seq, vocab)      # [batch*seq, vocab]
        targets = targets.view(batch*seq)           # [batch*seq]
        log_probs = self.logsoftmax(inputs)
        return self.nll_loss(log_probs,targets)
    
    # def forward(self,inputs,targets):
    #     batch, seq, vocab = inputs.shape
    #     inputs = inputs.view(-1, vocab)      # [batch*seq, vocab]
    #     targets = targets.view(-1)           # [batch*seq]
    #     # 下面用 log_softmax + nll_loss 或直接用 F.cross_entropy
    #     y_pred = softmax(inputs,1)
    #     p = y_pred[range(inputs.shape[0]),targets] #对于所有样本，选择真实标签对应的概率
    #     return -torch.mean(torch.log(p)) #用平均值来计算损失


