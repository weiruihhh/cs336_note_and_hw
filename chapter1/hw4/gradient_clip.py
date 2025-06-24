import torch
class GradientClip:
    def __init__(self, parameters,max_l2_norm,epslion=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epslion = epslion

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None] #我们求l2范数是对所有元素求的，所以要先把所有元素给flatten
        all_grads = torch.cat([grad.flatten() for grad in grads])
        grad_l2 = torch.norm(all_grads,2)
        if grad_l2 > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epslion)
            for grad in grads:
                grad.mul_(clip_coeff)

        