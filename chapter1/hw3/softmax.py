import torch
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

# def test_softmax():
#     x = torch.tensor([[1, 2, 3], [4, 5, 6]])
#     print(softmax(x, dim=0))
#     print(softmax(x, dim=1))

