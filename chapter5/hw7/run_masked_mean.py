def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # return torch.mean(tensor * mask,dim=dim)  这样写是把mask的总长度作为分母了。
    # 只对 mask=1 的位置求和；分母是 mask=1 的个数，不是该维总长度
    masked_sum = (tensor * mask).sum(dim=dim)
    masked_count = mask.sum(dim=dim)
    return masked_sum / masked_count 