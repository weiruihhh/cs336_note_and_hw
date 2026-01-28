def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    # ratio = π/π_old = exp(log π - log π_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clip(ratio, min=1.0 - cliprange, max=1.0 + cliprange)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    res = -torch.minimum(unclipped, clipped)#注意这里要逐元素进行比较，不能用min,这个只针对一个输入张量。
    return res, {}