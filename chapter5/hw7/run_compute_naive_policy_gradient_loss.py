def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:

    loss = - raw_rewards_or_advantages * policy_log_probs
    return loss
    