from run_compute_naive_policy_gradient_loss import run_compute_naive_policy_gradient_loss
from run_compute_grpo_clip_loss import run_compute_grpo_clip_loss

def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
"""把三种损失函数封装成一个函数"""
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss,metadata