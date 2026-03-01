import torch


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute GRPO-Clip per-token loss and clip statistics."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clip(ratio, min=1.0 - cliprange, max=1.0 + cliprange)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    loss = -torch.minimum(unclipped, clipped)
    clip_fraction = (ratio != clipped_ratio).float()
    return loss, {"clip_fraction": clip_fraction}
