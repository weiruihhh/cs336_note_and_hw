import torch


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    masked_policy_log_probs = policy_log_probs * response_mask
    masked_normalize_policy_log_probs = run_masked_normalize(masked_policy_log_probs, response_mask, dim=1, normalize_constant=normalize_constant)
    
    scaled_loss = -(masked_normalize_policy_log_probs.mean()) / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss, {}