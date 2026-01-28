import torch
def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    # 1. raw rewards，用 zip 可以省去手写索引
    raw_rewards = torch.tensor([
        reward_fn(r, g)["reward"] for r, g in zip(rollout_responses, repeated_ground_truths)
    ])

    # 2. 按组 reshape 成 (n_groups, group_size)，用向量化按组减均值、除标准差
    r = raw_rewards.reshape(-1, group_size)
    centered = r - r.mean(dim=1, keepdim=True)
    if normalize_by_std:
        normalized = centered / (r.std(dim=1, unbiased=True, keepdim=True) + advantage_eps)
    else:
        normalized = centered

    return normalized.flatten(), raw_rewards, {}