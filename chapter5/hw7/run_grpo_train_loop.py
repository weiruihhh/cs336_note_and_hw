"""完整GRPO训练循环实现，包括奖励归一化、微批次训练步骤、验证评估和结果记录等功能。"""

import argparse
import builtins
import importlib
import inspect
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from unittest.mock import patch

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HW4_ROOT = PROJECT_ROOT / "hw4"
if str(HW4_ROOT) not in sys.path:
    sys.path.insert(0, str(HW4_ROOT))

from cs336_alignment.drgrpo_grader import (
    extract_answer,
    grade,
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from run_get_response_log_probs import run_get_response_log_probs
# hw4/run_tokenize_prompt_and_output.py uses this name in annotations.
builtins.PreTrainedTokenizerBase = PreTrainedTokenizerBase
from run_tokenize_prompt_and_output import run_tokenize_prompt_and_output
from run_compute_group_normalized_rewards import run_compute_group_normalized_rewards
from run_grpo_microbatch_train_step import run_grpo_microbatch_train_step
from run_masked_mean import run_masked_mean

#因为严格要求格式的话正确率会一直低到0，导致奖励稀疏，训练非常困难，所以我设计了一个 answer_focused 的奖励模式，放宽格式要求，优先按答案正确与否给奖励。这个模式下的 prompt 模板和停止字符串也做了相应调整，更加宽松地捕捉模型输出中的答案信息。
STRICT_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
    "and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning "
    "process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)
ANSWER_FOCUSED_PROMPT_TEMPLATE = (
    "You are a math assistant. Solve the user's question.\n"
    "Give the final answer in LaTeX as \\\\boxed{{...}} on the last line.\n"
    "User: {question}\n"
    "Assistant:"
)
STRICT_STOP_STRINGS = ["</answer>"] #严格在</answer>处停止
ANSWER_FOCUSED_STOP_STRINGS = ["\nUser:", "</answer>"] #除了 </answer>，还会在出现下一轮对话前缀 \nUser: 时停止。这样可以防止模型继续自导自演”轮对话，尽量把输出截在当前答案处。


@dataclass
class GRPOConfig:
    model_path: str = str((PROJECT_ROOT / "models" / "Qwen2.5-Math-1.5B").resolve())
    train_file: str = str((PROJECT_ROOT / "MATH" / "train.jsonl").resolve())
    val_file: str = str((PROJECT_ROOT / "MATH" / "val.jsonl").resolve())
    output_dir: str = str((PROJECT_ROOT / "hw7" / "outputs").resolve())
    train_device: str = "cuda:7"
    eval_device: str = "cuda:6"
    seed: int = 42
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    optimizer_eps: float = 1e-4
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    max_grad_norm: float = 1.0
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    # strict(既要求答案正确性又要求格式) 或 answer_focused(放宽格式，优先答案正确性)。
    reward_mode: Literal["strict", "answer_focused"] = "answer_focused"
    cliprange: float = 0.2
    is_std_normalization: bool = True
    gpu_memory_utilization: float = 0.85
    val_every_steps: int = 10
    val_size: int = 1024
    eval_sampling_temperature: float = 0.0
    num_example_rollouts: int = 3
    example_rollout_steps: str = ""


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _ground_truth_candidates(ground_truth: Any) -> list[str]:
    """
    将ground_truth 转化成一个字符串列表，作为答案匹配的候选。这样可以兼容MATH里是数字、字符串或者列表等不同格式的情况。
    """
    if isinstance(ground_truth, (int, float)):
        return [str(ground_truth)]
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, list):
        return [str(x) for x in ground_truth]
    return [str(ground_truth)]


def _extract_answer_candidates(response: str) -> list[str]:
    """
    从模型的完整回复中尽可能地提取出一些“可能的答案片段”，作为奖励函数匹配正确答案的候选。
    """
    text = response.strip()
    candidates: list[str] = []

    if "<answer>" in text and "</answer>" in text:
        segment = text.split("<answer>")[-1].split("</answer>")[0].strip()
        if segment:
            candidates.append(segment)

    if "\\boxed" in text:
        try:
            boxed = extract_answer(text)
        except Exception:
            boxed = None
        if boxed:
            candidates.append(boxed.strip())

    if "</think>" in text:
        tail = text.split("</think>")[-1].strip()
        if tail:
            candidates.append(tail)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last_line = re.sub(r"^(final answer|answer)\s*[:：]\s*", "", lines[-1], flags=re.IGNORECASE).strip()
        if last_line:
            candidates.append(last_line)

        # 非 boxed 输出时，提取“最后一个简单数学片段”做答案候选。
        regexes = [
            r"(\\frac\{[^{}]+\}\{[^{}]+\})\s*$",
            r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$",
            r"([A-Za-z]\s*=\s*[-+]?\d+(?:\.\d+)?)\s*$",
        ]
        for pattern in regexes:
            m = re.search(pattern, last_line)
            if m:
                candidates.append(m.group(1).strip())

    # 从全文中抓取“最后出现的若干数学片段”，提高答案匹配召回率。
    global_patterns = [
        r"\\frac\{[^{}]+\}\{[^{}]+\}",
        r"-?\d+(?:\.\d+)?(?:/\d+)?",
    ]
    for pattern in global_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for item in matches[-3:]:
                candidates.append(item.strip())

    if text:
        candidates.append(text)

    unique: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return unique


def answer_focused_reward_fn(response: str, ground_truth: Any) -> dict[str, float]:
    """
    放宽格式限制，优先按答案是否正确给奖励。
    1) 优先走 question_only_reward_fn(对 boxed 答案友好)
    2) 再用若干候选答案片段尝试和ground truth匹配，只要有一个候选片段被判定为正确就给满奖励。
    """
    boxed_scores = question_only_reward_fn(response, ground_truth)
    if boxed_scores["reward"] > 0:
        return boxed_scores

    gt_candidates = _ground_truth_candidates(ground_truth)
    for candidate in _extract_answer_candidates(response):
        is_correct = False
        for gt in gt_candidates:
            try:
                is_correct |= bool(grade(candidate, gt, fast=True))
            except Exception:
                continue
        if is_correct:
            return {"format_reward": 0.0, "answer_reward": 1.0, "reward": 1.0}
    return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}


def resolve_reward_fn(mode: str) -> Callable[[str, Any], dict[str, float]]:
    """
    根据配置的 reward_mode 返回对应的奖励函数。
    strict要求模型输出完全符合格式要求才有奖励
    answer_focused放宽格式限制，更加关注答案正确与否。
    """
    if mode == "strict":
        return r1_zero_reward_fn
    if mode == "answer_focused":
        return answer_focused_reward_fn
    raise ValueError(f"Invalid reward mode: {mode}")


def resolve_prompt_template(mode: str) -> str:
    """
    选择对应的Prompt模板
    """
    if mode == "strict":
        return STRICT_PROMPT_TEMPLATE
    if mode == "answer_focused":
        return ANSWER_FOCUSED_PROMPT_TEMPLATE
    raise ValueError(f"Invalid reward mode: {mode}")


def resolve_stop_strings(mode: str) -> list[str]:
    """
    选择对应结束词
    """
    if mode == "strict":
        return STRICT_STOP_STRINGS
    if mode == "answer_focused":
        return ANSWER_FOCUSED_STOP_STRINGS
    raise ValueError(f"Invalid reward mode: {mode}")


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    llm_init_sig = inspect.signature(LLM.__init__)
    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        "enable_prefix_caching": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if "device" in llm_init_sig.parameters:
        llm_kwargs["device"] = device

    def _construct_llm() -> LLM:
        return LLM(**llm_kwargs)

    enable_profiling_patch = False
    try:
        worker_mod = importlib.import_module("vllm.worker.worker")
        worker_cls = getattr(worker_mod, "Worker", None)
        enable_profiling_patch = hasattr(worker_cls, "_assert_memory_footprint_increased_during_profiling")
    except Exception:
        enable_profiling_patch = False

    with world_size_patch:
        if enable_profiling_patch:
            with patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            ):
                return _construct_llm()
        return _construct_llm()


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    将当前策略模型的权重加载到 vLLM 实例中，以确保 vLLM 采样时使用的是最新的策略。
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def sample_prompt_batch(
    train_examples: list[dict[str, Any]],
    n_prompts_per_rollout_batch: int,
) -> tuple[list[str], list[str]]:
    """
    从训练示例中采样一个 batch 的 prompt 和对应的 ground truth 答案。每个 rollout batch 需要 n_prompts_per_rollout_batch 个不同的问题（prompt），每个问题会被 vLLM 采样 group_size 个答案。
    """
    if len(train_examples) >= n_prompts_per_rollout_batch:
        batch = random.sample(train_examples, k=n_prompts_per_rollout_batch)
    else:
        batch = [random.choice(train_examples) for _ in range(n_prompts_per_rollout_batch)]

    questions = [item["problem"] for item in batch]
    ground_truths = [item["answer"] for item in batch]
    return questions, ground_truths


def generate_rollouts(
    llm: LLM,
    questions: list[str],
    ground_truths: list[str],
    prompt_template: str,
    stop_strings: list[str],
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    使用 vLLM 生成 rollouts。对于输入的每个问题，vLLM 会生成 group_size 个不同的答案。返回值包含重复的 prompts（每个 prompt 重复 group_size 次）、rollout 的回答和对应的 ground truth 答案。
    """
    prompt_strs = [prompt_template.format(question=q) for q in questions]
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        n=group_size,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=stop_strings,
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompt_strs, sampling_params)

    repeated_prompts: list[str] = []
    rollout_responses: list[str] = []
    repeated_ground_truths: list[str] = []
    for prompt, gt, output_obj in zip(prompt_strs, ground_truths, outputs):
        for completion in output_obj.outputs:
            repeated_prompts.append(prompt)
            rollout_responses.append(completion.text)
            repeated_ground_truths.append(gt)
    return repeated_prompts, rollout_responses, repeated_ground_truths


def evaluate_validation_reward(
    llm: LLM,
    val_examples: list[dict[str, Any]],
    reward_fn: Callable[[str, Any], dict[str, float]],
    prompt_template: str,
    stop_strings: list[str],
    val_size: int,
    sampling_temperature: float,
    max_tokens: int,
) -> tuple[float, list[dict[str, Any]]]:
    """
    评估当前策略在验证集上的表现，返回平均奖励和部分示例。
    """
    # 只评估 val_size 个验证样本以节省时间
    if val_size > 0:
        eval_examples = val_examples[: min(len(val_examples), val_size)]
    else:
        eval_examples = val_examples

    prompts = [prompt_template.format(question=item["problem"]) for item in eval_examples]
    ground_truths = [item["answer"] for item in eval_examples]
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        n=1,
        max_tokens=max_tokens,
        min_tokens=4,
        stop=stop_strings,
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    rewards: list[float] = []
    examples: list[dict[str, Any]] = []
    for i, (output_obj, gt) in enumerate(zip(outputs, ground_truths)):
        response = output_obj.outputs[0].text
        reward = float(reward_fn(response, gt)["reward"])
        rewards.append(reward)
        if i < 3:
            examples.append(
                {
                    "question": eval_examples[i]["problem"],
                    "response": response,
                    "reward": reward,
                }
            )
    mean_reward = float(sum(rewards) / max(len(rewards), 1))
    return mean_reward, examples


def collect_rollout_examples(
    step: int,
    repeated_prompts: list[str],
    rollout_responses: list[str],
    raw_rewards: Tensor,
    group_size: int,
    num_example_rollouts: int,
) -> list[dict[str, Any]]:
    """
    从当前的 rollout batch 中收集一些示例，用于记录和分析训练过程中模型输出和奖励的变化趋势。对于每个组（group_size 个 rollouts），选择奖励最高的那个 rollout 作为该组的代表示例，记录它的 prompt、response 和 reward。
    """
    examples: list[dict[str, Any]] = []
    n_groups = len(rollout_responses) // group_size
    max_groups = min(n_groups, num_example_rollouts)
    for group_idx in range(max_groups):
        start = group_idx * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        best_offset = int(group_rewards.argmax().item())
        best_idx = start + best_offset
        examples.append(
            {
                "step": step,
                "group_idx": group_idx,
                "reward": float(raw_rewards[best_idx].item()),
                "prompt": repeated_prompts[best_idx],
                "response": rollout_responses[best_idx],
            }
        )
    return examples


def parse_int_csv(text: str, n_grpo_steps: int) -> set[int]:
    """
    把配置里的字符串 example_rollout_steps（比如 "1,50,100"）解析成 set[int]，用于决定“哪些 step 要保存 rollout 示例”。
    """
    if text.strip() == "":
        return {1, max(1, n_grpo_steps // 2), n_grpo_steps}
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def save_validation_plot(val_history: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[warn] matplotlib not installed, skip saving validation plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [entry["step"] for entry in val_history]
    rewards = [entry["mean_reward"] for entry in val_history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, marker="o")
    plt.title("Validation Reward vs. GRPO Steps")
    plt.xlabel("GRPO step")
    plt.ylabel("Mean validation reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_grpo_train_loop(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    llm: LLM,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    reward_fn: Callable[[str, Any], dict[str, float]],
    cfg: GRPOConfig,
) -> dict[str, Any]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available() and "cuda" in cfg.train_device:
        raise RuntimeError("CUDA is not available but `train_device` asks for CUDA.")

    device = torch.device(cfg.train_device if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.train()

    # 确保 microbatch 训练能整齐划分，不会省一截
    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    #每一道题会采样 group_size 个答案，所以 rollout_batch_size 也必须是 group_size 的整数倍，才能保证每个 rollouts batch 中的样本能被整齐划分成若干个 group。
    assert cfg.rollout_batch_size % cfg.group_size == 0, "rollout_batch_size must be divisible by group_size"
    assert cfg.train_batch_size >= cfg.group_size, "train_batch_size must be >= group_size" #保证一个训练 batch 至少容得下一整组样本，避免“比一组还小”的极端设置。

    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size
    prompt_template = resolve_prompt_template(cfg.reward_mode)
    stop_strings = resolve_stop_strings(cfg.reward_mode)

    #采样总回答数必须能被训练batch size 整除
    assert cfg.rollout_batch_size % cfg.train_batch_size == 0, (
        "rollout_batch_size must be divisible by train_batch_size"
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=cfg.optimizer_eps,
    )

    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
    #这个的目的就是为了记录几个快照作为example示例，类似以抽样的方式来观测训练过程中模型输出和奖励的变化趋势。
    example_steps = parse_int_csv(cfg.example_rollout_steps, cfg.n_grpo_steps)
    # 创建输出目录
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, Any] = {"train": [], "val": [], "examples": [], "config": asdict(cfg)}

    # 在训练之前先评估一次初始策略在验证集上的表现，记录初始奖励和示例。
    load_policy_into_vllm_instance(policy, llm)
    init_val_reward, init_examples = evaluate_validation_reward(
        llm=llm,
        val_examples=val_examples,
        reward_fn=reward_fn,
        prompt_template=prompt_template,
        stop_strings=stop_strings,
        val_size=cfg.val_size,
        sampling_temperature=cfg.eval_sampling_temperature,
        max_tokens=cfg.sampling_max_tokens,
    )
    #用作训练时的记录
    history["val"].append({"step": 0, "mean_reward": init_val_reward, "examples": init_examples})
    print(f"[val] step=0 mean_reward={init_val_reward:.4f}")

    optimizer_update_idx = 0
    for step in range(1, cfg.n_grpo_steps + 1):
        # Rollout with current policy.
        load_policy_into_vllm_instance(policy, llm)
        questions, ground_truths = sample_prompt_batch(train_examples, n_prompts_per_rollout_batch)
        repeated_prompts, rollout_responses, repeated_ground_truths = generate_rollouts(
            llm=llm,
            questions=questions,
            ground_truths=ground_truths,
            prompt_template=prompt_template,
            stop_strings=stop_strings,
            group_size=cfg.group_size,
            sampling_temperature=cfg.sampling_temperature,
            sampling_min_tokens=cfg.sampling_min_tokens,
            sampling_max_tokens=cfg.sampling_max_tokens,
        )

        if len(rollout_responses) != cfg.rollout_batch_size:
            raise RuntimeError(
                f"Expected {cfg.rollout_batch_size} rollouts, got {len(rollout_responses)}. "
                "Check vLLM sampling `n` and rollout grouping."
            )
        # 计算奖励并进行按组归一化
        advantages, raw_rewards, reward_metadata = run_compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.is_std_normalization,
        )
        reward_mean = float(reward_metadata.get("reward_mean", raw_rewards.float().mean().item()))
        reward_std = float(reward_metadata.get("reward_std", raw_rewards.float().std(unbiased=False).item()))

        if step in example_steps:
            history["examples"].extend(
                collect_rollout_examples(
                    step=step,
                    repeated_prompts=repeated_prompts,
                    rollout_responses=rollout_responses,
                    raw_rewards=raw_rewards,
                    group_size=cfg.group_size,
                    num_example_rollouts=cfg.num_example_rollouts,
                )
            )

        tokenized = run_tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device).float()
        advantages = advantages.to(device).unsqueeze(-1)
        raw_rewards = raw_rewards.to(device).unsqueeze(-1)

        old_log_probs = None
        if cfg.loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs = run_get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )["log_probs"].detach()

        step_losses: list[float] = []
        step_entropies: list[float] = []
        step_clip_fractions: list[float] = []

        # 最外层的循环目的是为了重复训练同一批 rollouts 多次（epochs_per_rollout_batch），以更充分地利用每次采样得到的数据。
        for _ in range(cfg.epochs_per_rollout_batch):
            #把原始的索引顺序打乱，增加训练的随机性和稳定性。每次迭代都会生成一个新的随机排列 perm，然后按照这个排列来划分训练 batch 和 microbatch。
            perm = torch.randperm(cfg.rollout_batch_size, device=device)
            # 第二层是按照batch一块一块练
            for batch_start in range(0, cfg.rollout_batch_size, cfg.train_batch_size):
                batch_indices = perm[batch_start : batch_start + cfg.train_batch_size]
                optimizer.zero_grad(set_to_none=True)
                
                #第三次循环，把大的batch再切成 microbatch，逐个计算 loss 并累积梯度，直到积累够 cfg.gradient_accumulation_steps 个 microbatches 的梯度才进行一次 optimizer.step() 更新模型。
                for micro_start in range(0, cfg.train_batch_size, micro_train_batch_size):
                    micro_indices = batch_indices[micro_start : micro_start + micro_train_batch_size]
                    outputs = run_get_response_log_probs(
                        model=policy,
                        input_ids=input_ids[micro_indices],
                        labels=labels[micro_indices],
                        return_token_entropy=True,
                    )
                    policy_log_probs = outputs["log_probs"]
                    token_entropy = outputs["token_entropy"]
                    micro_response_mask = response_mask[micro_indices]

                    loss, metadata = run_grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_response_mask,
                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                        loss_type=cfg.loss_type,
                        raw_rewards=raw_rewards[micro_indices] if cfg.loss_type == "no_baseline" else None,
                        advantages=advantages[micro_indices] if cfg.loss_type != "no_baseline" else None,
                        old_log_probs=old_log_probs[micro_indices] if old_log_probs is not None else None,
                        cliprange=cfg.cliprange if cfg.loss_type == "grpo_clip" else None,
                    )
                    step_losses.append(loss.item() * cfg.gradient_accumulation_steps)
                    step_entropies.append(run_masked_mean(token_entropy, micro_response_mask, dim=None).item())
                    #这个东西是用来记录有多少次模型的输出被 clip 掉了，反映了当前策略和旧策略的差距，clip 掉的越多说明更新越激进，可能不太稳定。
                    if "clip_fraction" in metadata:
                        step_clip_fractions.append(
                            run_masked_mean(metadata["clip_fraction"], micro_response_mask, dim=None).item()
                        )
                #原地梯度修改，防止梯度爆炸
                grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm).item())
                optimizer.step()
                optimizer_update_idx += 1

                history["train"].append(
                    {
                        "step": step,
                        "optimizer_update": optimizer_update_idx,
                        "loss": float(sum(step_losses[-cfg.gradient_accumulation_steps :]) / cfg.gradient_accumulation_steps),
                        "grad_norm": grad_norm,
                        "token_entropy": float(
                            sum(step_entropies[-cfg.gradient_accumulation_steps :]) / cfg.gradient_accumulation_steps
                        ),
                        "reward_mean": reward_mean,
                        "reward_std": reward_std,
                        "clip_fraction": float(
                            (sum(step_clip_fractions[-cfg.gradient_accumulation_steps :]) / cfg.gradient_accumulation_steps)
                            if step_clip_fractions
                            else 0.0
                        ),
                    }
                )

        if step % 5 == 0 or step == 1:
            train_loss = sum(step_losses) / max(len(step_losses), 1)
            train_entropy = sum(step_entropies) / max(len(step_entropies), 1)
            print(
                f"[train] step={step} reward_mean={reward_mean:.4f} "
                f"loss={train_loss:.4f} entropy={train_entropy:.4f}"
            )

        if step % cfg.val_every_steps == 0 or step == cfg.n_grpo_steps:
            load_policy_into_vllm_instance(policy, llm)
            val_reward, val_examples_snapshot = evaluate_validation_reward(
                llm=llm,
                val_examples=val_examples,
                reward_fn=reward_fn,
                prompt_template=prompt_template,
                stop_strings=stop_strings,
                val_size=cfg.val_size,
                sampling_temperature=cfg.eval_sampling_temperature,
                max_tokens=cfg.sampling_max_tokens,
            )
            history["val"].append(
                {
                    "step": step,
                    "mean_reward": val_reward,
                    "examples": val_examples_snapshot,
                }
            )
            print(f"[val] step={step} mean_reward={val_reward:.4f}")

    history_path = output_dir / "grpo_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_path = output_dir / "grpo_val_reward.png"
    save_validation_plot(history["val"], plot_path)
    print(f"[done] history={history_path} plot={plot_path}")
    return history


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = GRPOConfig()
    parser = argparse.ArgumentParser(description="Run GRPO train loop on MATH.")
    parser.add_argument("--model_path", type=str, default=defaults.model_path)
    parser.add_argument("--train_file", type=str, default=defaults.train_file)
    parser.add_argument("--val_file", type=str, default=defaults.val_file)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--train_device", type=str, default=defaults.train_device)
    parser.add_argument("--eval_device", type=str, default=defaults.eval_device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n_grpo_steps", type=int, default=defaults.n_grpo_steps)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--optimizer_eps", type=float, default=defaults.optimizer_eps)
    parser.add_argument("--advantage_eps", type=float, default=defaults.advantage_eps)
    parser.add_argument("--rollout_batch_size", type=int, default=defaults.rollout_batch_size)
    parser.add_argument("--group_size", type=int, default=defaults.group_size)
    parser.add_argument("--sampling_temperature", type=float, default=defaults.sampling_temperature)
    parser.add_argument("--sampling_min_tokens", type=int, default=defaults.sampling_min_tokens)
    parser.add_argument("--sampling_max_tokens", type=int, default=defaults.sampling_max_tokens)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=defaults.epochs_per_rollout_batch)
    parser.add_argument("--train_batch_size", type=int, default=defaults.train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument(
        "--loss_type",
        type=str,
        default=defaults.loss_type,
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default=defaults.reward_mode,
        choices=["strict", "answer_focused"],
    )
    parser.add_argument("--cliprange", type=float, default=defaults.cliprange)
    parser.add_argument(
        "--is_std_normalization",
        dest="is_std_normalization",
        action="store_true",
    )
    parser.add_argument(
        "--no-is_std_normalization",
        dest="is_std_normalization",
        action="store_false",
    )
    parser.set_defaults(is_std_normalization=defaults.is_std_normalization)
    parser.add_argument("--gpu_memory_utilization", type=float, default=defaults.gpu_memory_utilization)
    parser.add_argument("--val_every_steps", type=int, default=defaults.val_every_steps)
    parser.add_argument("--val_size", type=int, default=defaults.val_size)
    parser.add_argument("--eval_sampling_temperature", type=float, default=defaults.eval_sampling_temperature)
    parser.add_argument("--num_example_rollouts", type=int, default=defaults.num_example_rollouts)
    parser.add_argument("--example_rollout_steps", type=str, default=defaults.example_rollout_steps)
    return parser


def parse_args() -> GRPOConfig:
    parser = build_arg_parser()
    args = parser.parse_args()
    return GRPOConfig(**vars(args))


def main() -> None:
    cfg = parse_args()

    train_examples = load_jsonl(cfg.train_file)
    val_examples = load_jsonl(cfg.val_file)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    llm = init_vllm(
        model_id=cfg.model_path,
        device=cfg.eval_device,
        seed=cfg.seed,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    reward_fn = resolve_reward_fn(cfg.reward_mode)
    print(
        f"[info] reward_mode={cfg.reward_mode} loss_type={cfg.loss_type} "
        f"is_std_normalization={cfg.is_std_normalization}"
    )

    run_grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        llm=llm,
        train_examples=train_examples,
        val_examples=val_examples,
        reward_fn=reward_fn,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
