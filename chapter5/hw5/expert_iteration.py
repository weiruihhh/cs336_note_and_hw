"""在MATH数据集上运行专家迭代。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
import torch
import wandb
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# 将训练和评估分别放在不同的GPU上
TRAIN_DEVICE = "cuda:7"
EVAL_DEVICE = "cuda:6"
MODEL_PATH = str((project_root / "models" / "Qwen2.5-Math-1.5B").resolve())

# Prompt模板
PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """初始化vLLM实例"""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    """将策略权重加载到vLLM实例中。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def generate_rollouts(
    llm: LLM,
    questions: List[str],
    n_rollout: int,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> List[List[str]]:
    """
    为每个问题生成多个完成。

    Args:
        llm: vLLM实例
        questions: 问题列表
        n_rollout: 每个问题的完成数量
        temperature: 采样温度
        max_tokens: 最大令牌数

    Returns:
        包含n_rollout个完成的列表的列表
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        n=n_rollout,  # 为每个提示生成n_rollout个样本
        stop=["</answer>"],
        include_stop_str_in_output=True,
        min_tokens=10,  # 避免生成空字符串
    )

    # 格式化提示
    prompts = [PROMPTS_TEMPLATE.format(question=q) for q in questions]

    # 生成
    outputs = llm.generate(prompts, sampling_params)

    # 提取完成
    all_completions = []
    for output_obj in outputs:
        completions = [out.text for out in output_obj.outputs]
        all_completions.append(completions)

    return all_completions


def filter_correct_completions(
    questions: List[str],
    all_completions: List[List[str]],
    ground_truths: List[str],
) -> List[Dict[str, str]]:
    """
    过滤回复，只保留满足奖励值要求的回复。

    Args:
        questions: 问题列表
        all_completions: 包含回复列表的列表
        ground_truths: ground_truths答案列表

    Returns:
        包含'question'和'completion'键的列表
    """
    filtered_data = []

    for question, completions, gt in zip(questions, all_completions, ground_truths):
        for completion in completions:
            # 计算奖励
            scores = r1_zero_reward_fn(completion, gt)

            # 如果奖励值大于0，则保留
            if scores["reward"] > 0:
                filtered_data.append({
                    "question": question,
                    "completion": completion,
                })

    return filtered_data


class SFTDataset(Dataset):
    """使用分词的SFT训练的数据集。"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
 
        # 格式化提示和完成
        prompt = PROMPTS_TEMPLATE.format(question=item["question"])
        full_text = prompt + item["completion"]

        # 分词
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]

        # 创建标签（移位1）
        labels = input_ids.copy()

        # 掩码提示令牌在标签中
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_length = len(prompt_encoding["input_ids"])

        # 将提示令牌设置为-100（在损失中忽略）
        labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(labels[1:], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    """DataLoader的合并函数，将一个batch中的数据合并成一个tensor"""
    # 找到最大长度
    max_len = max(len(item["input_ids"]) for item in batch)

    # 填充序列
    input_ids = []
    labels = []

    for item in batch:
        input_len = len(item["input_ids"])
        pad_len = max_len - input_len

        # 填充input_ids
        padded_input = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        # 填充labels
        padded_label = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)#这里填充-100是因为在计算损失时，我们需要忽略掉的padding部分。tokenizer.pad_token_id这个是模型的忽略标记
        ])
        labels.append(padded_label)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


def sft_train(
    policy_model,
    train_data: List[Dict],
    tokenizer,
    num_steps: int,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    gradient_clip: float = 1.0,
):
    """
    在过滤后的数据上训练策略模型使用SFT。

    Args:
        policy_model: 要训练的模型
        train_data: 包含'question'和'completion'键的列表
        tokenizer: 分词器
        num_steps: 训练步数
        batch_size: 批次大小
        learning_rate: 学习率
        gradient_clip: 梯度裁剪值
    """
    # 创建数据集和DataLoader
    dataset = SFTDataset(train_data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    # 优化器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    # 训练循环
    policy_model.train()#开启训练模式
    step = 0
    total_loss = 0.0

    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break

            # 移动到设备
            batch = {k: v.to(TRAIN_DEVICE) for k, v in batch.items()}

            # 前向传播
            outputs = policy_model(**batch)
            loss = outputs.loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), gradient_clip)

            # 优化器步
            optimizer.step()
            optimizer.zero_grad()

            # 记录
            total_loss += loss.item()
            step += 1

            if step % 10 == 0:
                avg_loss = total_loss / 10
                print(f"  SFT Step {step}/{num_steps}, Loss: {avg_loss:.4f}")
                wandb.log({"sft/loss": avg_loss, "sft/step": step})
                total_loss = 0.0


def evaluate_model(
    llm: LLM,
    val_data: List[Dict],
    num_samples: int = None,
) -> float:
    """
    在验证集上评估模型。

    Args:
        llm: vLLM实例   
        val_data: 验证数据
        num_samples: 评估样本数量（None表示所有）

    Returns:
        准确率
    """
    if num_samples is not None:
        val_data = val_data[:num_samples]

    questions = [item["problem"] for item in val_data]
    ground_truths = [item["answer"] for item in val_data]

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # temperature=0.0表示不再随机采样，只要概率最大的那个结果；固定 temperature=0.0 后，llm.generate 基本变成确定性解码，便于比较 iteration 0/1/2/... 的真实趋势；temperature>0 时同一个题目每次可能答得不一样，eval/accuracy 方差会很大，迭代间很难判断“模型真进步了还是随机波动”。
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts = [PROMPTS_TEMPLATE.format(question=q) for q in questions]
    outputs = llm.generate(prompts, sampling_params)

    # 计算准确率
    correct = 0
    for output_obj, gt in zip(outputs, ground_truths):
        generated_text = output_obj.outputs[0].text # 获取生成的文本
        scores = r1_zero_reward_fn(generated_text, gt)
        if scores["reward"] > 0:
            correct += 1

    accuracy = correct / len(val_data)
    return accuracy


def expert_iteration(
    model_path: str,
    train_file: str,
    val_file: str,
    n_rollout: int = 6,
    mu_sft: int = 100,
    k_steps: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    train_samples_per_iter: int = 500,
    val_samples: int = 200,
    seed: int = 42,
):
    """
    在MATH数据集上运行专家迭代。

    Args:
        model_path: 基础模型的路径
        train_file: 训练数据的路径
        val_file: 验证数据的路径
        n_rollout: 每个问题的完成数量
        mu_sft: 每次迭代的SFT步数
        k_steps: 专家迭代轮数
        batch_size: SFT的批量大小
        learning_rate: SFT的学习率
        train_samples_per_iter: 每次迭代的训练样本数量
        val_samples: 验证级采样样本数量
        seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)

    # 初始化WandB
    wandb.init(
        project="expert-iteration-math",
        config={
            "n_rollout": n_rollout,
            "mu_sft": mu_sft,
            "k_steps": k_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples_per_iter": train_samples_per_iter,
        }
    )

    # 加载数据
    with open(train_file, 'r') as f:
        train_data = [json.loads(line) for line in f]

    with open(val_file, 'r') as f:
        val_data = [json.loads(line) for line in f]

    print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")

    # 加载模型和分词器
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )
    policy_model.to(TRAIN_DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化vLLM
    llm = init_vllm(model_path, EVAL_DEVICE, seed)

    # 初始评估
    initial_acc = evaluate_model(llm, val_data, val_samples)
    print(f"Initial accuracy: {initial_acc:.4f}")
    wandb.log({"eval/accuracy": initial_acc, "iteration": 0})

    # 专家迭代循环
    for iteration in range(1, k_steps + 1):
        print(f"Expert Iteration Round {iteration}/{k_steps}")

        # 采样训练问题
        random.seed(seed + iteration) #每一轮抽到的训练子集不同，避免每次迭代都用同一个子集，导致模型过拟合
        sampled_train = random.sample(train_data, min(train_samples_per_iter, len(train_data)))

        questions = [item["problem"] for item in sampled_train]
        ground_truths = [item["answer"] for item in sampled_train]

        # 生成回复
        print(f"Generating {n_rollout} rollouts for {len(questions)} questions...")
        all_completions = generate_rollouts(llm, questions, n_rollout)

        # 过滤满足奖励值要求的回复
        print("Filtering correct completions...")
        filtered_data = filter_correct_completions(questions, all_completions, ground_truths)
        print(f"Filtered {len(filtered_data)} correct completions from {len(questions) * n_rollout} total")

        if len(filtered_data) == 0:
            print("No correct completions found! Skipping SFT...")
            continue

        wandb.log({
            "rollout/total_completions": len(questions) * n_rollout,
            "rollout/correct_completions": len(filtered_data),
            "rollout/success_rate": len(filtered_data) / (len(questions) * n_rollout),
            "iteration": iteration,
        })

        # SFT训练
        print(f"Training on filtered data for {mu_sft} steps...")
        sft_train(
            policy_model,
            filtered_data,
            tokenizer,
            num_steps=mu_sft,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # 加载更新后的权重到vLLM，这种形式就可以不用重复初始化vLLM了
        print("Loading updated weights into vLLM...")
        load_policy_into_vllm_instance(policy_model, llm)

        # 评估
        print("Evaluating...")
        accuracy = evaluate_model(llm, val_data, val_samples)
        print(f"Iteration {iteration} accuracy: {accuracy:.4f}")
        wandb.log({"eval/accuracy": accuracy, "iteration": iteration})

        # 保存检查点
        checkpoint_dir = project_root / "hw5" / f"checkpoint_iter_{iteration}"
        policy_model.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

    # 最终评估在完整的验证集上
    print("\nFinal evaluation on full validation set...")
    final_acc = evaluate_model(llm, val_data, None)
    print(f"Final accuracy: {final_acc:.4f}")
    wandb.log({"eval/final_accuracy": final_acc})

    # 保存最终模型
    final_model_dir = project_root / "hw5" / "final_model"
    policy_model.save_pretrained(final_model_dir)
    print(f"Saved final model to {final_model_dir}")

    wandb.finish()

    return policy_model


if __name__ == "__main__":
    train_file = str(project_root / "MATH" / "train.jsonl")
    val_file = str(project_root / "MATH" / "val.jsonl")

    expert_iteration(
        model_path=MODEL_PATH,
        train_file=train_file,
        val_file=val_file,
        n_rollout=6,
        mu_sft=100,
        k_steps=3,
        batch_size=8,
        learning_rate=1e-5,
        train_samples_per_iter=500,
        val_samples=200,
        seed=114514,
    )
