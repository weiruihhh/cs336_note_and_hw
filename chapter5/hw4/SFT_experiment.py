import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import wandb
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from patch import patch
from transformers import PreTrainedModel, AutoModelForCausalLM
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# 训练设备配置
TRAIN_DEVICE = "cuda:7"
EVAL_DEVICE = "cuda:6"
MODEL_PATH = str((project_root / "models" / "Qwen2.5-Math-1.5B").resolve())

# 评估用的 prompt 模板
PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def init_vllm(model_id:str,device:str,seed:int,gpu_memory_utilization:float=0.85):
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

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def train_sft_loop(policy_model, train_loader, val_dataset, total_steps):
    """
    SFT 训练主循环
    
    Args:
        policy_model: 待训练的模型
        train_loader: 训练数据加载器
        val_dataset: 验证数据集，格式为 [{"problem": ..., "answer": ...}, ...]
        total_steps: 总训练步数
    """
    # 将模型移动到训练设备 (cuda:7)
    policy_model = policy_model.to(TRAIN_DEVICE)
    
    # 初始化 vLLM 实例（在评估 GPU 上: cuda:6）
    llm_engine = init_vllm(model_id=MODEL_PATH, device=EVAL_DEVICE, seed=42)
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    
    # 定义 WandB 指标
    wandb.define_metric("train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    for step, batch in enumerate(train_loader):
        if step >= total_steps:
            break
            
        # 训练 (TRAIN_DEVICE: cuda:7) 
        policy_model.train()
        
        # 将 batch 数据移动到训练设备
        batch = {k: v.to(TRAIN_DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        outputs = policy_model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # 梯度裁剪 (题目建议 clip value 1.0)
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        wandb.log({"train/loss": loss.item(), "train_step": step})
        
        if step % 10 == 0:
            print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")

        #  评估步骤 (EVAL_DEVICE: cuda:6)
        # 每 100 步评估一次
        if step % 100 == 0:
            print(f"Syncing weights to vLLM at step {step}...")
            
            # 将训练权重加载到 vLLM 中
            load_policy_into_vllm_instance(policy_model, llm_engine)
            
            # 使用 vLLM 运行验证集推理
            val_accuracy = run_validation(llm_engine, val_dataset)
            
            print(f"Step {step}, Validation Accuracy: {val_accuracy:.4f}")
            wandb.log({"eval/accuracy": val_accuracy, "eval_step": step})
    
    return policy_model


def run_validation(llm_engine, val_dataset):
    """
    使用 vLLM 对验证集进行推理并计算准确率
    
    Args:
        llm_engine: vLLM 推理引擎
        val_dataset: 验证数据集，格式为 [{"problem": ..., "answer": ...}, ...]
    
    Returns:
        float: 验证集准确率
    """
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],  # 遇到结束标签即停止
        include_stop_str_in_output=True
    )
    
    # 构建 prompts 和 ground_truths
    prompts = [PROMPTS_TEMPLATE.format(question=item["problem"]) for item in val_dataset]
    ground_truths = [item["answer"] for item in val_dataset]
    
    # 批量推理
    outputs = llm_engine.generate(prompts, sampling_params)
    
    correct_count = 0
    total_count = len(outputs)
    
    # 遍历结果并评分
    for i, output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]
        
        # 调用评分函数
        scores = r1_zero_reward_fn(generated_text, ground_truth)
        
        # 检查是否正确 (reward_fn 返回的 scores 中包含正确性指标)
        if scores.get("correctness", 0) > 0 or scores.get("reward", 0) > 0:
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy

if __name__ == "__main__":
    # 加载数据集
    train_dataset = load_dataset("json", data_files="MATH/train.jsonl")
    val_dataset = load_dataset("json", data_files="MATH/validation.jsonl")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,  # 先在 CPU 初始化，随后手动 .to(TRAIN_DEVICE)
        trust_remote_code=True,  # Qwen 系列常需要
    )
    # 将模型移动到训练设备 (cuda:7)
    policy_model.to(TRAIN_DEVICE)

    # 训练模型
    train_sft_loop(policy_model, train_loader, val_loader, total_steps=1000)
    # 保存模型
    policy_model.save_pretrained("sft_model")

    # 评估模型
    eval_accuracy = run_validation(policy_model, val_loader)
    print(f"Validation Accuracy: {eval_accuracy:.4f}")