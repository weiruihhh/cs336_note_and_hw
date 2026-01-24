import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vllm import LLM, SamplingParams
from typing import Callable, List
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


PROMPTS_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
eval_sampling_params: SamplingParams,
ground_truths: List[str],
output_file: str
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # 1. 批量推理
    # outputs 是 RequestOutput 对象列表
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    correct_count = 0
    format_error_count = 0
    
    # 2. 遍历结果并评分
    for i, output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]
        
        # 调用提供的评分函数解析答案并打分
        scores = reward_fn(generated_text, ground_truth)
        
        result_entry = {
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "scores": scores
        }
        results.append(result_entry)
        
    # 3. 保存结果到磁盘 
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            

if __name__ == "__main__":
    # HF repo id（会触发在线下载）
    # MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
    # 使用本地模型目录
    MODEL_PATH = str((project_root / "models" / "Qwen2.5-Math-1.5B").resolve())

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"], # 遇到结束标签即停止
        include_stop_str_in_output=True
    )

    validation_file = project_root / "MATH" / "validation.jsonl"
    with open(validation_file, "r") as f:
        data = [json.loads(line) for line in f]
    #提取出问题和答案
    prompts = [PROMPTS_TEMPLATE.format(question=item["problem"]) for item in data]
    ground_truths = [item["answer"] for item in data]
    
    llm = LLM(model=MODEL_PATH)
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params, ground_truths, "results.jsonl")