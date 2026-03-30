import torch
from transformers import PreTrainedTokenizerBase


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:

    # Alpaca-模板
    template = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{prompt}\n\n"
        "### Response:\n"
    )

    prompt_ids = tokenizer.encode(template, add_special_tokens=False)
    chosen_ids = tokenizer.encode(response_chosen, add_special_tokens=False)
    rejected_ids = tokenizer.encode(response_rejected, add_special_tokens=False)

    # 用不着加前缀了
    # if tokenizer.bos_token_id is not None:
    #     prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    
    if tokenizer.eos_token_id is not None:
        chosen_ids = chosen_ids + [tokenizer.eos_token_id]
        rejected_ids = rejected_ids + [tokenizer.eos_token_id]
    
    #求得chosen和rejected的logp
    chosen_logp = compute_response_logprobs(lm, prompt_ids, chosen_ids)
    rejected_logp = compute_response_logprobs(lm, prompt_ids, rejected_ids)

    #这里参考模型视为常量，不需要计算梯度，免得额外的计算开销。
    with torch.no_grad():
        chosen_ref_logp = compute_response_logprobs(lm_ref, prompt_ids, chosen_ids)
        rejected_ref_logp = compute_response_logprobs(lm_ref, prompt_ids, rejected_ids)

    # DPO loss的计算公式：-log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))
    pi_logratio = chosen_logp - rejected_logp
    ref_logratio = chosen_ref_logp - rejected_ref_logp
    dpo_loss = -torch.nn.functional.logsigmoid(beta * (pi_logratio - ref_logratio))
    return dpo_loss.squeeze(0)
    
def compute_response_logprobs(model, prompt_ids, response_ids):

    device = next(model.parameters()).device
    full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device) #这里套了一层[]把形状变成了(B,L)，但其实B=1，这样搞是为了统一格式。
    input_ids = full_ids[:,:-1]
    labels_ids = full_ids[:,1:]

    logits = model(input_ids=input_ids).logits
    token_logps = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, labels_ids.unsqueeze(-1))
        .squeeze(-1) #还是转变成(B,L)
    )
    #目前的token_logps是整个prompt+response的logp，我们只需要response部分的logp，所以要切片。response_ids的第一个token对应的logp是prompt_ids的最后一个token预测的，所以切片起点是len(prompt_ids)-1。
    response_logp = token_logps[:, len(prompt_ids)-1 : len(prompt_ids)-1 + len(response_ids)].sum(dim=-1) 
    return response_logp
