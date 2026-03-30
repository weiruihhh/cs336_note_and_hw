import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dpo_loss import compute_response_logprobs

#训练数据集
DEFAULT_HH_TRAIN_FILES = (
    "harmless-base/train.jsonl",
    "helpful-base/train.jsonl",
    "helpful-online/train.jsonl",
    "helpful-rejection-sampled/train.jsonl",
)

#把数据集转换成 ALPACA 模板格式
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
    "{response}"
)

# 判断“是不是单轮 Human->Assistant”，匹配这种格式的数据
TURN_PATTERN = re.compile(r"(Human|Assistant):\s*(.*?)(?=(?:\n\n(?:Human|Assistant):)|\Z)", re.S)

# frozen=True 表示这个 dataclass 实例创建后不可修改（近似“只读对象”）
@dataclass(frozen=True)
class PreferenceExample:
    instruction: str
    chosen: str
    rejected: str
    source: str


@dataclass(frozen=True)
class TokenizedPreferenceExample:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]
    source: str


@dataclass(frozen=True)
class TrainConfig:
    model_path: str
    hh_data_dir: str
    output_dir: str
    train_files: tuple[str, ...]
    train_device: str
    ref_device: str
    dtype: str
    max_seq_len: int
    num_epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    lr: float
    beta: float
    weight_decay: float
    rmsprop_alpha: float
    rmsprop_eps: float
    val_size: int
    seed: int
    log_every: int
    eval_every: int
    save_every: int
    clip_grad_norm: float
    gradient_checkpointing: bool
    max_train_examples: int
    max_val_examples: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="DPO training on Anthropic HH")
    parser.add_argument("--model-path", type=str, required=True, help="SFT checkpoint path.")
    parser.add_argument("--hh-data-dir", type=str, default="hh-dpo")
    parser.add_argument("--output-dir", type=str, default="hw/supplement/DPO/outputs/default_run")
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=list(DEFAULT_HH_TRAIN_FILES),
        help="Relative paths under --hh-data-dir.",
    ) #可手动传入指定的训练文件，默认是这四个文件
    parser.add_argument("--train-device", type=str, default="cuda:0")
    parser.add_argument("--ref-device", type=str, default="cuda:1")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1) #实际批次
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64) #实际批次*梯度累计值 = 等效批次
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-8)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max-val-examples", type=int, default=0, help="0 means no cap.")

    args = parser.parse_args()

    return TrainConfig(
        model_path=args.model_path,
        hh_data_dir=args.hh_data_dir,
        output_dir=args.output_dir,
        train_files=tuple(args.train_files),
        train_device=args.train_device,
        ref_device=args.ref_device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        beta=args.beta,
        weight_decay=args.weight_decay,
        rmsprop_alpha=args.rmsprop_alpha,
        rmsprop_eps=args.rmsprop_eps,
        val_size=args.val_size,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        clip_grad_norm=args.clip_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str) -> torch.dtype | None:
    if dtype_name == "auto":
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def parse_single_turn(conversation: str) -> tuple[str, str] | None:
    """捕获 Human -> Assistant 这种格式的单轮样本，如果是多轮样本就只取第一轮的结果"""
    turns = [(role.strip(), text.strip()) for role, text in TURN_PATTERN.findall(conversation)]
    if len(turns) != 2:
        return None
    if turns[0][0] != "Human" or turns[1][0] != "Assistant":
        return None
    instruction, assistant_response = turns[0][1], turns[1][1]
    if not instruction or not assistant_response:
        return None
    return instruction, assistant_response


def load_hh_preferences(hh_data_dir: Path, train_files: Iterable[str]) -> list[PreferenceExample]:
    """
    将原始样本转换为标准样本
    """
    examples: list[PreferenceExample] = []
    skipped_multi_turn = 0
    skipped_prompt_mismatch = 0

    for rel_path in train_files:
        full_path = hh_data_dir / rel_path
        source = Path(rel_path).parts[0] if Path(rel_path).parts else rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"HH file not found: {full_path}")

        with full_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chosen_turn = parse_single_turn(row["chosen"])
                rejected_turn = parse_single_turn(row["rejected"])

                if chosen_turn is None or rejected_turn is None:
                    skipped_multi_turn += 1
                    continue

                chosen_instruction, chosen_response = chosen_turn
                rejected_instruction, rejected_response = rejected_turn

                if chosen_instruction.strip() != rejected_instruction.strip():
                    skipped_prompt_mismatch += 1
                    continue

                examples.append(
                    PreferenceExample(
                        instruction=chosen_instruction.strip(),
                        chosen=chosen_response.strip(),
                        rejected=rejected_response.strip(),
                        source=source,
                    )
                )

    print(
        f"[data] loaded={len(examples)} "
        f"skipped_multi_turn={skipped_multi_turn} "
        f"skipped_prompt_mismatch={skipped_prompt_mismatch}"
    )
    return examples


def tokenize_examples(
    examples: list[PreferenceExample],
    tokenizer,
    max_seq_len: int,
) -> tuple[list[TokenizedPreferenceExample], int]:
    """
    将偏好样本转换为分词后的token ids
    """
    tokenized: list[TokenizedPreferenceExample] = []
    skipped_too_long = 0

    eos_token_id = tokenizer.eos_token_id

    for ex in examples:
        prompt_text = ALPACA_TEMPLATE.format(instruction=ex.instruction, response="")
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        chosen_ids = tokenizer.encode(ex.chosen, add_special_tokens=False)
        rejected_ids = tokenizer.encode(ex.rejected, add_special_tokens=False)

        if eos_token_id is not None:
            chosen_ids = chosen_ids + [eos_token_id]
            rejected_ids = rejected_ids + [eos_token_id]

        # token ids 的长度必须在模型的最大上下文长度范围内，才能被模型处理。
        if len(prompt_ids) < 1:
            continue

        if len(prompt_ids) + len(chosen_ids) > max_seq_len:
            skipped_too_long += 1
            continue
        if len(prompt_ids) + len(rejected_ids) > max_seq_len:
            skipped_too_long += 1
            continue

        tokenized.append(
            TokenizedPreferenceExample(
                prompt_ids=prompt_ids,
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
                source=ex.source,
            )
        )

    return tokenized, skipped_too_long


def split_train_val(
    examples: list[TokenizedPreferenceExample],
    val_size: int,
    seed: int,
) -> tuple[list[TokenizedPreferenceExample], list[TokenizedPreferenceExample]]:
    """
    把训练集切分成训练集和验证集，测试集最后再评估。
    """
    if len(examples) <= val_size:
        raise ValueError(f"Not enough examples ({len(examples)}) for val_size={val_size}.")
    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])

    train_examples = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val_examples = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train_examples, val_examples


def compute_response_logprob(
    model: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
) -> torch.Tensor:
    """返回每一个样本的 log p(response | prompt)"""
    return compute_response_logprobs(model, prompt_ids, response_ids).squeeze(0)


def compute_per_instance_dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ex: TokenizedPreferenceExample,
    beta: float,
) -> torch.Tensor:
    """
    计算DPO损失
    """
    policy_device = next(policy_model.parameters()).device

    chosen_logp = compute_response_logprob(policy_model, ex.prompt_ids, ex.chosen_ids)
    rejected_logp = compute_response_logprob(policy_model, ex.prompt_ids, ex.rejected_ids)

    with torch.no_grad():
        chosen_ref_logp = compute_response_logprob(ref_model, ex.prompt_ids, ex.chosen_ids).to(policy_device)
        rejected_ref_logp = compute_response_logprob(ref_model, ex.prompt_ids, ex.rejected_ids).to(policy_device)

    pi_logratio = chosen_logp - rejected_logp
    ref_logratio = chosen_ref_logp - rejected_ref_logp
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio))


@torch.no_grad()
def evaluate(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    val_examples: list[TokenizedPreferenceExample],
    beta: float,
) -> dict[str, float]:
    policy_model.eval()
    total = len(val_examples)
    correct = 0
    loss_sum = 0.0

    for ex in val_examples:
        #判断模型是否会给 chosen的样本更高的log概率
        chosen_logp = compute_response_logprob(policy_model, ex.prompt_ids, ex.chosen_ids)
        rejected_logp = compute_response_logprob(policy_model, ex.prompt_ids, ex.rejected_ids)
        if chosen_logp.item() > rejected_logp.item():
            correct += 1

        chosen_ref_logp = compute_response_logprob(ref_model, ex.prompt_ids, ex.chosen_ids).to(chosen_logp.device)
        rejected_ref_logp = compute_response_logprob(ref_model, ex.prompt_ids, ex.rejected_ids).to(
            chosen_logp.device
        )
        dpo_loss = -F.logsigmoid(beta * ((chosen_logp - rejected_logp) - (chosen_ref_logp - rejected_ref_logp)))
        loss_sum += dpo_loss.item()

    policy_model.train()
    return {
        "val_accuracy": (correct / total) if total else 0.0,
        "val_dpo_loss": (loss_sum / total) if total else math.nan,
        "val_size": float(total),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "inf"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_checkpoint(model, tokenizer, output_dir: Path, tag: str) -> Path:
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir


def maybe_cap_examples(
    examples: list[TokenizedPreferenceExample],
    max_examples: int,
    seed: int,
) -> list[TokenizedPreferenceExample]:
    """
    限制一下训练和验证的样本数量，也可用来快速测试。
    """
    #将样本控制在max_examples的范围内。
    if max_examples <= 0 or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    sampled = examples[:]
    rng.shuffle(sampled)
    return sampled[:max_examples]


def finalize_update_step(
    *, # *强制调用时用关键字参数，增加代码可读性，避免传参错误。
    cfg: TrainConfig,
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    output_dir: Path,
    metrics_path: Path,
    update_step: int,
    epoch: int,
    train_loss_accumulator: float,
    accum_microbatches: int,
    grad_rescale_factor: float,
    train_start_time: float,
    total_update_steps: int,
    effective_batch_size: int,
    best_val_accuracy: float,
    val_examples: list[TokenizedPreferenceExample],
) -> tuple[float, float]:
    # 梯度缩放,如果 grad_rescale_factor != 1.0 ，对 policy model 所有参数的梯度乘以该因子。通常用于多设备训练时，将梯度除以设备数量（如 factor = 1/num_gpus），避免梯度在跨设备累加后偏大。
    if grad_rescale_factor != 1.0:
        for p in policy_model.parameters():
            if p.grad is not None:
                p.grad.mul_(grad_rescale_factor)
    # 梯度裁剪
    if cfg.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.clip_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True) #set_to_none=True 是将梯度直接设为 None，而不是填充为全零张量。None 不占用显存，而零张量仍然分配内存

    avg_train_loss = train_loss_accumulator / max(accum_microbatches, 1) #总损失除以梯度累计的批次，max()用来避免除以0
    
    # 时间和吞吐量等指标
    elapsed_seconds = time.perf_counter() - train_start_time
    updates_per_sec = update_step / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_updates = max(total_update_steps - update_step, 0)
    eta_seconds = (remaining_updates / updates_per_sec) if updates_per_sec > 0 else math.inf
    
    lr = float(optimizer.param_groups[0]["lr"])

    train_record = {
        "type": "train",
        "epoch": epoch,
        "update_step": update_step,
        "total_update_steps": total_update_steps,
        "train_loss": avg_train_loss,
        "lr": lr,
        "effective_batch_size": effective_batch_size,
        "samples_seen_estimate": update_step * effective_batch_size,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "updates_per_sec": updates_per_sec,
    }
    append_jsonl(metrics_path, train_record)

    # 每 log_every 步更新一下日志
    if update_step % cfg.log_every == 0:
        print(
            f"[train] epoch={epoch} update_step={update_step}/{total_update_steps} "
            f"loss={avg_train_loss:.4f} lr={lr:.2e} "
            f"elapsed={format_seconds(elapsed_seconds)} eta={format_seconds(eta_seconds)}"
        )
    #每 save_every 步存一下新模型
    if cfg.save_every > 0 and update_step % cfg.save_every == 0:
        save_checkpoint(policy_model, tokenizer, output_dir, f"checkpoint_step_{update_step}")

    #每 eval_every 步用 val 验证集验证一下最新模型。
    if update_step % cfg.eval_every == 0:
        val_metrics = evaluate(policy_model, ref_model, val_examples, cfg.beta)
        val_record = {
            "type": "val",
            "epoch": epoch,
            "update_step": update_step,
            "total_update_steps": total_update_steps,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            **val_metrics,
        }
        append_jsonl(metrics_path, val_record)
        print(
            f"[val] epoch={epoch} update_step={update_step}/{total_update_steps} "
            f"val_accuracy={val_metrics['val_accuracy']:.4f} "
            f"val_dpo_loss={val_metrics['val_dpo_loss']:.4f} "
            f"elapsed={format_seconds(elapsed_seconds)} eta={format_seconds(eta_seconds)}"
        )

        if val_metrics["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["val_accuracy"]
            ckpt_dir = save_checkpoint(policy_model, tokenizer, output_dir, "best_checkpoint")
            print(f"[save] new best checkpoint at {ckpt_dir}")

    return best_val_accuracy, avg_train_loss


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "train_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[setup] metrics file: {metrics_path}")

    print("[setup] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    # 一般还是手动配一下特殊 token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[setup] loading HH preferences...")
    raw_examples = load_hh_preferences(Path(cfg.hh_data_dir), cfg.train_files)
    tokenized_examples, skipped_too_long = tokenize_examples(raw_examples, tokenizer, cfg.max_seq_len)
    print(f"[data] tokenized={len(tokenized_examples)} skipped_too_long={skipped_too_long}")

    # 把处理好的样本划分数据集验证集
    train_examples, val_examples = split_train_val(tokenized_examples, cfg.val_size, cfg.seed)
    train_examples = maybe_cap_examples(train_examples, cfg.max_train_examples, cfg.seed)
    val_examples = maybe_cap_examples(val_examples, cfg.max_val_examples, cfg.seed + 1)
    print(f"[data] train={len(train_examples)} val={len(val_examples)}")

    model_dtype = resolve_dtype(cfg.dtype)
    model_kwargs = {"trust_remote_code": True}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype

    print("[setup] loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)
    print("[setup] loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)

    policy_model.to(cfg.train_device)
    ref_model.to(cfg.ref_device)

    # 利用梯度检查点来实现推理加速，开启后就不保存前向传播中间激活值，反向传播再重计算(类似Flash Attention思路)
    if cfg.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        if hasattr(policy_model, "config"):
            policy_model.config.use_cache = False

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    policy_model.train()
    optimizer = torch.optim.RMSprop(
        policy_model.parameters(),
        lr=cfg.lr,
        alpha=cfg.rmsprop_alpha,
        eps=cfg.rmsprop_eps,
        weight_decay=cfg.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)#训练循环开始前调用一次 zero_grad，确保环境干净。

    effective_batch = cfg.micro_batch_size * cfg.gradient_accumulation_steps
    total_microbatches = math.ceil(len(train_examples) / cfg.micro_batch_size) #ceil表示向上取整(ceiling天花板)，floor表示向下取整
    est_updates_per_epoch = math.ceil(total_microbatches / cfg.gradient_accumulation_steps)
    total_update_steps = est_updates_per_epoch * cfg.num_epochs
    print(
        f"[setup] effective_batch={effective_batch} "
        f"micro_batch_size={cfg.micro_batch_size} "
        f"grad_accum={cfg.gradient_accumulation_steps} "
        f"est_updates_per_epoch={est_updates_per_epoch} "
        f"total_update_steps={total_update_steps}"
    )

    best_val_accuracy = float("-inf")
    train_start = time.perf_counter()
    update_step = 0
    last_train_loss = math.nan

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"[epoch] start {epoch}/{cfg.num_epochs}")
        epoch_examples = train_examples[:]
        random.Random(cfg.seed + epoch).shuffle(epoch_examples)

        train_loss_accumulator = 0.0
        accum_microbatches = 0

        for i in range(0, len(epoch_examples), cfg.micro_batch_size):
            batch = epoch_examples[i : i + cfg.micro_batch_size]
            if not batch:
                continue

            batch_loss = torch.zeros((), device=cfg.train_device)
            for ex in batch: #逐样本计算 DPO loss，累加起来
                batch_loss = batch_loss + compute_per_instance_dpo_loss(policy_model, ref_model, ex, cfg.beta)
            batch_loss = batch_loss / len(batch)

            (batch_loss / cfg.gradient_accumulation_steps).backward() #梯度累计要除回来
            train_loss_accumulator += batch_loss.item()
            accum_microbatches += 1

            #跑完梯度累计的step后统一更新
            if accum_microbatches == cfg.gradient_accumulation_steps:
                update_step += 1
                best_val_accuracy, last_train_loss = finalize_update_step(
                    cfg=cfg,
                    policy_model=policy_model,
                    ref_model=ref_model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    metrics_path=metrics_path,
                    update_step=update_step,
                    epoch=epoch,
                    train_loss_accumulator=train_loss_accumulator,
                    accum_microbatches=accum_microbatches,
                    grad_rescale_factor=1.0,
                    train_start_time=train_start,
                    total_update_steps=total_update_steps,
                    effective_batch_size=effective_batch,
                    best_val_accuracy=best_val_accuracy,
                    val_examples=val_examples,
                )
                train_loss_accumulator = 0.0
                accum_microbatches = 0

        # 更改梯度累计后的梯度为正常梯度，也就是除一下.
        if accum_microbatches > 0:
            update_step += 1
            grad_rescale_factor = cfg.gradient_accumulation_steps / accum_microbatches
            best_val_accuracy, last_train_loss = finalize_update_step(
                cfg=cfg,
                policy_model=policy_model,
                ref_model=ref_model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                output_dir=output_dir,
                metrics_path=metrics_path,
                update_step=update_step,
                epoch=epoch,
                train_loss_accumulator=train_loss_accumulator,
                accum_microbatches=accum_microbatches,
                grad_rescale_factor=grad_rescale_factor,
                train_start_time=train_start,
                total_update_steps=total_update_steps,
                effective_batch_size=effective_batch,
                best_val_accuracy=best_val_accuracy,
                val_examples=val_examples,
            )
        print(f"[epoch] done {epoch}/{cfg.num_epochs}")

    elapsed = time.perf_counter() - train_start
    summary = {
        "total_updates": update_step,
        "total_update_steps_target": total_update_steps,
        "best_val_accuracy": best_val_accuracy,
        "last_train_loss": last_train_loss,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[summary]")
    print(json.dumps(summary, indent=2))

    final_dir = save_checkpoint(policy_model, tokenizer, output_dir, "final_checkpoint")
    print(f"[save] final checkpoint at {final_dir}")


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
