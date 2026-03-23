import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from DataLoader import Dataset


@dataclass
class TrainConfig:
    model_name_or_path: str #指定名称就是从huggingface上下载，指定本地路径就是从本地加载。
    train_path: str
    val_path: str
    output_dir: str
    seq_length: int = 512 #每个样本的最大长度，超过这个长度就切分成多个样本。
    per_device_batch_size: int = 2 #
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 500
    metrics_file: str = "metrics.jsonl"
    use_flash_attn: bool = True
    save_best: bool = True
    save_final: bool = True



def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Instruction fine-tuning for causal LM")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--metrics_file", type=str, default="metrics.jsonl")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--no_save_best", action="store_true")
    parser.add_argument("--no_save_final", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        metrics_file=args.metrics_file,
        use_flash_attn=not args.no_flash_attn,
        save_best=not args.no_save_best,
        save_final=not args.no_save_final,
    )



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def build_model_and_tokenizer(cfg: TrainConfig):
    """
    加载并配置模型参数
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model_kwargs = {"dtype": torch.bfloat16}
    if cfg.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    return model, tokenizer



def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    """
    这里改用惰性yield的方式
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [dataset[j] for j in batch_indices]
        yield {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }


def build_datasets(cfg: TrainConfig, tokenizer):
    train_dataset = Dataset(
        tokenizer=tokenizer,
        dataset_path=cfg.train_path,
        seq_length=cfg.seq_length,
        shuffle=True,
    )
    val_dataset = Dataset(
        tokenizer=tokenizer,
        dataset_path=cfg.val_path,
        seq_length=cfg.seq_length,
        shuffle=False,
    )
    return train_dataset, val_dataset



def build_optimizer_and_scheduler(cfg: TrainConfig, model, total_update_steps: int):
    """
    优化器
    """
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_steps = int(total_update_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    return optimizer, scheduler



def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算损失
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))



def evaluate(model, val_dataset: Dataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in iterate_batches(val_dataset, batch_size=batch_size, shuffle=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids).logits
            loss = compute_lm_loss(logits, labels)
            losses.append(loss.item())
    model.train()

    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))



def save_checkpoint(model, tokenizer, save_dir: str) -> None:
    """"
    保存权重
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_seconds(seconds: float) -> str:
    """
    转换时间格式
    """
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"



def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.output_dir, cfg.metrics_file)

    model, tokenizer = build_model_and_tokenizer(cfg)
    model.to(device)
    model.train()

    train_dataset, val_dataset = build_datasets(cfg, tokenizer)

    train_steps_per_epoch = math.ceil(len(train_dataset) / cfg.per_device_batch_size)
    update_steps_per_epoch = math.ceil(train_steps_per_epoch / cfg.gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * cfg.num_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, total_update_steps)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    update_step = 0
    running_loss = 0.0
    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train batches/epoch: {train_steps_per_epoch}")
    print(f"Val batches: {math.ceil(len(val_dataset) / cfg.per_device_batch_size)}")
    print(f"Total update steps: {total_update_steps}")
    print(f"Effective batch size: {cfg.per_device_batch_size * cfg.gradient_accumulation_steps}")
    print(f"Metrics file: {metrics_path}")


    train_start_time = time.time()
    for epoch in range(cfg.num_epochs):
        print(f"Starting epoch {epoch + 1}/{cfg.num_epochs}")
        for batch_idx, batch in enumerate(
            iterate_batches(train_dataset, batch_size=cfg.per_device_batch_size, shuffle=True)
        ):
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids).logits
            loss = compute_lm_loss(logits, labels)

            running_loss += loss.item()
            scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()

            #只在到达了梯度累计的步数之后才会更新
            should_update = (global_step % cfg.gradient_accumulation_steps == 0) or (batch_idx == train_steps_per_epoch - 1)
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % cfg.log_interval == 0:
                    avg_train_loss = running_loss / cfg.log_interval
                    running_loss = 0.0
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed_seconds = time.time() - train_start_time
                    progress = (update_step / max(1, total_update_steps)) * 100
                    updates_per_sec = update_step / max(1e-6, elapsed_seconds)
                    remaining_updates = max(0, total_update_steps - update_step)
                    eta_seconds = remaining_updates / max(1e-6, updates_per_sec)
                    print(
                        f"epoch={epoch + 1} update_step={update_step}/{total_update_steps} "
                        f"train_loss={avg_train_loss:.4f} lr={current_lr:.6e} "
                        f"progress={progress:.2f}% elapsed={format_seconds(elapsed_seconds)} "
                        f"eta={format_seconds(eta_seconds)}"
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "train",
                            "epoch": epoch + 1,
                            "update_step": update_step,
                            "global_step": global_step,
                            "train_loss": avg_train_loss,
                            "lr": current_lr,
                        },
                    )

                if update_step % cfg.eval_interval == 0:
                    val_loss = evaluate(model, val_dataset, cfg.per_device_batch_size, device)
                    print(f"epoch={epoch + 1} update_step={update_step} val_loss={val_loss:.4f}")
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "val",
                            "epoch": epoch + 1,
                            "update_step": update_step,
                            "global_step": global_step,
                            "val_loss": val_loss,
                        },
                    )

                    if cfg.save_best and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, tokenizer, os.path.join(cfg.output_dir, "best"))
                        print(f"Saved new best checkpoint to {os.path.join(cfg.output_dir, 'best')}")


    final_val_loss = evaluate(model, val_dataset, cfg.per_device_batch_size, device)
    print(f"Final val_loss={final_val_loss:.4f}")
    append_jsonl(
        metrics_path,
        {
            "type": "val_final",
            "epoch": cfg.num_epochs,
            "update_step": update_step,
            "global_step": global_step,
            "val_loss": final_val_loss,
        },
    )

    if cfg.save_final:
        save_checkpoint(model, tokenizer, os.path.join(cfg.output_dir, "final"))
        print(f"Saved final checkpoint to {os.path.join(cfg.output_dir, 'final')}")



def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
