import wandb
import time
import torch
import json
import pickle
import time
import argparse
from tokenizer_encode import Tokenizer
from pair_all_bpe_tokenzier import run_train_bpe

from adamw import AdamW
from dataloader import DataLoader
from transformermodule import TransformerModule
from transformermodule_withoutrmsnorm import TransformerModuleWithoutRMSNorm
from cross_entropy import CrossEntropyLoss
from lr_cosine_shedule import CosineSchedule

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:6")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--train_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--no-rmsnorm', dest='use_rmsnorm', action='store_false', help="Disable RMSNorm and use LayerNorm instead")
parser.set_defaults(use_rmsnorm=True)
args = parser.parse_args()

device = args.device
epochs = args.epochs
train_steps = args.train_steps
batch_size = args.batch_size

timestamp = time.strftime("%Y%m%d_%H%M%S")
wandb.login()
run = wandb.init(project="cs336_final_train", 
                 config = {
                # Experiment
                "experiment_name": f"tinystories_17M_{timestamp}",
                "total_tokens_processed": 327_680_000,
                
                # Data
                "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
                "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
                "vocab_path": "vocab.json",
                "merges_path": "merges.txt",

                # Model
                "vocab_size": 10000,
                "context_length": 256,
                "d_model": 512,
                "d_ff": 1344,
                "n_layers": 4,
                "n_heads": 16,
                "rope_theta": 10000.0,

                # Training
                "batch_size": batch_size, # Adjust based on your GPU memory
                # "learning_rate": 3e-5,
                #学习率退火相关参数
                "initial_lr": 3e-5,
                "max_learning_rate": 3e-5,
                "min_learning_rate": 1e-5,
                "lr_warmup_steps": 2000,
                "cosine_cycle_iters": 10000,

                #优化器相关参数
                "weight_decay": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "eps": 1e-8,

                #梯度裁剪
                "grad_clip": 1.0,

                #训练相关参数
                "epochs": epochs,
                "train_steps": train_steps,
                
                # Logging & Checkpointing
                "log_interval": 20,
                "val_interval": 20,
                "checkpoint_interval": 60,
                "checkpoint_dir": "checkpoints",
            }
            )
config = run.config




device = torch.device(device if torch.cuda.is_available() else "cpu")
data_path = config["train_data_path"]
vocab_size = config["vocab_size"]
# # 训练BPE分词器
# special_tokens = ["<|endoftext|>"]
# vocab, merges = run_train_bpe(data_path, vocab_size, special_tokens)
# print("已经训练好BPE分词器")

# 从 vocab.pkl 加载词汇表
# with open("vocab.pkl", "rb") as f:
#     # pickle.load 会自动恢复字典，并且值是 bytes 类型
#     vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
# with open("merges.pkl", "rb") as f:
#     # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
#     merges = pickle.load(f)
# special_tokens = ["<|endoftext|>"]  

# tokenizer = Tokenizer(vocab, merges, special_tokens)
# 加载训练数据
# with open(data_path, "r",encoding="utf-8") as f:
#     original_data = f.read()
# encode_ids = tokenizer.encode(original_data)
# encode_ids = torch.tensor(encode_ids, dtype=torch.long)
# print("数据加载完成")
#直接导入编码后的数据
with open("encoded_ids_train.pkl", "rb") as f:
    train_encode_ids = pickle.load(f)
with open("encoded_ids_valid.pkl", "rb") as f:
    valid_encode_ids = pickle.load(f)

train_data_loader = DataLoader(train_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 训练集导入
valid_data_loader = DataLoader(valid_encode_ids, config["batch_size"],config["context_length"],shuffle=True) # 验证集导入
# 加载模型
if args.use_rmsnorm:
    model = TransformerModuleWithoutRMSNorm(config["d_model"], config["n_heads"], config["d_ff"], config["context_length"], config["rope_theta"], config["n_layers"], vocab_size, device).to(device)
else:
    model = TransformerModule(config["d_model"], config["n_heads"], config["d_ff"], config["context_length"], config["rope_theta"], config["n_layers"], vocab_size, device).to(device)
# 这一行是错误的，因为 args 对象里没有 no_rmsnorm 这个属性
# 加载优化器
lr_scheduler = CosineSchedule(config["max_learning_rate"], config["min_learning_rate"], config["lr_warmup_steps"], config["cosine_cycle_iters"]) # 学习率退火   
optimizer = AdamW(model.parameters(), config["initial_lr"], (config["adam_beta1"], config["adam_beta2"]), config["eps"], config["weight_decay"])
# 加载损失函数
loss_fn = CrossEntropyLoss()
print("模型加载完成")
# 4. 训练循环
model.train()
global_step = 0  # 初始化全局步数
for epoch in range(config["epochs"]):
    # for step in range(len(train_data_loader)):
    # 学习率更新移到 step 循环内部
    # print("epoch",epoch)
    for step in range(args.train_steps):
        # 更新学习率
        new_lr = lr_scheduler(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # print("epoch",epoch,"lr",new_lr)
        x,y = train_data_loader.get_train_batch_data()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)  # shape: (batch, seq, vocab_size)
        loss = loss_fn.forward(logits, y)
        # loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        global_step += 1 # 增加全局步数

        if step % 100 == 0:
            print(f"Epoch {epoch} Step {step} LR {new_lr:.6f} Loss: {loss.item()}")
    # print("训练完了")
    wandb.log({"epoch": epoch, "loss": loss.item()})
    # print("经过了wandblog")
    if (epoch+1) % config["val_interval"] == 0:
        model.eval()
        with torch.no_grad():
            for x, y in valid_data_loader.get_valid_batch_data_iter():
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                #   loss = loss_fn.forward(logits, y)
                # loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss = loss_fn.forward(logits, y)
                wandb.log({"epoch": epoch, "loss": loss.item()})
    # print("经过验证集")
    if (epoch+1) % config["checkpoint_interval"] == 0:
        # torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 可以加上'loss': loss.item()等
        }, f"checkpoints/model_epoch_{epoch}_{timestamp}.pth")
        print(f"Checkpoint saved at epoch {epoch}")

torch.save(model.state_dict(), f"checkpoints/model_final_{timestamp}.pth")
print("Final checkpoint saved")

