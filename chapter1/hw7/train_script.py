import subprocess
import os

# 定义要执行的命令列表，每个命令都是一个包含其参数的列表
# 这样做比直接使用字符串更安全、更推荐
experiments = [
    # {
    #     "name": "Default Parameters",
    #     "command": ["python", "final_train.py"]
    # },
    # {
    #     "name": "Batch Size 128",
    #     "command": ["python", "final_train.py", "--batch_size", "128"]
    # },
    {
        "name": "Without RMSNorm",
        "command": ["python", "final_train.py", "--no-rmsnorm"]
    }
]

# 依次执行每个实验
for i, exp in enumerate(experiments):
    print("="*60)
    print(f"Running Experiment {i+1}: {exp['name']}")
    print(f"Command: {' '.join(exp['command'])}")
    print("="*60)
    
    try:
        # 使用 subprocess.run 执行命令
        # check=True 会在命令返回非零退出码（即发生错误）时抛出异常
        subprocess.run(exp["command"], check=True)
        print(f"\n--- Experiment '{exp['name']}' finished successfully. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- Experiment '{exp['name']}' failed with exit code {e.returncode}. ---\n")
        break # 如果一个实验失败，可以选择停止后续实验
    except FileNotFoundError:
        print("\n--- Error: 'python' command not found. Make sure Python is in your PATH. ---\n")
        break

print("All experiments have been run.")
