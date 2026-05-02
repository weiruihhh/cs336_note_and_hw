import torch
import torch.distributed as dist
import time
import os
import argparse
from torch.multiprocessing.spawn import spawn

def setup(master_addr, master_port, rank, world_size, backend):
    """ 初始化分布式环境 """
    os.environ['MASTER_ADDR'] = master_addr # 设置主节点IP地址以及端口，其他节点需要通过这个地址连接到主节点
    os.environ['MASTER_PORT'] = str(master_port)
    # 根据后端初始化进程组
    dist.init_process_group(backend, rank=rank, world_size=world_size) # 初始化进程组，rank是当前进程的rank，world_size是总进程数，backend: 这是指定通信后端的参数。常见的后端有：gloo(CPU), nccl(GPU), mpi等。

def cleanup():
    """ 清理分布式环境 """
    dist.destroy_process_group()
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_all_reduce(rank, world_size, tensor_size_mb, backend, device, master_addr, master_port):
    """ 评测函数主体 """
    # 1. 设置环境和设备
    setup(master_addr, master_port, rank, world_size, backend)
    if device == 'cuda':
        # 将当前进程绑定到对应的GPU
        torch.cuda.set_device(rank)
        # 清理GPU缓存以避免内存冲突
        torch.cuda.empty_cache()

    # 2. 创建测试数据
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    # float32 是 4 字节
    num_elements = tensor_size_bytes // 4
    tensor_data = torch.randn(num_elements, device=device)

    # 3. 预热 (Warm-up)，非常重要！
    # 按照作业要求，预热5次
    for _ in range(5):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)
        # 如果是GPU，需要同步等待操作完成
        if device == 'cuda':
            torch.cuda.synchronize()
    
    dist.barrier()
    # 4. 正式计时
    start_time = time.time()
    num_iterations = 20 # 可以多跑几次取平均值
    for _ in range(num_iterations):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)
    
    # GPU需要同步以确保所有操作完成
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    duration = end_time - start_time
    avg_time = duration / num_iterations
    
    
    # 计算带宽（所有 rank 都需要，以便返回结构一致）
    # 带宽 = 数据大小 / 时间（简化计算）
    bandwidth_gbps = (tensor_size_bytes / avg_time) / 1e9

    # # 只在主进程打印结果
    if rank == 0:
        print(f"Backend: {backend}, Device: {device}, World Size: {world_size}, Tensor Size: {tensor_size_mb}MB")
        print(f"Average time per all-reduce: {avg_time * 1000:.4f} ms")
        print(f"Achieved Bandwidth: {bandwidth_gbps:.4f} GB/s\n")

    local_result = {
            'rank': rank,
            'world_size': world_size,
            'backend': backend,
            'device': device,
            'tensor_size_mb': tensor_size_mb,
            'avg_time_ms': avg_time * 1000.0,
            'bandwidth_gbps': bandwidth_gbps
        }
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    # 最后清理环境
    cleanup()
    if rank == 0:
        return gathered_results 


