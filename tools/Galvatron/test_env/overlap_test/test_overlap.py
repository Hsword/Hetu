import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
import numpy as np
import random
import time
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from utils import read_json_config, write_json_config

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def profile(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    model = nn.Linear(4096, 4096, bias=False).cuda()
    compute_tensor = torch.randn((1024,4096), device=device)
    comm_tensor = torch.randn((4096,4096), device=device)
    compute_iters = 4096
    allreduce_warmup_iters = 10
    allreduce_iters = 30

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    torch.cuda.Stream.synchronize(compute_stream)
    allreduce_time_list = []

    def trace_handler(prof):
        if local_rank > -1:
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
            table = table.split('\n')
            def split_line(line):
                line = line.split('  ')
                ls = []
                for s in line:
                    if len(s):
                        ls.append(s.strip())
                return ls
            def str2time(s):
                if 'ms' in s:
                    return float(s[:-2])
                elif 'us' in s:
                    return float(s[:-2])*1e-3
                else:
                    return float(s[:-1])*1e3
            for line in table:
                if 'Name' in line:
                    title = split_line(line)
                if 'ncclKernel_AllReduce' in line:
                    result = split_line(line)
            for i in range(len(title)):
                if 'CUDA total' in title[i]:
                    cuda_total_idx = i
                if '# of Calls' in title[i]:
                    call_times_idx = i
            allreduce_time = str2time(result[cuda_total_idx])/int(result[call_times_idx])
            allreduce_time = torch.tensor([allreduce_time]).to(device)
            torch.distributed.reduce(allreduce_time, 0, op=torch.distributed.ReduceOp.SUM)
            allreduce_time = allreduce_time.cpu().numpy()[0] / world_size
            if local_rank == 0:
                print('Average allreduce time (ms):', allreduce_time)
                allreduce_time_list.append(allreduce_time)

    if rank == 0:
        print('Profiling allreduce time when not overlapping with computation...')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                            schedule=torch.profiler.schedule(wait=0,warmup=1,active=1),
                            on_trace_ready=trace_handler) as p:
        # Warming up
        if rank == 0:
            print('Warming up...')
        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_warmup_iters):
                torch.distributed.all_reduce(comm_tensor)
        torch.cuda.Stream.synchronize(comm_stream)
        p.step()

        # Profiling allreduce time when not overlapping with computation
        if rank == 0:
            print('Profiling...')
        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_iters):
                torch.distributed.all_reduce(comm_tensor)
        torch.cuda.Stream.synchronize(comm_stream)
        p.step()

    if rank == 0:
        print('\nProfiling allreduce time when overlapping with computation...')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                            schedule=torch.profiler.schedule(wait=0,warmup=1,active=1),
                            on_trace_ready=trace_handler) as p:
        # Warming up
        if rank == 0:
            print('Warming up...')
        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_warmup_iters):
                torch.distributed.all_reduce(comm_tensor)
        torch.cuda.Stream.synchronize(comm_stream)
        p.step()

        # Profile allreduce time when overlapping with computation
        if rank == 0:
            print('Profiling...')
        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_iters):
                torch.distributed.all_reduce(comm_tensor)
        with torch.cuda.stream(compute_stream):
            for i in range(compute_iters):
                output = model(compute_tensor)
        torch.cuda.Stream.synchronize(comm_stream)
        p.step()

    if local_rank == 0:
        env_config_path = '../../env_configs/overlap_coefficient.json'
        config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
        key = 'overlap_coe'
        config[key] = allreduce_time_list[1] / allreduce_time_list[0]
        print('\n********************')
        print('Overlap coefficient:', config[key])
        write_json_config(config, env_config_path)
        print('Already written overlap_coefficient into env config file %s!'%(env_config_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank" ,type=int,default=-1)
    args = parser.parse_args()
    set_seed()
    profile(args)