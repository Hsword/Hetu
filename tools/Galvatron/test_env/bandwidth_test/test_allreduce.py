import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import random
import h5py
from tqdm import tqdm
import time
import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from utils import print_peak_memory
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import gen_groups
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from utils import read_json_config, write_json_config


def init_method_constant(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)
    return init_

class pre_sync_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, hidden_states):
        return hidden_states

class pre_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return hidden_states

def _reduce(input_, group):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None

def reduce_from_tensor_model_parallel_region_group(input_, group):
    return _ReduceFromModelParallelRegion.apply(input_, group)

def copy_to_tensor_model_parallel_region_group(input_, group):
    return _CopyToModelParallelRegion.apply(input_, group)

class allreduce_block(nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        hidden_states = reduce_from_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        return hidden_states

class DataLoaderRandom(Dataset):
    def __init__(self, local_bsz):
        self.dataset_size = local_bsz*8*11//args.pp_deg
        # self.input = np.random.randint(0, 100, size=(self.dataset_size, 512, 1024))
        self.input = np.ones((self.dataset_size, 512, 1024))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input = torch.FloatTensor(self.input[idx])
        return input

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.num_layers = 24
    args.local_batch_size = 32
    train_batch_size_input = args.local_batch_size
    if rank == 0:
        print('local_bsz = %d'%train_batch_size_input)

    import torch.distributed.rpc as rpc
    rpc.init_rpc(
            name="worker%d" % rank,
            rank = rank,
            world_size=world_size,
    )

    dataset = DataLoaderRandom(train_batch_size_input)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=False))

    all_tp_sizes = [args.global_tp_deg] * 24
    tp_consecutive_flags = [args.global_tp_consec] * 24
    tp_groups, _, _, _ = gen_groups(all_tp_sizes, tp_consecutive_flags)
    
    model = nn.Sequential()
    model.add_module('pre_sync_module', pre_sync_module())
    model.add_module('pre_mlp', pre_mlp())
    for i in range(len(all_tp_sizes)):
        module = allreduce_block(tp_group=tp_groups[i])
        model.add_module('mlp_%d'%i, module)

    avg_num_layers = args.num_layers // args.pp_deg
    pp_ranks_enc = []
    for i in range(args.pp_deg):
        pp_ranks_enc += [i] * avg_num_layers
    
    devices = [i * world_size + rank for i in range(args.pp_deg)]
    pp_devices = [devices[i] for i in pp_ranks_enc]
    model[0] = DDP(model[0].cuda(devices[0])) # for sync
    model[1] = model[1].cuda(devices[0])
    for i in range(len(all_tp_sizes)):
        model[i+2] = model[i+2].cuda(pp_devices[i])
    model = Pipe(model, chunks=1, checkpoint='never')

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    # Calculate theoretical communication message size
    tp_size = args.global_tp_deg
    pp_size = args.pp_deg
    dp_size = world_size // tp_size
    bs = args.local_batch_size
    allreduce_message_size_per_layer = 2*(tp_size-1)/tp_size*(bs*512*1024*2*4/1024/1024)
    allreduce_message_size_total = allreduce_message_size_per_layer * 24
    if rank == 0:
        print('Strategy: %d_%d_%d'%(pp_size,tp_size,args.global_tp_consec))
        print('[allreduce_message_size]: per_layer %d MB, total %d MB'%(allreduce_message_size_per_layer,allreduce_message_size_total))

    def trace_handler(prof):
        if rank == 0:
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
            print(table)
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
                # print('%s: %s'%(title[i],result[i]))
                if 'CUDA total' in title[i]:
                    cuda_total_idx = i
            allreduce_time_24_layer = str2time(result[cuda_total_idx])/10
            comm_coe = allreduce_time_24_layer / allreduce_message_size_total
            print('**********')
            print('comm_coe_%d_%d_%d (ms/MB):'%(pp_size,tp_size,args.global_tp_consec), comm_coe)
            print('**********')

            comm_type = 'allreduce'
            gpu_num = world_size * pp_size
            env_config_path = '../../env_configs/%s_bandwidth_%d_gpus.json'%(comm_type, gpu_num)
            config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
            key = '%d_%d_%d'%(pp_size,tp_size,args.global_tp_consec)
            config[key] = comm_coe
            write_json_config(config, env_config_path)
            print('Already written comm_coe_%s into env config file %s!'%(key, env_config_path))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                schedule=torch.profiler.schedule(wait=0,warmup=1,active=10),
                                on_trace_ready=trace_handler) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            out = model(input)
            loss = out.local_value().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16],
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16],
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=32, help="local training batch size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=48, help="Number of layers"
    )
    parser.add_argument("--local_rank" ,type=int,default=-1)
    args = parser.parse_args()
    set_seed()
    train(args)
