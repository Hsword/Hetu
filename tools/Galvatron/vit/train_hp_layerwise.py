import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import ViTConfig, ViTForImageClassification
import argparse
from tqdm import tqdm
import numpy as np
import random
import os
import time
from data import build_dataset
from config import get_config
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from utils import print_peak_memory
from megatron.initialize import initialize_megatron
from megatron import get_args, _print_args
from torch.utils.data.distributed import DistributedSampler
import torch.distributed.rpc as rpc
from hybrid_parallel_model import get_hybrid_parallel_configs, construct_hybrid_parallel_model, overwrite_megatron_args

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args, conf):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    rpc.init_rpc(name="worker%d" % rank, rank = rank, world_size=world_size)

    hybrid_parallel_configs = get_hybrid_parallel_configs(args)

    if local_rank == 0:
        print("Creating Dataloader...")
    dataset, num_classes = build_dataset(is_train=True, config=conf)
    train_batch_size_input = args.global_train_batch_size // world_size
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=True))

    if local_rank == 0:
        print("Creating Model...")
    config = ViTConfig(hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob,
                        image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_channels=args.num_channels)
    config.num_labels = num_classes
    overwrite_megatron_args(config, args)
    vit_model = ViTForImageClassification(config)
    model = construct_hybrid_parallel_model(model=vit_model, 
                                            model_config=config, 
                                            training_args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profile_rank = 0
    if args.profile and rank == profile_rank:
        print_peak_memory("After creating model", rank, args.profile_type)

    start_iter, end_iter = 10, 20
    if local_rank == 0:
        print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            start_time = time.time()
            if args.profile:
                if iter == start_iter:
                    total_start_time = start_time
                elif iter == end_iter:
                    total_end_time = start_time
                    avg_time = (total_end_time-total_start_time)/(end_iter-start_iter)
                    print("Average iteration time is: %.4f s"%avg_time)
                    return
            input, label= [tensor.to(device) for tensor in batch]
            if args.profile and rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            outputs = model(input).local_value()
            lossft = nn.CrossEntropyLoss()
            loss = lossft(outputs.view(-1, config.num_labels), label.view(-1))

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Forward", rank, args.profile_type)

            loss.backward()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Backward", rank, args.profile_type)

            optimizer.step()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After optimizer_step", rank, args.profile_type)
            
            optimizer.zero_grad()

            end_time = time.time()
            if args.check_loss or args.profile:
                print('[Epoch %d] (Iteration %d): Loss = %.6f'% (ep,iter,loss.item()))

def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=196, help="Maximum sequence len"
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        '--data-folder', default = 'ImageNet', type=str, help='path to dataset'
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    group.add_argument(
        "--image_size", type=int, default=224, help="Input image size."
    )
    group.add_argument(
        "--patch_size", type=int, default=16, help="Patch size of ViT."
    )
    group.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels."
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.0, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    group.add_argument(
        '--zip', type=bool, default=True, help='use zipped dataset instead of folder dataset'
    )
    group.add_argument(
        '--cache-mode', type=str, default='no', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )
    group.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    group.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    group.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8],
    )
    parser.add_argument(
        "--chunks", type=int, default=-1, help="Pipeline chunk num.",
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--fsdp", type=int, default=0, help="Apply FSDP", choices=[0, 1],
    )
    parser.add_argument(
        "--apply_strategy", type=int, default=0, help="Apply searched strategy.", choices=[0, 1],
    )
    parser.add_argument(
        "--galvatron_config_path", type=str, default=None, help="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )

    return parser

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()
    config = get_config(args)
    set_seed()

    train(args, config)