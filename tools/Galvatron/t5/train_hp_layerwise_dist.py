import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor
from dataloader import DataLoaderForT5
import argparse
from tqdm import tqdm
import numpy as np
import random
import time
import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from utils import print_peak_memory
from megatron.initialize import initialize_megatron
from megatron import get_args, _print_args
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, List
from hybrid_parallel_model_dist import get_hybrid_parallel_configs, construct_hybrid_parallel_model, overwrite_megatron_args

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def loss_func(labels, outputs):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    logits = outputs[0]
    labels = labels[0]
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def forward_step_func(inputs, model):
    if isinstance(inputs, (Tuple, List)):
        outputs = model(*inputs)
    else:
        outputs = model(inputs)
    return outputs, loss_func

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    hybrid_parallel_configs = get_hybrid_parallel_configs(args)

    if local_rank == 0:
        print("Creating Dataloader...")
    dataset = DataLoaderForT5(args)
    data_num_replicas = world_size // hybrid_parallel_configs['pp_deg']
    train_batch_size_input = args.global_train_batch_size // data_num_replicas
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=True,num_replicas=data_num_replicas,rank=rank%data_num_replicas))

    if local_rank == 0:
        print("Creating Model...")
    config = T5Config.from_pretrained(args.model_config, dropout_rate=args.dropout_prob)
    config.num_layers = args.num_encoder_layer if args.num_encoder_layer > 0 else config.num_layers
    config.num_decoder_layers = args.num_decoder_layer if args.num_decoder_layer > 0 else config.num_decoder_layers
    config.num_heads = args.num_head if args.num_head > 0 else config.num_heads
    overwrite_megatron_args(config, args)
    t5_model = T5ForConditionalGeneration(config)
    model = construct_hybrid_parallel_model(model=t5_model, 
                                            model_config=config, 
                                            training_args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_iter, end_iter = 10, 20
    profile_rank = 0
    if args.profile and rank == profile_rank:
        print_peak_memory("After creating model", rank, args.profile_type)

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
            input_ids, attention_mask, labels= [tensor.to(device) for tensor in batch]
            batch = [[input_ids, labels, attention_mask], [labels]]

            if args.profile and rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            loss = model.gpipe_forward(forward_step_func, batch)

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Forward", rank, args.profile_type)

            model.gpipe_backward()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Backward", rank, args.profile_type)

            optimizer.step()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After optimizer_step", rank, args.profile_type)

            optimizer.zero_grad()
            end_time = time.time()
            if args.check_loss or args.profile:
                if len(loss):
                    loss = np.mean([l.item() for l in loss])
                    print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss.item()))

def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Training batch size"
    )
    group.add_argument(
        "--model_config", type=str, default='t5-base', help="T5 model name", choices=['t5-base', 't5-large']
    )
    group.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
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
    group.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32],
    )
    parser.add_argument(
        "--num_encoder_layer", type=int, default=0, help="overwrite encoder layer num"
    )
    parser.add_argument(
        "--num_decoder_layer", type=int, default=0, help="overwrite decoder layer num"
    )
    parser.add_argument(
        "--num_head", type=int, default=0, help="overwrite attention head num"
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
    args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'}
    initialize_megatron(extra_args_provider=add_arguments, args_defaults=args_defaults)
    args = get_args()
    set_seed()
    train(args)