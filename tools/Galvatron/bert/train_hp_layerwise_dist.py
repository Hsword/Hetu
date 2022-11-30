import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
from dataloader import DataLoaderForBert_wrapped
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
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
    prediction_scores, seq_relationship_score = outputs
    masked_lm_labels, next_sentence_label = labels
    loss_fct = nn.CrossEntropyLoss(ignore_index = -1)
    masked_lm_loss = loss_fct(prediction_scores.view(-1, 30522), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    loss = masked_lm_loss + next_sentence_loss
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
    dataset = DataLoaderForBert_wrapped()
    data_num_replicas = world_size // hybrid_parallel_configs['pp_deg']
    train_batch_size_input = args.global_train_batch_size // data_num_replicas
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=True,num_replicas=data_num_replicas,rank=rank%data_num_replicas))

    if local_rank == 0:
        print("Creating Model...")
    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob)
    overwrite_megatron_args(config, args)
    bert_model = BertForPreTraining(config)
    model = construct_hybrid_parallel_model(model=bert_model, 
                                            model_config=config, 
                                            training_args=args, 
                                            hybrid_parallel_configs=hybrid_parallel_configs)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profile_rank = 0 # profile first stage memory
    if args.profile and local_rank == profile_rank:
        print_peak_memory("After creating model", local_rank, args.profile_type)

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
            batch[0] = [tensor.to(device) for tensor in batch[0]]
            batch[1] = [tensor.to(device) for tensor in batch[1]]

            if args.profile and local_rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(local_rank)
                print_peak_memory("\nBefore Forward", local_rank, args.profile_type)

            loss = model.gpipe_forward(forward_step_func, batch)

            if args.profile and local_rank == profile_rank and iter <= 2:
                print_peak_memory("After Forward", local_rank, args.profile_type)

            model.gpipe_backward()

            if args.profile and local_rank == profile_rank and iter <= 2:
                print_peak_memory("After Backward", local_rank, args.profile_type)

            optimizer.step()

            if args.profile and local_rank == profile_rank and iter <= 2:
                print_peak_memory("After optimizer_step", local_rank, args.profile_type)

            optimizer.zero_grad()

            end_time = time.time()
            if args.check_loss or args.profile:
                if len(loss):
                    loss = np.mean([l.item() for l in loss])
                    print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss.item()))


def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Global training batch size"
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
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
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("--max_predictions_per_seq", type=int, default=20)
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
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
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32],
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
    set_seed()
    train(args)
