import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import T5ForConditionalGeneration, T5Config
from dataloader import DataLoaderForT5
import argparse
from tqdm import tqdm
import numpy as np
import random
import time
import os
import sys
sys.path.insert(0, '..')
from utils import print_peak_memory

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    rank = args.gpu_id

    print("Creating Dataloader...")
    dataset = DataLoaderForT5(args)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            shuffle=False)

    print("Creating Model...")
    config = T5Config.from_pretrained(args.model_config, dropout_rate=args.dropout_prob)
    if args.num_encoder_layer > 0:
        config.num_layers = args.num_encoder_layer
    if args.num_decoder_layer > 0:
        config.num_decoder_layers = args.num_decoder_layer
    model = T5ForConditionalGeneration(config)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.profile:
        print_peak_memory("After creating model", rank, args.profile_type)

    start_iter, end_iter = 10, 20
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

            if args.profile and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            if args.profile and iter <= 2:
                print_peak_memory("After Forward", rank, args.profile_type)
            
            loss.backward()

            if args.profile and iter <= 2:
                print_peak_memory("After Backward", rank, args.profile_type)
            
            optimizer.step()

            if args.profile and iter <= 2:
                print_peak_memory("After optimizer_step", rank, args.profile_type)
            
            optimizer.zero_grad()
            end_time = time.time()
            if args.check_loss or args.profile:
                print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--model_config", type=str, default='t5-base', help="T5 model name", choices=['t5-base', 't5-large']
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )

    parser.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--num_encoder_layer", type=int, default=0, help="overwrite encoder layer num"
    )
    parser.add_argument(
        "--num_decoder_layer", type=int, default=0, help="overwrite decoder layer num"
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )

    args = parser.parse_args()
    set_seed()
    train(args)