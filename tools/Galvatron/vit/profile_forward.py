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
from utils import print_peak_memory
from utils import read_json_config, write_json_config


def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def run_forward(args, conf):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    rank = args.gpu_id

    print("Creating Dataloader...")
    dataset, num_classes = build_dataset(is_train=True, config=conf)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            shuffle=False)

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
    model = ViTForImageClassification(config)
    model.to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_iter, end_iter = 10, 20

    print("Profiling...")
    time_list = []
    for ep in range(args.epochs):
        for iter, batch in enumerate(trainloader):
            if iter == end_iter:
                avg_time = sum(time_list) / len(time_list) / args.train_batch_size
                print("Average forward computation time of %d ViT layers (hidden_size=%d) is: %.4f ms / bsz"%(args.num_hidden_layers, args.hidden_size, avg_time))
                return avg_time
            input, label= [tensor.to(device) for tensor in batch]
            torch.cuda.synchronize()
            start.record()
            loss = model(pixel_values = input, labels = label).loss
            end.record()
            torch.cuda.synchronize()
            iter_time = start.elapsed_time(end)
            if iter >= start_iter:
                time_list.append(iter_time)

def profile(args, conf):
    args.num_hidden_layers = 24
    time_24_layers = run_forward(args, conf)
    args.num_hidden_layers = 12
    time_12_layers = run_forward(args, conf)
    time_per_layer = (time_24_layers-time_12_layers)/12

    fwd_config_path = './configs/forward_profiling_config.json'
    config = read_json_config(fwd_config_path) if os.path.exists(fwd_config_path) else dict()
    key = 'fwd_time_hidden_%d'%(args.hidden_size)
    config[key] = time_per_layer
    print('********************')
    print("Average forward computation time of ViT layer (hidden_size=%d) is: %.4f ms / layer / bsz"%(args.hidden_size, time_per_layer))
    write_json_config(config, fwd_config_path)
    print('Already written forward profiling config into env config file %s!\n'%(fwd_config_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Input image size."
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch size of ViT."
    )
    parser.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.0, help="Dropout rate."
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        '--data-folder', default = 'ImageNet', type=str, help='path to dataset'
    )
    parser.add_argument(
        '--zip', type=bool, default=True, help='use zipped dataset instead of folder dataset'
    )
    parser.add_argument(
        '--cache-mode', type=str, default='no', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )
    parser.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel'
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    set_seed()

    profile(args, config)