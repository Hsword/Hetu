import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import SwinConfig, SwinForImageClassification
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
    config = SwinConfig(drop_path_rate=args.drop_path_rate,
                       embed_dim=args.embed_dim,
                       depths=args.depths,
                       num_heads=args.num_heads,
                       window_size=args.window_size)
    config.num_labels = num_classes
    model = SwinForImageClassification(config)
    model.to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start_iter, end_iter = 10, 20

    print("Profiling...")
    time_list = []
    hidden_sizes = [args.embed_dim, args.embed_dim*2, args.embed_dim*4, args.embed_dim*8]
    for ep in range(args.epochs):
        for iter, batch in enumerate(trainloader):
            if iter == end_iter:
                avg_time = sum(time_list) / len(time_list) / args.train_batch_size
                print("Average forward computation time of Swin", config.depths, "layers (hidden_size=", hidden_sizes, ") is: %.4f ms / bsz"%avg_time)
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
    args.depths = [2,2,6,2]
    time_2_2_6_2 = run_forward(args, conf)
    args.depths = [14,2,6,2]
    time_14_2_6_2 = run_forward(args, conf)
    args.depths = [2,14,6,2]
    time_2_14_6_2 = run_forward(args, conf)
    args.depths = [2,2,18,2]
    time_2_2_18_2 = run_forward(args, conf)
    args.depths = [2,2,6,14]
    time_2_2_6_14 = run_forward(args, conf)

    time_per_layer0 = (time_14_2_6_2-time_2_2_6_2)/12
    time_per_layer1 = (time_2_14_6_2-time_2_2_6_2)/12
    time_per_layer2 = (time_2_2_18_2-time_2_2_6_2)/12
    time_per_layer3 = (time_2_2_6_14-time_2_2_6_2)/12

    fwd_config_path = './configs/forward_profiling_config.json'
    config = read_json_config(fwd_config_path) if os.path.exists(fwd_config_path) else dict()
    hidden_sizes = [args.embed_dim, args.embed_dim*2, args.embed_dim*4, args.embed_dim*8]
    key = 'fwd_time_embed_dim_%d'%(args.embed_dim)
    config[key] = {'layer_type_0': time_per_layer0, 'layer_type_1': time_per_layer1,'layer_type_2': time_per_layer2,'layer_type_3': time_per_layer3}
    print('********************')
    print("Average forward computation time of Swin layer_type_0 (hidden_size=%d) is: %.4f ms / layer / bsz"%(hidden_sizes[0], time_per_layer0))
    print("Average forward computation time of Swin layer_type_1 (hidden_size=%d) is: %.4f ms / layer / bsz"%(hidden_sizes[1], time_per_layer1))
    print("Average forward computation time of Swin layer_type_2 (hidden_size=%d) is: %.4f ms / layer / bsz"%(hidden_sizes[2], time_per_layer2))
    print("Average forward computation time of Swin layer_type_3 (hidden_size=%d) is: %.4f ms / layer / bsz"%(hidden_sizes[3], time_per_layer3))
    write_json_config(config, fwd_config_path)
    print('Already written forward profiling config into env config file %s!\n'%(fwd_config_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.2, help="Drop path rate."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=12, help="Embed dim.",
    )
    parser.add_argument(
        "--depths", nargs='+', type=int, default=[2,2,18,2], help="Depths."
    )
    parser.add_argument(
        "--num_heads", nargs='+', type=int, default=[2], help="Num heads."
    )
    parser.add_argument(
        "--window_size", type=int, default=7, help="Window size."
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1.25e-4,
                        help="Learning rate of adam.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam."
    )
    parser.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--data-folder", default = 'ImageNet', type=str, help="Path to dataset."
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
        "--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    set_seed()

    profile(args, config)
