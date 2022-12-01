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
from utils import print_peak_memory
from utils import read_json_config, write_json_config

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def run_forward(args):
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
    config.num_layers = args.num_encoder_layer
    config.num_decoder_layers = args.num_decoder_layer
    model = T5ForConditionalGeneration(config)
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
                print("Average forward computation time of T5 %d encoder and %d decoder layers (hidden_size=%d) is: %.4f ms / bsz"%(config.num_layers, config.num_decoder_layers, args.hidden_size, avg_time))
                return avg_time
            input_ids, attention_mask, labels= [tensor.to(device) for tensor in batch]
            torch.cuda.synchronize()
            start.record()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            end.record()
            torch.cuda.synchronize()
            iter_time = start.elapsed_time(end)
            if iter >= start_iter:
                time_list.append(iter_time)

def profile(args):
    args.hidden_size = 1024
    args.num_encoder_layer, args.num_decoder_layer = 24, 24
    time_24_enc_24_dec = run_forward(args)
    args.num_encoder_layer, args.num_decoder_layer = 24, 12
    time_24_enc_12_dec = run_forward(args)
    args.num_encoder_layer, args.num_decoder_layer = 12, 24
    time_12_enc_24_dec = run_forward(args)
    time_per_layer_enc = (time_24_enc_24_dec-time_12_enc_24_dec)/12
    time_per_layer_dec = (time_24_enc_24_dec-time_24_enc_12_dec)/12

    fwd_config_path = './configs/forward_profiling_config.json'
    config = read_json_config(fwd_config_path) if os.path.exists(fwd_config_path) else dict()
    key = 'fwd_time_hidden_%d'%(args.hidden_size)
    config[key] = {'encoder': time_per_layer_enc, 'decoder': time_per_layer_dec}
    print('********************')
    print("Average forward computation time of T5 encoder layer (hidden_size=%d) is: %.4f ms / layer / bsz"%(args.hidden_size, time_per_layer_enc))
    print("Average forward computation time of T5 decoder layer (hidden_size=%d) is: %.4f ms / layer / bsz"%(args.hidden_size, time_per_layer_dec))
    write_json_config(config, fwd_config_path)
    print('Already written forward profiling config into env config file %s!\n'%(fwd_config_path))

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
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8],
    )
    parser.add_argument(
        "--num_encoder_layer", type=int, default=0, help="overwrite encoder layer num"
    )
    parser.add_argument(
        "--num_decoder_layer", type=int, default=0, help="overwrite decoder layer num"
    )

    args = parser.parse_args()
    set_seed()
    profile(args)