import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
from dataloader import DataLoaderForBert
import argparse
from tqdm import tqdm
import numpy as np
import random
import h5py
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

def model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label):
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    prediction_scores, seq_relationship_score = outputs.prediction_logits, outputs.seq_relationship_logits
    loss_fct = nn.CrossEntropyLoss(ignore_index = -1)
    masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), masked_lm_labels.view(-1))
    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
    loss = masked_lm_loss + next_sentence_loss
    return loss, masked_lm_loss, next_sentence_loss

def run_forward(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    rank = args.gpu_id

    print("Creating Dataloader...")
    dataset = DataLoaderForBert()
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            shuffle=False)

    print("Creating Model...")
    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob)
    model = BertForPreTraining(config)
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
                print("Average forward computation time of %d Bert layers (hidden_size=%d) is: %.4f ms / bsz"%(args.num_hidden_layers, args.hidden_size, avg_time))
                return avg_time
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label= [tensor.to(device) for tensor in batch]
            torch.cuda.synchronize()
            start.record()
            loss, masked_lm_loss, next_sentence_loss = \
                model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label)
            end.record()
            torch.cuda.synchronize()
            iter_time = start.elapsed_time(end)
            if iter >= start_iter:
                time_list.append(iter_time)

def profile(args):
    args.num_hidden_layers = 24
    time_24_layers = run_forward(args)
    args.num_hidden_layers = 12
    time_12_layers = run_forward(args)
    time_per_layer = (time_24_layers-time_12_layers)/12

    fwd_config_path = './configs/forward_profiling_config.json'
    config = read_json_config(fwd_config_path) if os.path.exists(fwd_config_path) else dict()
    key = 'fwd_time_hidden_%d'%(args.hidden_size)
    config[key] = time_per_layer
    print('********************')
    print("Average forward computation time of Bert layer (hidden_size=%d) is: %.4f ms / layer / bsz"%(args.hidden_size, time_per_layer))
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
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
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

    args = parser.parse_args()
    set_seed()
    profile(args)