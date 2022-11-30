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

def train(args):
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
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

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
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label= [tensor.to(device) for tensor in batch]
            if args.profile and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            loss, masked_lm_loss, next_sentence_loss = \
                model_forward(config, model, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label)

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
                print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f'% \
                    (ep,iter,loss.item(), masked_lm_loss.item(), next_sentence_loss.item()))

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
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
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
    train(args)