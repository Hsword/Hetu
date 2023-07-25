from tqdm import tqdm
import os
import math
import logging
import hetu as ht

from hetu_reformer import ReformerForPretraining
from reformer_config import ReformerConfig
from utils import DataLoader
import numpy as np
import time
import argparse
    
def pretrain(args):        
    dataset = args.dataset
    num_epochs = args.epochs
    lr = args.lr

    input_shape = (args.batch_size, args.seq_length)
    
    config = ReformerConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_attention_heads=args.num_attention_heads, 
                        max_position_embeddings=4096, 
                        hidden_dropout_prob=args.dropout_prob,
                        local_attention_probs_dropout_prob=args.dropout_prob,
                        lsh_attention_probs_dropout_prob=args.dropout_prob,
                        classifier_dropout=args.dropout_prob,
                        hidden_act=args.hidden_act)
                        
    data_dir = os.path.join('data', dataset)
    train_file = os.path.join(data_dir, 'train.bin')   

    model = ReformerForPretraining(config)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    labels = ht.Variable(name='labels', trainable=False)
    
    loss, lm_logits = model(input_ids, input_shape, labels=labels)
    loss = ht.reduce_mean_op(loss, [0, 1])

    opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    #opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)


    if args.num_gpus == 1:
        device_id = args.gpu_id
        executor_ctx = ht.gpu(device_id)
        executor = ht.Executor([loss, train_op], ctx=executor_ctx)
        rank, nrank = 0, 1
    elif args.num_gpus > 1:
        executor_ctx = [ht.gpu(i) for i in range(args.num_gpus)]
        strategy = ht.dist.DataParallel(aggregate='allreduce')
        executor = ht.Executor([loss, train_op], dist_strategy=strategy)
        rank, nrank = executor.rank, executor.config.nrank

    global_step_num = 0
    for ep in range(num_epochs):
        step_num = 0
        dataloader = DataLoader(train_file, args.batch_size, seed=ep)  
        batch_num_per_device = int(dataloader.batch_num/nrank)
        for i in range(batch_num_per_device):
            start_time = time.time()
            batch_data = dataloader.get_batch(i * nrank + rank)
            feed_dict = {
                input_ids: batch_data['X'],
                labels: batch_data['Y'],
            }
            results = executor.run(feed_dict = feed_dict)
            loss_out = results[0].asnumpy()
            end_time = time.time()
            print('[Epoch %d] (Iteration %d): Loss = %.3f, Time = %.3f'%(ep,step_num,loss_out, end_time-start_time))
            step_num += 1
            global_step_num += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--num_gpus', type=int, default=1, help='Num of gpus used to train the model.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Training batch size"
    )
    parser.add_argument(
        "--dataset", type=str, default='shakespeare', help="Dataset used to train."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50265, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size of transformer model",
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
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    args = parser.parse_args()

    pretrain(args)
