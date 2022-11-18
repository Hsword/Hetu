from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_bert import BertForSequenceClassification
from bert_config import BertConfig
from load_data import DataLoaderForGlue
import numpy as np
import time
import argparse

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.mean(outputs == labels)

def finetune(args):
    device_id=args.gpu_id
    executor_ctx = ht.gpu(device_id)

    num_epochs = args.epochs
    lr = args.lr

    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob,
                        batch_size=args.train_batch_size,
                        hidden_act=args.hidden_act)

    task_name = args.task_name
    if task_name in ['sst-2','cola', 'mrpc']:
        num_labels = 2
    elif task_name in ['mnli']:
        num_labels = 3

    model = BertForSequenceClassification(config=config, num_labels = num_labels)

    dataloader = DataLoaderForGlue(task_name=task_name, batch_size = config.batch_size)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)

    label_ids = ht.Variable(name='label_ids', trainable=False)

    loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)
    loss= ht.reduce_mean_op(loss, [0])

    opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    # opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)

    executor = ht.Executor([loss, logits, train_op], ctx=executor_ctx)

    for ep in range(num_epochs):
        for i in range(dataloader.batch_num):
            start_time = time.time()

            batch_data = dataloader.get_batch(i)

            feed_dict = {
                input_ids: batch_data['input_ids'],
                token_type_ids: batch_data['token_type_ids'],
                attention_mask: batch_data['attention_mask'],
                label_ids: batch_data['label_ids'],
            }
            
            results = executor.run(feed_dict = feed_dict)

            loss_out = results[0].asnumpy()
            logits_out = results[1].asnumpy()
            acc = accuracy(logits_out, batch_data['label_ids'])

            end_time = time.time()
            print('[Epoch %d] (Iteration %d): Loss = %.3f, Accuracy = %.4f Time = %.3f'%(ep,i,loss_out, acc, end_time-start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--task_name", type=str, default='sst-2', help="Glue task to finetune."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
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

    finetune(args)
