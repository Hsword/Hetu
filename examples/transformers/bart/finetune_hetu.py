from tqdm import tqdm
import os
import math
import logging
import hetu as ht

from hetu_bart import BartForSequenceClassification
from bart_config import BartConfig

from utils import DataLoaderForGlue
import numpy as np
import time
import argparse
import urllib
import torch
      
      
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.mean(outputs == labels)

def finetune(args):
    device_id=args.gpu_id
    executor_ctx = ht.gpu(device_id)

    num_epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    seq_len = args.seq_length

    task_name = args.task_name
    if task_name in ['sst-2','cola', 'mrpc']:
        num_labels = 2
    elif task_name in ['mnli']:
        num_labels = 3
    
    vocab_size = args.vocab_size
    if task_name in ['sst-2','cola']:
        new_num_tokens = vocab_size
    else:
        new_num_tokens = vocab_size + 1
            
    config = BartConfig(vocab_size=new_num_tokens, 
                        d_model=args.hidden_size,
                        encoder_layers=args.num_hidden_layers, 
                        decoder_layers=args.num_hidden_layers, 
                        encoder_attention_heads=args.num_attention_heads, 
                        decoder_attention_heads=args.num_attention_heads, 
                        encoder_ffn_dim=args.hidden_size*4, 
                        decoder_ffn_dim=args.hidden_size*4, 
                        max_position_embeddings=1024, 
                        attention_dropout=args.dropout_prob,
                        activation_dropout=args.dropout_prob,
                        dropout=args.dropout_prob,
                        encoder_layerdrop=args.dropout_prob,
                        decoder_layerdrop=args.dropout_prob,
                        classifier_dropout=args.dropout_prob,
                        num_labels=num_labels,
                        activation_function=args.hidden_act)

    model = BartForSequenceClassification(config)

    dataloader = DataLoaderForGlue(task_name=task_name, batch_size = args.batch_size)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)
    label_ids = ht.Variable(name='label_ids', trainable=False)

    loss, logits = model(input_ids, (batch_size, seq_len), attention_mask, label_ids)
    loss= ht.reduce_mean_op(loss, [0])

    opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    #opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    #opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)

    executor = ht.Executor([loss, logits, train_op], ctx=executor_ctx)
    

    if not os.path.exists('pytorch_model.bin'):
        origin = "https://huggingface.co/facebook/bart-base/resolve/main/pytorch_model.bin"
        print('Downloading model from %s' % origin)
        urllib.request.urlretrieve(origin, 'pytorch_model.bin')    
           
    state_dict = torch.load('pytorch_model.bin')    
    model_dict = {key:state_dict[key].cpu().numpy() for key in state_dict}
    embedding_node = model.get_input_embeddings()

    for node in executor.param_nodes:
        if node == embedding_node and vocab_size!=new_num_tokens:
            value = model_dict[node.name]
            assert value.shape[0] == vocab_size
            pre_shape = executor.config.placeholder_to_arr_map[node].shape
            
            new_shape = (new_num_tokens, pre_shape[1])        
            new_value = np.zeros(new_shape)
            n = min(vocab_size, new_num_tokens)
            new_value[:n] = value[:n]
            executor.config.placeholder_to_arr_map[node] = ht.array(new_value, ctx=node.ctx)
            continue
        
        if node.name.startswith('classification_head'):
            continue  

        pre_shape = executor.config.placeholder_to_arr_map[node].shape
        value = model_dict[node.name]
        if node.name=='logit_scale':
            executor.config.placeholder_to_arr_map[node][:] = value
            continue
        cur_shape = value.shape
        assert pre_shape == cur_shape, 'Shape not conform! Got {} and {} for {}.'.format(pre_shape, cur_shape, node.name)
        executor.config.placeholder_to_arr_map[node][:] = value 


    for ep in range(num_epochs):
        for i in range(dataloader.batch_num):
            start_time = time.time()
            batch_data = dataloader.get_batch(i)

            feed_dict = {
                input_ids: batch_data['input_ids'],
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
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--task_name", type=str, default='sst-2', help="Glue task to finetune."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50265, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=6, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Sequence len"
    )
    parser.add_argument(
        "--max_position_embeddings", type=int, default=1024, help="Maximum sequence len"
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
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
