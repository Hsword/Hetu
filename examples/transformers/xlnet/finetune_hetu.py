from tqdm import tqdm
import os
import math
import logging
import hetu as ht

from hetu_xlnet import XLNetForSequenceClassification
from xlnet_config import XLNetConfig
from utils import DataLoaderForGlue
import numpy as np
import time
import argparse
import urllib
import torch
      
      
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.mean(outputs == labels)

def get_tokenize(model="xlnet-base-cased", add_tokens=None):
    if not os.path.exists('spiece.model'):
        origin = ht.tokenizers.XLNetTokenizer.pretrained_vocab_files_map["vocab_file"][model]
        print('Downloading vocab from %s' % origin)
        urllib.request.urlretrieve(origin, "spiece.model")

    tokenizer = ht.tokenizers.XLNetTokenizer('spiece.model')
    if add_tokens is not None:
        tokenizer.add_tokens(add_tokens)
    return tokenizer
    
def finetune(args):
    device_id=args.gpu_id
    executor_ctx = ht.gpu(device_id)

    num_epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    seq_len = args.seq_length
    
    add_tokens = None
    task_name = args.task_name
    
    if task_name in ['sst-2','cola', 'mrpc']:
        num_labels = 2
    elif task_name in ['mnli']:
        num_labels = 3
        
    if task_name in {'mrpc', 'mnli'}:
        add_tokens = '<$>'
   
    tokenizer = get_tokenize(add_tokens=add_tokens)
    vocab_size = args.vocab_size
    new_num_tokens = len(tokenizer)
            
    config = XLNetConfig(vocab_size=new_num_tokens, 
                    d_model=args.hidden_size,
                    d_inner=args.hidden_size*4,
                    n_layer=args.num_hidden_layers, 
                    n_head=args.num_attention_heads, 
                    dropout=args.dropout_prob,
                    ff_activation=args.hidden_act,                   
                     num_labels=num_labels,
                    pad_token_id=tokenizer.pad_token_id)

    model = XLNetForSequenceClassification(config)

    dataloader = DataLoaderForGlue(task_name=task_name, batch_size = args.batch_size)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    label_ids = ht.Variable(name='label_ids', trainable=False)

    loss, logits = model(input_ids, (batch_size, seq_len), label_ids)
    loss= ht.reduce_mean_op(loss, [0])

    opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = args.adam_weight_decay)
    #opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)

    executor = ht.Executor([loss, logits, train_op], ctx=executor_ctx)
    

    if not os.path.exists('pytorch_model.bin'):
        origin = "https://huggingface.co/xlnet-base-cased/resolve/main/pytorch_model.bin"
        print('Downloading model from %s' % origin)
        urllib.request.urlretrieve(origin, 'pytorch_model.bin')    
           
    state_dict = torch.load('pytorch_model.bin')    
    model_dict = {key:state_dict[key].cpu().numpy() for key in state_dict}
    embedding_node = model.get_input_embeddings()

    for node in executor.param_nodes:
        if node == embedding_node.embedding_table and vocab_size!=new_num_tokens:
            value = model_dict['transformer.'+node.name]
            
            new_shape = executor.config.placeholder_to_arr_map[node].shape
            pre_shape = value.shape
                
            new_value = np.zeros(new_shape)
            n = min(new_shape[0], pre_shape[0])
            new_value[:n] = value[:n]
            executor.config.placeholder_to_arr_map[node] = ht.array(new_value, ctx=node.ctx)
            continue
        
        if node.name.startswith('summary') or node.name.startswith('logits_proj') :
            continue  

        pre_shape = executor.config.placeholder_to_arr_map[node].shape
        value = model_dict['transformer.'+node.name]
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
        "--batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--task_name", type=str, default='sst-2', help="Glue task to finetune."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000, help="Total number of vocab"
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
