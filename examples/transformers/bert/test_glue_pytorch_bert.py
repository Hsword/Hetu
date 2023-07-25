from tqdm import tqdm
import os
import math
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from pytorch_bert import BertForSequenceClassification
from bert_config import BertConfig
from load_data import DataLoaderForGlue
import time
import argparse

def params_from_official_pytorch_pretrained_model(state_dict):
    weights_path = './pretrained_params/bert-base-uncased/pytorch_model.bin'
    pretrained_state_dict = torch.load(weights_path, map_location='cpu' if not torch.cuda.is_available() else None)

    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in pretrained_state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if 'intermediate.dense' in key:
            new_key = key.replace('intermediate.dense', 'intermediate.dense_act.Linear')
        if 'pooler.dense' in key:
            new_key = key.replace('pooler.dense', 'pooler.dense_act.Linear')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pretrained_state_dict[new_key] = pretrained_state_dict.pop(old_key)
    for key in state_dict.keys():
        if 'bert.' in key:
            state_dict[key] = pretrained_state_dict[key]
    print("Successfully loaded pretrained parameters from %s"%weights_path)
    return state_dict

def params_from_pytorch_pretrained_model(state_dict, model_path):
    pytorch_state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    for key in state_dict.keys():
        if 'bert.' in key:
            state_dict[key] = pytorch_state_dict[key]
    return state_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)


def finetune(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    task_name = args.task_name
    if task_name in ['sst-2','cola', 'mrpc']:
        num_labels = 2
    elif task_name in ['mnli']:
        num_labels = 3

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

    model = BertForSequenceClassification(config=config, num_labels=num_labels)
    model.to(device)

    dataloader = DataLoaderForGlue(task_name=task_name, batch_size = config.batch_size)

    dataloader_dev = DataLoaderForGlue(task_name=task_name, batch_size = config.batch_size, datatype='dev')

    #initialize parameters
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(m.weight)

    # #save parameters
    # params = model.state_dict()
    # for key, val in params.items():
    #     params[key] = val.cpu().numpy()
    # torch.save(params, "pytorch_params_glue.file") 

    start_model = 'random'

    # load parameters of BERT from official pretrained pytorch model
    # state_dict = params_from_official_pytorch_pretrained_model(model.state_dict())
    # start_model = 'pytorch_official'

    # load parameters of BERT from pretrained pytorch model
    # load_ep, load_i = 2, 145000
    # #pytorch_model_path = './pretrained_params/pytorch_pretrained_params/epoch_%d_iter_%d.params'%(load_ep,load_i)
    # pytorch_model_path = './pretrained_params/pytorch_pretrained_params_adam/epoch_%d_iter_%d.params'%(load_ep,load_i)
    # state_dict= params_from_pytorch_pretrained_model(model.state_dict(), pytorch_model_path)
    # start_model = 'pytorch_ep%d_iter%d'%(load_ep, load_i)


    # model.load_state_dict(state_dict)

    opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay = args.adam_weight_decay)
    opt_name = 'Adam'
    # opt = SGD(model.parameters(), lr=lr)
    # opt_name = 'SGD'

    for ep in range(num_epochs):
        for i in range(dataloader.batch_num):
            start_time = time.time()

            batch_data = dataloader.get_batch(i)
            input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
            token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
            attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
            label_ids = torch.LongTensor(batch_data['label_ids']).to(device)

            model.train()
            opt.zero_grad()
            loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)        
            loss.backward()
            opt.step()

            loss_out = loss.item()
            pred = logits.argmax(dim=1)
            acc = torch.eq(pred, label_ids).float().mean().item()

            end_time = time.time()
            print('[Epoch %d] (Iteration %d): Loss = %.3f, Accuracy = %.4f Time = %.3f'%(ep,i,loss_out, acc, end_time-start_time))
        
        # # validate model on dev set
        # acc_list=[]
        # for i in range(dataloader_dev.batch_num):
        #     batch_data = dataloader_dev.get_batch(i)
        #     input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
        #     token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
        #     attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
        #     label_ids = torch.LongTensor(batch_data['label_ids']).to(device)

        #     model.eval()
        #     start_time = time.time()
        #     loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)
        #     end_time = time.time()

        #     loss_out = loss.item()
        #     pred = logits.argmax(dim=1)
        #     acc = torch.eq(pred, label_ids).float().mean().item()
        #     acc_list.append(acc)
        #     print('[Validate] (Iteration %d): Loss = %.3f, Accuracy = %.4f Time = %.3f'%(i,loss_out, acc, end_time-start_time))

        # print('\tDev accuracy after epoch %d is %.4f'%(ep, np.mean(np.array(acc_list))))    


        save_path = './finetuned_params/pytorch_finetuned_params/%s/'%start_model
        # save_path = './finetuned_params/pytorch_finetuned_params_adam_pretrain/%s/'%start_model
        save_file = '%s_epoch_%d_%s.params'%(task_name,ep,opt_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+save_file) 
        print('Saved model to %s.'%(save_path+save_file))

def validate(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    task_name = args.task_name
    if task_name in ['sst-2','cola', 'mrpc']:
        num_labels = 2
    elif task_name in ['mnli']:
        num_labels = 3

    config = BertConfig(vocab_size=args.vocab_size, 
                        hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        max_position_embeddings=args.seq_length, 
                        attention_probs_dropout_prob=0.0,
                        hidden_dropout_prob=0.0,
                        batch_size=args.train_batch_size,
                        hidden_act=args.hidden_act)

    model = BertForSequenceClassification(config=config, num_labels=num_labels)
    model.to(device)

    dataloader = DataLoaderForGlue(task_name=task_name, batch_size = config.batch_size, datatype='dev')

    start_model = 'random'

    # start_model = 'pytorch_official'

    # load_ep, load_i =2, 145000
    # start_model = 'pytorch_ep%d_iter%d'%(load_ep, load_i)

    load_finetune_ep = 3
    opt_name = 'Adam'
    # opt_name = 'SGD'
    save_path = './finetuned_params/pytorch_finetuned_params/%s/'%start_model
    #save_path = './finetuned_params/pytorch_finetuned_params_adam_pretrain/%s/'%start_model
    save_file = '%s_epoch_%d_%s.params'%(task_name,load_finetune_ep,opt_name)
    state_dict = torch.load(save_path+save_file, map_location='cpu' if not torch.cuda.is_available() else None)
    model.load_state_dict(state_dict)

    # validate model on dev set
    acc_list=[]
    for i in range(dataloader.batch_num):
        start_time = time.time()

        batch_data = dataloader.get_batch(i)
        input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
        token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
        attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
        label_ids = torch.LongTensor(batch_data['label_ids']).to(device)

        model.eval()

        loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)

        loss_out = loss.item()
        pred = logits.argmax(dim=1)
        acc = torch.eq(pred, label_ids).float().mean().item()
        acc_list.append(acc)

        end_time = time.time()
        print('[Validate] (Iteration %d): Loss = %.3f, Accuracy = %.4f Time = %.3f'%(i,loss_out, acc, end_time-start_time))

    print('\tDev accuracy after epoch %d is %.4f'%(load_finetune_ep, np.mean(np.array(acc_list))))    

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
    #validate(args)

