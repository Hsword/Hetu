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
from load_data import DataLoader4Glue
import time

''' Usage example:
    In dir Hetu/examples/nlp/bert/: python test_glue_pytorch_bert.py
'''

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

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:4" if cuda_condition else "cpu")
#task_name = 'cola'
task_name = 'sst-2'
if task_name in ['sst-2','cola', 'mrpc']:
    num_labels = 2
elif task_name in ['mnli']:
    num_labels = 3

def finetune():
    num_epochs = 4
    lr = 5e-5

    config = BertConfig(vocab_size=30522, 
                        hidden_size=768,
                        num_hidden_layers=12, 
                        num_attention_heads=12, 
                        intermediate_size=3072, 
                        max_position_embeddings=512, 
                        attention_probs_dropout_prob=0.0,
                        hidden_dropout_prob=0.0,
                        batch_size=16)

    model = BertForSequenceClassification(config=config, num_labels=num_labels)
    model.to(device)

    batch_size = config.batch_size
    seq_len = config.max_position_embeddings
    vocab_size = config.vocab_size

    dataloader = DataLoader4Glue(task_name=task_name, batch_size = batch_size)
    data_names = ['input_ids','token_type_ids','attention_mask','label_ids']

    dataloader_dev = DataLoader4Glue(task_name=task_name, batch_size = batch_size, datatype='dev')

    #initialize parameters
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(m.weight)

    #save parameters
    params = model.state_dict()
    for key, val in params.items():
        params[key] = val.cpu().numpy()
    torch.save(params, "pytorch_params_glue.file") 

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

    opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay = 0.01)
    opt_name = 'Adam'
    # opt = SGD(model.parameters(), lr=lr)
    # opt_name = 'SGD'

    dataloader.make_epoch_data()
    dataloader_dev.make_epoch_data()

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

def validate():
    config = BertConfig(vocab_size=30522, 
                        hidden_size=768,
                        num_hidden_layers=12, 
                        num_attention_heads=12, 
                        intermediate_size=3072, 
                        max_position_embeddings=512, 
                        attention_probs_dropout_prob=0.0,
                        hidden_dropout_prob=0.0,
                        batch_size=16)

    model = BertForSequenceClassification(config=config, num_labels=num_labels)
    model.to(device)

    batch_size = config.batch_size
    seq_len = config.max_position_embeddings
    vocab_size = config.vocab_size

    dataloader = DataLoader4Glue(task_name=task_name, batch_size = batch_size, datatype='dev')
    data_names = ['input_ids','token_type_ids','attention_mask','label_ids']

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
    dataloader.make_epoch_data()
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
    finetune()
    #validate()

