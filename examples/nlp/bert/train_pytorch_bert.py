from tqdm import tqdm
import os
import math
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import numpy as np
from pytorch_bert import BertModel, BertForPreTraining
from bert_config import BertConfig
from load_data import DataLoader
import time

''' Usage example:
    In dir Hetu/examples/nlp/bert/: python train_pytorch_bert.py
'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(123)

num_epochs = 1
lr = 1e-4

cuda_condition = torch.cuda.is_available()
device = torch.device("cuda:5" if cuda_condition else "cpu")

config = BertConfig(vocab_size=30522, 
                    hidden_size=768,
                    num_hidden_layers=12, 
                    num_attention_heads=12, 
                    intermediate_size=3072, 
                    max_position_embeddings=512, 
                    #attention_probs_dropout_prob=0.0,
                    #hidden_dropout_prob=0.0,
                    batch_size=6)

model = BertForPreTraining(config=config)
model.to(device)

batch_size = config.batch_size
seq_len = config.max_position_embeddings
vocab_size = config.vocab_size

dataloader = DataLoader(dataset='bookcorpus', doc_num=200, save_gap=200, batch_size = batch_size)
data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']

#save parameters
for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_normal_(m.weight)

params = model.state_dict()
torch.save(model.state_dict(), "pytorch_params.file") 

#opt = Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay = 0)
opt = SGD(model.parameters(), lr=lr)

dataloader.make_epoch_data()

for ep in range(num_epochs):
    for i in range(dataloader.batch_num):
        batch_data = dataloader.get_batch(i)
        input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
        token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)
        attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
        masked_lm_labels = torch.LongTensor(batch_data['masked_lm_labels']).to(device)
        next_sentence_label = torch.LongTensor(batch_data['next_sentence_label']).to(device)

        opt.zero_grad()
        start_time = time.time()
        _,_, masked_lm_loss_mean, next_sentence_loss_mean = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)   
        loss = masked_lm_loss_mean + next_sentence_loss_mean        
        loss.backward()
        opt.step()
        end_time = time.time()

        masked_lm_loss_out = masked_lm_loss_mean.item()
        next_sentence_loss_out = next_sentence_loss_mean.item()
        loss_out = loss.item()

        print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'%(ep,i,loss_out, masked_lm_loss_out, next_sentence_loss_out, end_time-start_time))


