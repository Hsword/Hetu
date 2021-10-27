from tqdm import tqdm
import os
import math
import logging
import hetu as ht
from hetu_bert import BertForSequenceClassification
from bert_config import BertConfig
from load_data import DataLoader4Glue
import numpy as np
import time

''' Usage example:
    In dir Hetu/examples/nlp/bert/: python test_glue_hetu_bert.py
'''

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.mean(outputs == labels)

device_id=6
executor_ctx = ht.gpu(device_id)

num_epochs = 1
lr = 1e-4

config = BertConfig(vocab_size=30522, 
                    hidden_size=768,
                    num_hidden_layers=12, 
                    num_attention_heads=12, 
                    intermediate_size=3072, 
                    max_position_embeddings=512, 
                    attention_probs_dropout_prob=0.0,
                    hidden_dropout_prob=0.0,
                    batch_size=16)

task_name = 'cola'
if task_name in ['sst-2','cola', 'mrpc']:
    num_labels = 2
elif task_name in ['mnli']:
    num_labels = 3

model = BertForSequenceClassification(config=config, num_labels = num_labels)

batch_size = config.batch_size
seq_len = config.max_position_embeddings
vocab_size = config.vocab_size

dataloader = DataLoader4Glue(task_name=task_name, batch_size = batch_size)
data_names = ['input_ids','token_type_ids','attention_mask','label_ids']

input_ids = ht.Variable(name='input_ids', trainable=False)
token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
attention_mask = ht.Variable(name='attention_mask', trainable=False)

label_ids = ht.Variable(name='label_ids', trainable=False)

loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids)
loss= ht.reduce_mean_op(loss, [0])

opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = 0.01)
# opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
# opt = ht.optim.SGDOptimizer(learning_rate=lr)
train_op = opt.minimize(loss)

executor = ht.Executor([loss, logits, train_op],ctx=executor_ctx,dynamic_memory=True)


dataloader.make_epoch_data()
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

