from tqdm import tqdm
import logging
import math
import os
import hetu as ht
from hetu_bert import BertForPreTraining
from bert_config import BertConfig
from load_data import DataLoader
import numpy as np
import time
import argparse

''' Usage example:
    In dir Hetu/examples/nlp/bert/: heturun -c config4.yml python train_hetu_bert_dp.py --num_gpus 4
'''

def pretrain(args):
    if args.num_gpus == 1:
        device_id=0
        executor_ctx = [ht.gpu(device_id)]
    elif args.num_gpus > 1:
        executor_ctx = [ht.gpu(i) for i in range(args.num_gpus)]

    num_epochs = 1
    lr = 1e-4

    with ht.context(executor_ctx):
        config = BertConfig(vocab_size=30522, 
                            hidden_size=768,
                            num_hidden_layers=12, 
                            num_attention_heads=12, 
                            intermediate_size=3072, 
                            max_position_embeddings=512, 
                            # attention_probs_dropout_prob=0.0,
                            # hidden_dropout_prob=0.0,
                            batch_size=20)

        model = BertForPreTraining(config=config)

        batch_size = config.batch_size
        seq_len = config.max_position_embeddings
        vocab_size = config.vocab_size

        dataloader = DataLoader(dataset='bookcorpus', doc_num=200, save_gap=200, batch_size = batch_size)
        data_names = ['input_ids','token_type_ids','attention_mask','masked_lm_labels','next_sentence_label']

        input_ids = ht.Variable(name='input_ids', trainable=False)
        token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
        attention_mask = ht.Variable(name='attention_mask', trainable=False)

        masked_lm_labels = ht.Variable(name='masked_lm_labels', trainable=False)
        next_sentence_label = ht.Variable(name='next_sentence_label', trainable=False)

        loss_position_sum = ht.Variable(name='loss_position_sum', trainable=False)

        #prediction_scores, seq_relationship_score, masked_lm_loss, next_sentence_loss = model(input_ids, token_type_ids, attention_mask, masked_lm_labels_one_hot, next_sentence_label_one_hot)
        _,_, masked_lm_loss, next_sentence_loss = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)

        masked_lm_loss_mean = ht.div_op(ht.reduce_sum_op(masked_lm_loss, [0,1]), loss_position_sum)
        next_sentence_loss_mean = ht.reduce_mean_op(next_sentence_loss, [0])

        loss = masked_lm_loss_mean + next_sentence_loss_mean
        opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = 0.01)
        # opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # opt = ht.optim.SGDOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)

        executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean, loss, train_op],dynamic_memory=True)

        dataloader.make_epoch_data()
        for ep in range(num_epochs):
            for i in range(dataloader.batch_num):
                start_time = time.time()

                batch_data = dataloader.get_batch(i)
                feed_dict = {
                    input_ids: batch_data['input_ids'],
                    token_type_ids: batch_data['token_type_ids'],
                    attention_mask: batch_data['attention_mask'],
                    masked_lm_labels: batch_data['masked_lm_labels'],
                    next_sentence_label: batch_data['next_sentence_label'],
                    loss_position_sum: np.array([np.where(batch_data['masked_lm_labels'].reshape(-1)!=-1)[0].shape[0]]),
                }

                results = executor.run(feed_dict = feed_dict)

                masked_lm_loss_mean_out = results[0].asnumpy()
                next_sentence_loss_mean_out = results[1].asnumpy()
                loss_out = results[2].asnumpy()

                end_time = time.time()
                print('[Epoch %d] (Iteration %d): Loss = %.3f, MLM_loss = %.3f, NSP_loss = %.6f, Time = %.3f'%(ep,i,loss_out, masked_lm_loss_mean_out, next_sentence_loss_mean_out, end_time-start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Num of gpus used to train the model.')
    args = parser.parse_args()
    if args.num_gpus >8 or args.num_gpus<1:
        print("num_gpus should be int between 1~8.")
    pretrain(args)
