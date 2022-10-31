#!/bin/bash

heturun -s 1 -w 4 python train_hetu_bert_ps.py \
--train_batch_size 32 \
--dataset wikicorpus_en \
--vocab_size 30522 \
--hidden_size 768 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--seq_length 128 \
--epochs 20 \
--lr 1e-5 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1
