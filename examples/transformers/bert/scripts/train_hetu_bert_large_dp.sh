workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../train_hetu_bert_dp.py
config=${workdir}/../config4.yml
data_path=${workdir}/../data

heturun -c ${config} python ${mainpy} \
--num_gpus 4 \
--train_batch_size 32 \
--data_path ${data_path} \
--dataset wikicorpus_en \
--vocab_size 30522 \
--hidden_size 1024 \
--num_hidden_layers 24 \
--num_attention_heads 16 \
--seq_length 128 \
--epochs 20 \
--lr 1e-5 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1