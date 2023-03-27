workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../train_hetu_bert_dp.py
config=${workdir}/../config4.yml
data_path=${workdir}/../data
export PYTHONPATH=$HETU_PATH
heturun -c ${config} python ${mainpy} \
--num_gpus 4 \
--train_batch_size 64 \
--data_path ${data_path} \
--dataset wikicorpus_en \
--vocab_size 30522 \
--hidden_size 768 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--seq_length 512 \
--epochs 80 \
--lr 1e-5 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1
