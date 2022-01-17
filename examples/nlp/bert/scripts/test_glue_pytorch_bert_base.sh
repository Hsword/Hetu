python test_glue_pytorch_bert.py \
--gpu_id 1 \
--train_batch_size 64 \
--task_name sst-2 \
--vocab_size 30522 \
--hidden_size 768 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--seq_length 128 \
--epochs 20 \
--lr 2e-5 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1