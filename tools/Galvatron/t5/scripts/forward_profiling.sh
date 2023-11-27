python profile_forward.py \
--gpu_id 0 \
--train_batch_size 1 \
--model_config t5-large \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0