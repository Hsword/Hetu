python train.py \
--embed_dim 320 \
--depths 2 2 26 2 \
--num_heads 8 16 32 64 \
--window_size 7 \
--train_batch_size 8 \
--epochs 10 \
--lr 1.25e-4 \
--adam_weight_decay 0.05 \
--data-folder ImageNet \
--drop_path_rate 0.2 \
--check_loss 0 \
--profile 0
