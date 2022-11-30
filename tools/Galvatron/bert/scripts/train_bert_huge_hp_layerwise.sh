# Please make sure that nproc_per_node==8//pp_deg when gpu_num==8!
# JSON config mode:   set nproc_per_node based on pp_deg in the JSON file in ./configs
# GLOBAL config mode: set nproc_per_node based on pp_deg in this script.
# PYTHON config mode: set nproc_per_node based on pp_deg in apply_strategy() in hybrid_parallel_model.py
# Please refer to README.md for detailed guidance!

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 train_hp_layerwise.py \
--global_train_batch_size 8 \
--vocab_size 30522 \
--hidden_size 1280 \
--num_hidden_layers 32 \
--num_attention_heads 16 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0 \
--pp_deg 2 \
--global_tp_deg 2 \
--global_tp_consec 1 \
--chunks -1 \
--fsdp 0 \
--profile 0 \
--apply_strategy 0 \
--galvatron_config_path ./configs/galvatron_config_8gpus_1280hidden_32layers_example.json
