# Please make sure that nproc_per_node==8//pp_deg when gpu_num==8!
# JSON config mode:   set nproc_per_node based on pp_deg in the JSON file in ./configs
# GLOBAL config mode: set nproc_per_node based on pp_deg in this script.
# PYTHON config mode: set nproc_per_node based on pp_deg in apply_strategy() in hybrid_parallel_model.py
# Please refer to README.md for detailed guidance!

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9995 train_hp_layerwise.py \
--seq-length 320 \
--global_train_batch_size 32 \
--embed_dim 320 \
--depths 2 2 42 2 \
--num_heads 8 16 32 64 \
--window_size 7 \
--epochs 10 \
--lr 1.25e-4 \
--adam_weight_decay 0.05 \
--data-folder ImageNet \
--pp_deg 2 \
--global_tp_deg 1 \
--global_tp_consec 1 \
--chunks -1 \
--fsdp 0 \
--profile 0 \
--check_loss 0 \
--apply_strategy 0 \
--galvatron_config_path ./configs/galvatron_config_8gpus_320embeddim_2_2_42_2_layers_example.json