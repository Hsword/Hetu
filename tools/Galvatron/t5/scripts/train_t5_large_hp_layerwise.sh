# Please make sure that nproc_per_node==8//pp_deg when gpu_num==8!
# JSON config mode:   set nproc_per_node based on pp_deg in the JSON file in ./configs
# GLOBAL config mode: set nproc_per_node based on pp_deg in this script.
# PYTHON config mode: set nproc_per_node based on pp_deg in apply_strategy() in hybrid_parallel_model.py
# Please refer to README.md for detailed guidance!

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9992 train_hp_layerwise.py \
--global_train_batch_size 8 \
--model_config t5-large \
--num_encoder_layer 24 \
--num_decoder_layer 24 \
--seq_length 512 \
--epochs 10 \
--lr 1e-4 \
--weight_decay 0.01 \
--dropout_prob 0.1 \
--check_loss 0 \
--pp_deg 2 \
--global_tp_deg 2 \
--global_tp_consec 1 \
--chunks -1 \
--fsdp 0 \
--profile 0 \
--apply_strategy 0 \
--galvatron_config_path ./configs/galvatron_config_8gpus_1024hidden_24enc_24dec_example.json