export NUM_NODES=2
export NUM_GPUS_PER_NODE=8

# # Export the right MASTER_ADDR, NCCL_SOCKET_IFNAME and NODE_RANK into environment before running scripts
# export MASTER_ADDR=162.105.146.118
# export NCCL_SOCKET_IFNAME=ib0
# export NODE_RANK=0

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=9991 --node_rank=$NODE_RANK train_hp_layerwise_dist.py \
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
--global_tp_deg 2 \
--global_tp_consec 1 \
--chunks -1 \
--fsdp 0 \
--profile 0 \
--check_loss 0 \
--apply_strategy 0 \
--galvatron_config_path ./configs/galvatron_config_16gpus_320embeddim_2_2_42_2_layers_example.json
