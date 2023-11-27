export NUM_NODES=2
export NUM_GPUS_PER_NODE=8

# # Export the right MASTER_ADDR, NCCL_SOCKET_IFNAME and NODE_RANK into environment before running scripts
# export MASTER_ADDR=162.105.146.118
# export NCCL_SOCKET_IFNAME=ib0
# export NODE_RANK=0

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=9991 --node_rank=$NODE_RANK train_hp_layerwise_dist.py \
--global_train_batch_size 64 \
--hidden_size 1024 \
--num_hidden_layers 24 \
--num_attention_heads 16 \
--image_size 224 \
--patch_size 16 \
--seq_length 196 \
--num_channels 3 \
--epochs 10 \
--lr 3e-5 \
--adam_weight_decay 0.01 \
--dropout_prob 0.0 \
--data-folder ImageNet \
--check_loss 0 \
--pp_deg 2 \
--global_tp_deg 2 \
--global_tp_consec 1 \
--chunks -1 \
--fsdp 0 \
--profile 0 \
--apply_strategy 0 \
--galvatron_config_path ./configs/galvatron_config_16gpus_1024hidden_24layers_example.json
