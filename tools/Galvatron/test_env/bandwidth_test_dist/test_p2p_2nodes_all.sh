export NUM_NODES=2
export NUM_GPUS_PER_NODE=8
export MASTER_PORT=9991

# # Export the right MASTER_ADDR, NCCL_SOCKET_IFNAME and NODE_RANK into environment before running scripts
# export MASTER_ADDR=162.105.146.118
# export NCCL_SOCKET_IFNAME=ib0
# export NODE_RANK=0

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK test_p2p_dist.py \
--global_tp_deg 1 \
--global_tp_consec 1 \
--pp_deg 2

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK test_p2p_dist.py \
--global_tp_deg 1 \
--global_tp_consec 1 \
--pp_deg 4

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK test_p2p_dist.py \
--global_tp_deg 1 \
--global_tp_consec 1 \
--pp_deg 8

python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK test_p2p_dist.py \
--global_tp_deg 1 \
--global_tp_consec 1 \
--pp_deg 16