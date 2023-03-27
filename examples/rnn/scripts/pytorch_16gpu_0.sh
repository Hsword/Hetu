#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=162.105.146.117
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../torch_main.py

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ${mainpy} \
        --model $1 --dataset $2 --learning-rate 0.01 --validate --timing --distributed