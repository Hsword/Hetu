export NUM_NODES=2
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"

MODEL_ARGS="
    --model_size llama-7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 32 \
    --num_attention_heads 32 \
    --seq_length 2048"

TRAIN_ARGS="
    --global_train_batch_size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --save_profiled_memory 0"

PARALLEL_ARGS="
    --pp_deg 2 \
    --global_tp_deg 2 \
    --global_tp_consec 1 \
    --sdp 0 \
    --global_checkpoint 0 \
    --chunks 1 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --galvatron_config_path ./configs/galvatron_config_llama-7b_2nodes_8gpus_per_node_40GB_bf16_example.json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS}