# Galvatron Model Usage

Galvatron provides sample code for a bunch of mainstream models to demonstrate how a Transformer model should be rewritten to accommodate Galvatron's automatic optimization API. In addition, users can quickly start from these models, optimizing parallelism strategies in their own hardware environment. Enter model directory by ```cd model_name``` to start.

## Training with Galvatron

To train the model with Galvatron, run:
``` shell
sh scripts/train_dist.sh
```

Users can customize multiple training options:

### Model Configuration
Users can set `model_size` and easily get a pre-defined model configuration. User can also customize model configuration: specify `set_model_config_manually` to `1` and specify model configs manually, or specify `set_layernum_manually` to `1` and specify layer numbers manually only.

### Cluster Environment
Galvatron can perform training over multiple nodes with same number of GPUs. Users should set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK``` according to the environment.

### Parallelism Strategy

In distributed training with Galvatron, users can either train models with the optimal parallel strategy searched by the parallelism optimization to obtain the optimal throughput, or specify the hybrid parallel strategies as they like.

#### JSON Config Mode [Recommended]
JSON config mode is a **recommended** layerwise hybrid parallel training mode, activated by assigning argument `galvatron_config_path` with the config path in `configs` directory. In JSON config mode, users don't need be aware of the details of searched parallel strategies, and don't need to tune any parallel strategies or hyper-parameters. As the optimized parallel strategy will be saved in `configs` directory after parallelism optimization, users can run the optimal parallel strategies by simply assigning `galvatron_config_path` with the corresponding config path (e.g., assign ```galvatron_config_path``` as ```./configs/galvatron_config_xxx.json```).

#### GLOBAL Config Mode
GLOBAL config mode is a global hybrid parallel training mode, activated by assigning argument `galvatron_config_path` as `None`. In this mode, users can specify `pp_deg`, `global_tp_deg`, `global_tp_consec`, `sdp`, `global_train_batch_size`, `chunks`, `global_checkpoint`, `pipeline_type` to determine the global parallelism strategy, and all the layers of the Transformer model uses the same hybrid parallel strategy assigned by the users (just as in Megatron-LM).

### Arguments
1. JSON Config Mode
- galvatron_config_path: str, json config path, whether to activate JSON config mode. If activated, arguments in GLOBAL config mode will be ignored and overwritten by the JSON config.
2. GLOBAL Config Mode
- global_train_batch_size: Integer, global batch size of distributed training.
- pp_deg: Integer, pipeline (PP) degree,.
- global_tp_deg: Integer, tensor parallel (TP) degree.
- global_tp_consec: 0/1, whether the communication group of TP is consecutive, (eg., [0,1,2,3] is consecutive while [0,2,4,6] is not).
- sdp: 0/1, whether to use SDP instead of DP.
- chunks: Integer, number of microbatches of PP.
- global_checkpoint: 0/1, whether to turn on activation checkpointing to the whole model.
- pipeline_type: `gpipe` or `pipedream_flush`, choose the pipeline type to use.

### Other Training Optimizations
Set `mixed_precision` to allow mixed precision training, e.g., `bf16`. Set `use-flash-attn` to allow [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) features.

Please refer to function ```galvatron_training_args``` in [arguments.py](../core/arguments.py) for the full list of training arguments.