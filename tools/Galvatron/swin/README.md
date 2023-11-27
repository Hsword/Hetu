# Swin Transformer

This directory contains scripts to search and train Swin Transformer model using Galvatron.

## Data preparation
We use standard ImageNet 1K dataset, which you can download from http://image-net.org/. Please use zipped ImageNet to speedup data reading. The ImageNet folder should be placed in this directory and look like this:

```
- ImageNet
    - train_map.txt
    - train.zip
    - val_map.txt
    - val.zip
```

## Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model forward computation time.

(1) Firstly, profile the hardward environment. Please refer to the previous directory for details. Make sure that the hardward environment is already profiled before running any script in this directory!

(2) Secondly, profile the forward computation time of Swin model.
``` shell
sh scripts/forward_profiling.sh
```

## Optimizing with Galvatron
After profiling the environments, Galvatron is able to automatically optimize the parallelism strategy for the given Transformer model. Given the memory budget, Galvatron provides the layerwise hybrid parallel strategy with maximum throughput.

The optimized parallelism strategy will be saved in `./configs` for the training. Users can train the model with the provided optimal strategy to obtain the optimal throughput. 

- For 8 GPUs on 1 node, run the following scripts to conduct parallelim optimization:
``` shell
sh scripts/search_layerwise_hp_8gpus.sh
```
- For 16 GPUs on 2 nodes, run the following scripts to conduct parallelim optimization:
``` shell
sh scripts/search_layerwise_hp_dist_16gpus.sh
```

### Arguments
The Transformer model configuration can be assigned in the scripts above by assigning the arguments below:
- memory_constraint: Integer, memory budget (GB) of each GPU.
- layer_nums: List of integer, number of each type of the Swin layers.
- embed_dim: Integer, embedding dimension of the Swin model. Currently, we support embed_dim=320 for Swin.
- type: str, Galvatron parallelism optimization type, default is 'full', choices include 'full','dp+tp','dp+pp'.


## Training with Galvatron
- To train Swin model on a single GPU, run the following scripts:
``` shell
sh scripts/train_swin_huge.sh
```
- To train Swin model on 8 GPUs on 1 node, make sure that ```nproc_per_node=8//pp_deg```, and run the following scripts:
``` shell 
sh scripts/train_swin_huge_hp_layerwise.sh
```

- To train Swin model on 16 GPUs on 2 nodes, first export the right ```MASTER_ADDR```, ```NCCL_SOCKET_IFNAME``` and ```NODE_RANK``` into the environment for multi-node training, make sure ```nproc_per_node=8```, and then run:
``` shell 
sh scripts/train_swin_huge_hp_layerwise_dist_2nodes.sh
```

### Hybrid Parallel Config Mode

In distributed training, users can either train Transformer models with the optimal parallel strategy searched by the parallelism optimization block to obtain the optimal throughput, or train the model using any combination of data parallel (DP), tensor parallel (TP), pipeline parallel (PP) and sharded data parallel (SDP) as they like. These strategies can be applied either globally or layerwisely.

Galvatron support three hybrid parallel config modes, including **JSON config mode**, **PYTHON config mode**, and **GLOBAL config mode**. Users can specify one of these modes to apply any parallel strategy by modifying the arguments of ```scripts/train_xxx_hp_layerwise.sh``` or ```scripts/train_xxx_hp_layerwise_dist_2nodes.sh```.

#### JSON Config Mode [Recommended]
JSON config mode is a **recommended** layerwise hybrid parallel training mode, activated by assigning argument `galvatron_config_path` with the config path in `configs` directory. In JSON config mode, users don't need be aware of the details of searched parallel strategies, and don't need to tune any parallel strategies or hyper-parameters. As the optimized parallel strategy will be saved in `configs` directory after parallel optimization, users can run the optimal parallel strategies by simply assigning `galvatron_config_path` with the corresponding config path. 

For example, assign ```galvatron_config_path=./configs/galvatron_config_8gpus_320embeddim_2_2_42_2_layers_example.json``` and Galvatron will run this parallel strategy automatically. 

Note that when training on 8GPUs on 1 node using ```scripts/train_xxx_hp_layerwise.sh```, user still needs to make sure that ```nproc_per_node=8//pp_deg```.

#### GLOBAL Config Mode
GLOBAL config mode is a global hybrid parallel training mode, activated by assigning argument `galvatron_config_path=None` and `apply_strategy=0` and modifying ```pp_deg, global_tp_deg, global_tp_consec, fsdp```.
```pp_deg``` refers to PP degree, ```global_tp_deg``` refers to TP degree, ```global_tp_consec``` refers to whether the TP communication group is consecutive (eg., [0,1,2,3] is consecutive while [0,2,4,6] is not). If TP is divided before DP and SDP on the decision tree, ```global_tp_consec=0```, else ```global_tp_consec=1```. ```fsdp``` refers to whether to use SDP instead of DP.

In this global parallel training mode, all the layers of the Transformer model uses the same hybrid parallel strategy assigned by the users.

Here are several examples, and the strategies are given following the top-to-bottom order on the decision tree: 

- Strategy 1: 2-way PP, 2-way DP, 2-way TP, on 8 GPUs on 1 node

    ```nproc_per_node=4, pp_deg=2, global_tp_deg=2, global_tp_consec=1, fsdp=0```

- Strategy 2: 4-way PP, 2-way SDP, 1-way TP, on 8 GPUs on 1 node

    ```nproc_per_node=2, pp_deg=4, global_tp_deg=1, global_tp_consec=1, fsdp=1```

- Strategy 3: 2-way PP, 2-way TP, 4-way SDP, on 16 GPUs on 2 nodes

    ```nproc_per_node=8, pp_deg=2, global_tp_deg=2, global_tp_consec=0, fsdp=1```

- Strategy 4: 1-way PP, 2-way TP, 8-way SDP, on 16 GPUs on 2 nodes

    ```nproc_per_node=8, pp_deg=1, global_tp_deg=2, global_tp_consec=0, fsdp=1```

Note that when training on 8 GPUs on 1 node using ```scripts/train_xxx_hp_layerwise.sh```, user needs to make sure that ```nproc_per_node=8//pp_deg```, and when training on 16 GPUs on 2 nodes using ```scripts/train_xxx_hp_layerwise_dist_2nodes.sh```, user needs to make sure that ```nproc_per_node=8```.


#### PYTHON Config Mode 
PYTHON config mode is an advanced layerwise hybrid parallel training mode, activated by assigning argument `galvatron_config_path=None` and `apply_strategy=1`. PYTHON config mode allows users to manually assign layerwise parallel strategies through python scripts. This mode is NOT recommended for most users and only suitable for users with advanced needs to carefully tune the hybrid parallel strategies. 

To be specific, 
- When training on 8 GPUs on 1 node using ```scripts/train_xxx_hp_layerwise.sh```, users can design parallel strategies in function ```apply_strategy()``` in ```hybrid_parallel_model.py```. Make sure that ```nproc_per_node=8//pp_deg```.
- When training on 16 GPUs on 2 nodes using ```scripts/train_xxx_hp_layerwise_dist_2nodes.sh```, users can design parallel strategies in function ```apply_strategy()``` in ```hybrid_parallel_model_dist.py```. Make sure that ```nproc_per_node=8```.

A complete layerwise hybrid parallel strategy for Swin includes:
- pp_deg: Integer, pipeline (PP) degree.
- tp_sizes_enc: List of integer, TP degree of each layer, length equal to number of hidden layers.
- tp_consecutive_flags: List of integer, whether use consecutive communication group for TP, length equal to number of hidden layers.
- dp_types_enc: List of integer, whether to use SDP for each layer, length equal to number of hidden layers.
- global_bsz: Integer, global training batch size.
- chunks: Integer, number of microbatches of PP.
- pp_division: List of integer, denote number of layers on each PP stage, length equal to pp_deg.

A sample layerwise hybrid parallel strategy for Swin-Huge-48 is given in ```hybrid_parallel_model.py```: 

For layer 0-1 and layer 6-43, apply strategy 2-way PP, 4-way DP;
For layer 2-5, apply strategy 2-way PP, 4-way SDP;
For layer 44-45, apply strategy 2-way PP, 2-way DP, 2-way TP;
For layer 46-47, apply strategy 2-way PP, 4-way TP.

``` python
### Example strategy
pp_deg = 2
tp_sizes_enc = [1]*44+[2]*2+[4]*2
tp_consecutive_flags = [1]*48
dp_types_enc = [0]*2+[1]*4+[0]*42
global_bsz = 64
chunks = 2
pp_division = [19, 29]
```



### Arguments
- global_train_batch_size: Integer, global batch size of distributed training.
- embed_dim: Integer, embedding dimension of Swin model.
- depths: List of integer, number of each type of the Swin layers.
- profile: 0/1, whether to profile memory overhead and execution time.
- check_loss: 0/1, whether to print loss during training.
- pp_deg: Integer, pipeline (PP) degree, only used in GLOBAL config mode.
- global_tp_deg: Integer, tensor parallel (TP) degree, only used in GLOBAL config mode.
- global_tp_consec: 0/1, whether the communication group of TP is consecutive, only used in GLOBAL config mode.
- fsdp: 0/1, whether to use SDP instead of DP, only used in GLOBAL config mode.
- chunks: Integer, number of microbatches of PP, only used in GLOBAL config mode. If assigned -1, Galvatron will use the recommended chunks.
- apply_strategy: 0/1, whether to turn on PYTHON config mode, only used in PYTHON config mode.
- galvatron_config_path: str, json config path, whether to turn on JSON config mode, only used in JSON config mode.

Note that `global_train_batch_size`, `chunks`, `num_hidden_layers` will be overwritten in JSON config mode and PYTHON config mode by the given parallel strategy.


