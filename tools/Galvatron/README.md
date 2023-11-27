# Galvatron
Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism.

A new system framework that incorporates multiple popular parallelism dimensions and automatically finds the most efficient hybrid parallelism strategy.

## Environment Setup
Run the following script to prepare the conda environment:
``` shell
conda create -n galvatron python=3.8
conda activate galvatron
cd Galvatron
sh prepare_env.sh
```

## Galvatron System Architecture
Galvatron is consisted of four modules, including an automatic profiler, a strategy cost estimator, parallelism optimization block, and Galvatron runtime framework. To train Transformer models over multiple GPUs using automatic parallelism with Galvatron, users only need to provide with hardware environment and the Transformer model configuration.

<div align=center> <img src="./figs/api.jpg" width="800" /> </div>

## Usage

### Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model forward computation time. Galvatron will automatically write the tested results into config files.

(1) Firstly, profile the hardward environment:
- For 8 GPUs on 1 node, run the following scripts:
``` shell
cd ./test_env
sh profile_env_8gpus.sh
```

- For 16 GPUs on 2 nodes, first export the right ```MASTER_ADDR```, ```NCCL_SOCKET_IFNAME``` and ```NODE_RANK``` into environment for multi-node test, and then run the following scripts on both nodes:
``` shell
cd ./test_env
sh profile_env_16gpus.sh
```

(2) Secondly, profile the forward computation time of the chosen Transformer model.
``` shell
cd ./model_name
sh scripts/forward_profiling.sh
```

### Optimizing with Galvatron
After profiling the environments, Galvatron is able to automatically optimize the parallelism strategy for the given Transformer model. Currently, Galvatron support parallelism optimization on BERT, T5, ViT, and Swin Transformer. Given the memory budget, Galvatron provides the layerwise hybrid parallel strategy with maximum throughput. Note that the strategy cost estimator block is executed automatically before parallelism optimization. 

The optimized parallelism strategy will be saved in `./model_name/configs` for the training. Users can train the model with the provided optimal strategy to obtain the optimal throughput. 

- For 8 GPUs on 1 node, run the following scripts to conduct parallelim optimization:
``` shell
cd ./model_name
sh scripts/search_layerwise_hp_8gpus.sh
```
- For 16 GPUs on 2 nodes, run the following scripts to conduct parallelim optimization:
``` shell
cd ./model_name
sh scripts/search_layerwise_hp_dist_16gpus.sh
```

The Transformer model configuration can be assigned in the scripts above by modifying only a few arguments. Enter the corresponding directory of the model to see detailed guidance.

### Training with Galvatron

Galvatron provides a simple way to train Transformer models in layerwise hybrid parallel fashion. We provide four models as examples in this repo, including BERT, T5, ViT, Swin Transformer, and provide scripts to train these models on single GPU or on multiple GPUs using hybrid parallel strategy. 

Users can either train Transformer models with the optimal parallel strategy searched by the parallelsim optimization block to obtain the optimal throughput, or use any hybrid parallel strategy as they like. Galvatron support three hybrid parallel config modes, including JSON config mode, PYTHON config mode, and GLOBAL config mode. Users can specify hybrid parallel strategies by modifying only a few arguments or a few lines of codes. Enter the corresponding directory of the model to see detailed guidance.

It is fairly simple to construct a layerwise hybrid parallel model in Galvatron, which only requires a few modification to the training scripts on single GPU.

In the model directory ```bert, t5, vit, swin```, to train the model on single GPU, run:
``` shell
sh scripts/train_xxx.sh
```
To train the model on 8 GPUs on 1 node, run:
``` shell 
sh scripts/train_xxx_hp_layerwise.sh
```
To train the model on 16 GPUs on 2 nodes, first export the right ```MASTER_ADDR```, ```NCCL_SOCKET_IFNAME``` and ```NODE_RANK``` into the environment for multi-node test, and then run:
``` shell 
sh scripts/train_xxx_hp_layerwise_dist_2nodes.sh
```