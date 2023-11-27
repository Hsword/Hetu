# Environment Profiling
This directory contains scripts to profile the environment configs, including allreduce bandwidth, P2P bandwidth, overlap coefficient.

## Usage
- For 8 GPUs on 1 node, run the following scripts to profile the environment:
``` shell
sh profile_env_8gpus.sh
```
- For 16 GPUs on 2 nodes, first export the right ```MASTER_ADDR```, ```NCCL_SOCKET_IFNAME``` and ```NODE_RANK``` into environment for multi-node test, and then run the following scripts on both nodes:
``` shell
sh profile_env_16gpus.sh
```

Galvatron will automatically write the tested results into config files.