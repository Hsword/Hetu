# CTR Examples (with Distributed Settings)
In this directory we provide several models for CTR tasks. We use Wide & Deep model to train on Adult and Criteo dataset, and DeepFM, DCN, DC models on Criteo dataset.

## Structure
```
- ctr
    - datasets/             contains sampled criteo data
    - models/               ctr models in hetu
    - tf_models/            ctr models in tensorflow
    - settings/             configurations for distributed training
    - tests/                test scripts
    - kill.sh               script to kill all python processes
    - run_hetu.py           basic trainer for hetu
    - run_tf_local.py       local trainer for tensorflow
    - run_tf_horovod.py     trainer for tensorflow in horovod setting
    - run_tf_parallax.py    trainer for tensorflow in parallax setting
    - tf_launch_server.py   launcher for server in tensorflow
    - tf_launch_worker.py   launcher for worker in tensorflow
```

## Prepare criteo data
* We have provided a sampled version of kaggle-criteo dataset, which locates in ./datasets/criteo/ . To use the given data, please do not specify the 'all' flag and 'val' flag when running test files.
* To download the original kaggle-criteo dataset, please specify a source in models/load_data.py and use ```python models/load_data.py``` to download the whole kaggle-criteo dataset.


## Flags for test files
Here we explain some of the flags you may use in test files:
* model: to specify the model, candidates are ('wdl_criteo', 'dfm_criteo', 'dcn_criteo', 'wdl_adult')
* val: whether using validation.
* cache: whether using cache in PS/Hybrid mode.
* bsp: whether using bsp (default asp) in PS/Hybrid mode. (In Hybrid, AllReduce can enforce dense parameters to use bsp, so there will be no stragglers.) bsp 0, asp -1, ssp > 0
* all: whether to use all criteo data.
* bound: per embedding entry staleness in cache setting, default to be 100.


## Usage
If memory available, you can try to run the model locally, by running
```bash
# run locally
bash tests/local_{model}_{dataset}.sh
# run in ps setting (locally)
bash tests/ps_{model}_{dataset}.sh
# run in hybrid setting (locally)
bash tests/hybrid_{model}_{dataset}.sh

# run tensorflow locally
python run_tf_local.py --model {model}_{dataset}
# run tensorflow in horovod
horovodrun -np 8 -H localhost:8 python run_tf_horovod.py --model {model}_{dataset}
# run tensorflow in parallax
python {absolute_path_to}/run_tf_parallax.py
# run tensorflow in ps setting
python tf_launch_server.py --config {config} --id {rank}
python tf_launch_worker.py --model {model}_{dataset} --rank {rank} --config {config}
```


## Configuration
We use a simple yaml file to specify the heturun configuration.

```yaml
nodes:
  - host: hostname1
    servers: 1 
    workers: 2
    chief: true
  - host: hostname2
    servers: 1
    workers: 2
    chief: false
```

Users only need to specify the hostname and the number of servers and workers on each host (machine). The hostname can be found by `socket.gethostname()` in Python. The number of servers determines the partitions of parameters; and the number of workers `n_workers` determines the GPUs to be used, the first `n_workers`  GPUs each bounded with one worker. Whether using PS or AllReduce should be specified in arguments of scripts, not in configuration files. One should set the number of servers larger than 0 in PS or Hybrid mode, while the number of servers equal to 0 in AllReduce mode.

If running on single GPU, `heturun` command is useless; if running on multiple GPUs on the same host (machine), one can replace the configuration file with command `heturun -s {n_server} -w {n_worker}`, or simplify the hostname in configuration file using `localhost`; if running on multiple GPUs on multiple hosts (machines), a configuration file must be used, where the `chief` host launches the script.


## Examples
### Local execution
Run wdl with criteo locally(if the whole dataset is downloaded, you can use all data or use validate data):
```bash
python run_hetu.py --model wdl_criteo (--all) (--val)
```

### PS mode execution
Run ps locally with multiple GPUs, here we can also run on multiple nodes.
```bash
heturun -s 1 -w 2 python run_hetu.py --comm PS --model wdl_criteo (--all) (--val) (--cache lfuopt) (--bound 10)
```
You can also specify the cache to be used and also the cache bound.


### Hybrid mode execution
```bash
heturun -s 1 -w 2 python run_hetu.py --comm Hybrid --model wdl_criteo (--all) (--val) (--cache lfuopt) (--bound 10)
```
Or if in distributed nodes setting:
```
heturun -c config.yml python run_hetu.py --comm Hybrid --model wdl_criteo (--all) (--val) (--cache lfuopt) (--bound 10)
```
