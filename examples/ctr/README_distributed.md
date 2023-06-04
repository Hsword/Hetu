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
* config: to specify the configuration file in settings.
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
We use a simple yaml file to specify the run configuration.

```yaml
shared :
    DMLC_PS_ROOT_URI : 127.0.0.1
    DMLC_PS_ROOT_PORT : 13100
    DMLC_NUM_WORKER : 4
    DMLC_NUM_SERVER : 1
launch :
    worker : 4
    server : 1
    scheduler : true
```

The 4 k-v pair in "shared" are used for PS-lite parameter server and will be added into environment. When running on a cluster, you should change "DMLC_PS_ROOT_URI" into an available IP address in the cluster.

The following "launch" is only used in PS-mode (ommitted in hybrid mode). This means that the number of worker, server and scheduler launched locally on this machine. In hybrid mode, workers are launched by mpirun. Servers and schedulers will be launched by


## Examples
### Local execution
Run wdl with criteo locally(if the whole dataset is downloaded, you can use all data or use validate data):
```bash
python run_hetu.py --model wdl_criteo (--all) (--val)
```

### PS mode execution
Run ps locally, here we can also run on multiple nodes.
```bash
# launch scheduler and server, -n means number of servers, --sched means using scheduler
python -m hetu.launcher {config} -n 1 --sched
# launch workers (or run scheduler and server together if configured in config file)
python run_hetu.py --comm PS --model wdl_criteo --config {config} (--all) (--val) (--cache lfuopt) (--bound 10)
```
You can also specify the cache to be used and also the cache bound.


### Hybrid mode execution
You must launch a scheduler and server in one terminal:
```bash
python -m hetu.launcher {config} -n 1 --sched
```
And then launch the workers simultaneously using mpirun command:
```bash
mpirun -np {num_worker} --allow-run-as-root python run_hetu.py --comm Hybrid ...
```
Or if in distributed nodes setting:
```
mpirun -mca btl_tcp_if_include (network card name or ip) -x NCCL_SOCKET_IFNAME=(network card name) --host (host ips) --allow-run-as-root python run_hetu.py --comm Hybrid ...
```
