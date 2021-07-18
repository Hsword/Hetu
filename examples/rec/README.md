# Recommendation Model Example (with Distributed Settings)
In this directory we provide NCF model for recommendation task on movielens dataset.

## Structure
```
- rec
    - run_hetu.py           basic trainer for hetu
    - run_tf.py             basic trainer for tensorflow
    - run_tfworker.py       trainer for tensorflow in PS setting
    - run_parallax.py       trainer for tensorflow in parallax setting
    - hetu_ncf.py           model implementatino in hetu
    - tf_ncf.py             model implementation in tensorflow
    - movielens.py          script to download and handle dataset
```

## Prepare movielens data
Simply `python movielens.py` .

## Usage
```bash
# run locally
python run_hetu.py
# run in ps setting (locally)
bash ps_ncf.sh
# run in hybrid setting (locally)
bash hybrid_ncf.sh

# run tensorflow locally
python run_tf.py
# run tensorflow in parallax
python {absolute_path_to}/run_parallax.py
# run tensorflow in ps setting
python ../ctr/tf_launch_server.py --config {config} --id {rank}
python run_tfworker.py --rank {rank} --config {config}
# or
python ../ctr/tf_launch_server.py --config ../ctr/settings/tf_local_s1_w8.json --id 0
bash tf_8workers.sh
```


## Configuration
Please refer to `ctr` directory.
