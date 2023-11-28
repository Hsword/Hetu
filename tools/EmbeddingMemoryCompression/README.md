# (VLDB'24) Experimental Analysis of Large-scale Learnable Vector Storage Compression

This directory is used for VLDB'24 Artifact Evaluation of paper #282, titled "Experimental Analysis of Large-scale Learnable Vector Storage Compression".

## Setup

Our codes are based on Hetu. We provide a cmake file `config.cmake` in this directory; please copy it to `../../cmake` directory and compile Hetu according to the [README](../../README.md#installation) file in the home page of Hetu. After compiling, you can train or infer with `run_compressed.py` in this directory.

## Prepare Datasets
Currently we support Avazu and Criteo datasets.

  - Avazu is provided by Avazu, a mobile advertising platform, at the 2015 ctr prediction competition on Kaggle. It contains 10 days of real data, 40M records in total. There are 22 feature fields for a total of 9.4 million unique features.Avazu can be downloaded manually from https://www.kaggle.com/c/avazu-ctr-prediction/data.

  - Criteo is part of the 7-day real data provided by the Criteo Lab on Kaggle in 2014. with a total of 46M records. It contains 13 continuous features and 26 feature field with a total of 34M unique features. Criteo is also adopted in MLPerf, a standard benchmark for machine learning performance. Criteo can be downloaded manually from 
  https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz.

  - Statistics of the datasets

  | Datasets | Fields | Features | Samples | 
  | :---: | :---: | :---: | :---: |
  | Avazu | 22 | 9,449,445 | 40,428,967 |
  | Criteo | 26 | 33,762,577 | 45,840,617 |


- You can place the downloaded datasets into a new folder `datasets` under current directory, and check the script `models/load_data.py` for preprocessing.

## Training
```
usage: run_compressed.py [--model MODEL] [--method METHOD] [--phase PHASE]
               [--ectx ECTX] [--ctx CTX] [--bs BS] [--opt OPT] [--dim DIM] [--lr LR]
               [--dataset DATASET] [--data_path DATA_PATH] [--nepoch NEPOCH] 
               [--num_test_every_epoch NUM_TEST_EVERY_EPOCH] [--seed SEED] 
               [--separate_fields --SEPERATE_FIELDS] [--use_multi USE_MULTI]
               [--compress_rate COMPRESS_RATE] [--logger LOGGER] [--run_id RUN_ID]
               [--debug DEBUG] [--threshold THRESHOLD] [--stage STAGE] [--load_ckpt LOAD_CKPTS]
               [--log_dir LOG_DIR] [--save_dir SAVE_DIR] [--save_topk SAVE_TOPK] 
               [--check_val CHECK_VAL] [--check_test CHECK_TEST] 
               [--early_stop_steps EARLY_STOP_STEPS]
               
optional arguments:
  --model               Model to be tested
  --method              Method to be used
  --phase               Train or Test
  --ectx                Context for embedding table
  --ctx                 Context for model
  --bs                  Batch size to be used
  --opt                 Optimizer to be used
  --dim                 Dimension to be used
  --lr                  Learning rate to be used
  --dataset             Dataset to be used 
  --data_path           Path to dataset
  --nepoch              Num of epochs 
  --num_test_every_epoch
                        Evaluate each 1/100 epoch in default(default:100)
  --seed                Random seed
  --separate_fields     Whether seperate fields
  --use_multi           Whether use multi embedding
  --compress_rate       Compress rate to be used
  --logger              Logger to be used
  --run_id              Run id to be logged
  --debug               Whether in debug mode
  --threshold           The threshold for inter-feature compression
  --stage               The start stage for train/test
  --load_ckpt           CheckPoints to be used
  --log_dir             Directory for logging
  --save_dir            Directory for saving
  --save_topk           Number of ckpts to be saved
  --check_val           Whether check validation data during training
  --check_test          Whether check test data during training
  --early_stop_steps    Early stopping if no improvement over steps
```
An example of training usage is shown as follows:
```
python run_compressed.py --model dlrm --method compo --compress_rate 0.5 --ctx 0 --nepoch 10 --early_stop_steps 40 --opt adam --dim 16 --lr 0.001 
```

## Inference
```
usage: run_compressed.py [-model MODEL] [--ctx CTX] [--method METHOD] [--dataset DATASET]
               [--opt OPT] [--load_ckpt CHECKPOINTS] [--compress_rate CR] [--phase test]

```
An example of inference usage is shown as follows:
```
python run_compressed.py --model dlrm --method compo --compress_rate 0.5 --ctx 0 --phase test
--load_ckpt /path/to/ckpt
```



## Implementation

* Layers of different embedding compression methods are implemented in [./methods/layers](./methods/layers/). It is possible to implement custom compression methods inheriting `Embedding` class.
* Trainers are located in [./methods/scheduler](./methods/scheduler/). New trainers must be implemented for custom compression methods.
* Related gpu operators are implemented into the Hetu system in [Hetu/python/hetu/gpu_ops](../../python/hetu/gpu_ops/).


## Configurations of Embedding Compression Methods
- CompoEmb: we multiple the embeddings from different tables as in the original paper; we only compress the fields that achieve less memory after compression.
- TT-Rec: we decompose the tensor into three small tensors, and vary the rank to adapt to different memory budgets; we only compress the fields that achieve less memory after compression.
- DHE: we use 1000000 buckets and 1024 hash functions as in the original paper, and we use uniform distribution for the input of MLP; we only compress the fields that achieve less memory after compression.
- ROBE: we use Z=1 so that the embedding is not split.
- Dedup: we use the same LSH configurations as in the original paper, and we adjust the block capacity, similariy threshold, fp threshold based on actual memory budget.
- MGQE: we provide 256 choices for high-frequency features and 64 choices for low-frequency features as in the original paper; we vary number of split parts to adapt to different memory budgets; we use top_percent=0.1, which means the top 10% features are considered high-frequency features.
- AdaptEmb: we use 'top_percent' to denote the ratio of high-frequency features; for different memory budgets, we use different top percent.
- INT8/16: 'digit' can be 8 or 16 for INT8 and INT16 respectively; 'middle' is set to zero to enforce symmetry; 'scale' is tested among several values as in the original paper.
- ALPT: 'digit' can be 8 or 16 for INT8 and INT16 respectively; 'init_scale' is the initial value for scaling; 'scale_lr' is the learning rate for scale, which we set 2e-5.
- AutoDim: we test several configurations, whether ignore the second order term in the gradient of alpha or whether retrain from pre-trained parameters; we set alpha_lr=0.001 as in the original paper.
- OptEmbed: we set the hyperparameters according to the original paper, including 'alpha', 'thresh_lr', 'keep_num', 'mutation_num', 'crossover_num', 'm_prob', 'nepoch_search'.
- DeepLight: we set stop_deviation=1e-3 to ensure the deviation between final pruning ratio and target ratio is smaller than 1e-3.
- AutoSrh: we set nsplit=6, alpha_l1=0.00001 and alpha_lr=0.001.

## Citation
If you find this work useful, please cite [our paper](https://arxiv.org/abs/2311.15578).

