# MAE Example
In this directory we provide implementations for MAE model on Hetu.

## Dataset Preparing
You need to prepare ImageNet-1K dataset at first, which can be downloaded from: https://www.image-net.org/download.php.

## Training MAE model
All scripts for training MAE model are in `./scripts` directory. Users are free to modify hyperparameters in shell file.

To train Hetu MAE model, run:
```bash
sh ./scripts/train_hetu.sh     (1 GPU)
sh ./scripts/train_hetu_dp.sh  (4 GPUs)
```
