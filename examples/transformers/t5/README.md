# T5 Example
In this directory we provide implementations for T5 model on Hetu.

## Dataset Preparing
We support Shakespeare and OpenWebText dataset for T5 training.

Shakespeare is a small dataset on which you can conduct simple model training. You may have to run the following scripts to prepare Shakespeare dataset.
```bash
python prepare.py
```

We also provide OpenWebText dataset for language model training. You can set the dataset to openwebtext in the command line and adjust the number of processes for data preparation.
```bash
python prepare.py --dataset openwebtext --num_proc 4
```

Note: You can also use your own dataset, but you need to tokenize them firstly.

## Training T5 model
All scripts for training T5 model are in `./scripts` directory. Users are free to modify hyperparameters in shell file.

To train Hetu T5 model, run:
```bash
sh ./scripts/train_hetu.sh     (1 GPU)
sh ./scripts/train_hetu_dp.sh  (4 GPUs)
```

