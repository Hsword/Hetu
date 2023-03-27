# BigBird Example
In this directory we provide implementations for BigBird model on Hetu.

## Dataset Preparing
We support wikicorpus_en and bookcorpus dataset for BigBird pretraining, and support GLUE dataset for finetuning, including 
sst-2, cola, mrpc.

To download, verify, extract the datasets, create the shards in `.hdf5` format for pretraining, run:
```bash
sh ./scripts/create_datasets_from_start.sh
```
Bookcorpus server are usually overloaded, so downloading bookcorpus maybe time-consuming. If user wants to skip bookcorpus and only use wikicorpus_en for pretraining (recommended), run: 
```bash
sh ./scripts/create_datasets_from_start.sh wiki_only
```
To download both bookcorpus and wikicorpus_en dataset, user may have to repeatedly run the following scripts until the required number of files are downloaded:
```bash
sh ./scripts/create_datasets_from_start.sh wiki_books
```
Note: Ensure a complete Wikipedia download. If in any case, the download breaks, remove the output file wikicorpus_en.xml.bz2 and start again. If a partially downloaded file exists, the script assumes successful download which causes the extraction to fail.

Users are free to modify `./scripts/create_datasets_from_start.sh` to run specified scripts according to needs.

In order to finetune BigBird model by GLUE dataset, you may have to run the following scripts. 
```bash
python prepare_glue.py
```

Note: You can also use your own dataset, but you need to tokenize them firstly.

## Training BigBird model
All scripts for training BigBird model are in `./scripts` directory. Users are free to modify hyperparameters in shell file.

To train Hetu BigBird model, run:
```bash
sh ./scripts/train_hetu.sh     (1 GPU)
sh ./scripts/train_hetu_dp.sh  (4 GPUs)
```
To finetune Hetu BigBird model, run:
```bash
sh ./scripts/finetune_hetu.sh     (1 GPU)
```
