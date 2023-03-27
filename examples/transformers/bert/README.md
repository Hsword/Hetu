# BERT Example
In this directory we provide implementations for BERT model on PyTorch and Hetu.

## Dataset Preparing
We support wikicorpus_en and bookcorpus dataset for BERT pretraining, and support GLUE dataset for finetuning, including 
sst-2, cola, mrpc. Data downloading and preprocessing scripts are based on https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT. 

To download, verify, extract the datasets, create the shards in `.hdf5` format for pretraining, and preprocess all GLUE data for finetuing, run:
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

## Pretraining and Finetuning BERT model
All scripts for pretraining or finetuning BERT model are in `./scripts` directory. Users are free to modify hyperparameters in shell file.

To pretrain Hetu BERT base or BERT large model, run:
```bash
sh ./scripts/train_hetu_bert_base.sh
sh ./scripts/train_hetu_bert_large.sh
```
To pretrain Hetu BERT base or BERT large model using data parallel distributedly, run:
```bash
sh ./scripts/train_hetu_bert_base_dp.sh
sh ./scripts/train_hetu_bert_large_dp.sh
```
To pretrain Hetu BERT base or BERT large model using PS, run:
```bash
sh ./scripts/train_hetu_bert_base_ps.sh
sh ./scripts/train_hetu_bert_large_ps.sh
```
To pretrain Pytorch BERT base or BERT large model, run:
```bash
sh ./scripts/train_pytorch_bert_base.sh
sh ./scripts/train_pytorch_bert_large.sh
```
To finetune Hetu BERT base or BERT large model on GLUE task, run:
```bash
sh ./scripts/test_glue_hetu_bert_base.sh
sh ./scripts/test_glue_hetu_bert_large.sh
```
To finetune Pytorch BERT base or BERT large model on GLUE task, run:
```bash
sh ./scripts/test_glue_pytorch_bert_base.sh
sh ./scripts/test_glue_pytorch_bert_large.sh
```