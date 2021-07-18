# NLP Examples
In this directory we provide simple implementations for Transformer model. We use the IWSLT2016 de-en dataset. 
## Structure
```
- nlp
    - hparams.py                    Hyperparameters
    - prepare_data.py               Downloading and preparing data
    - data_load.py                  Dataloader
    - hetu_transformer.py           Transformer model in hetu
    - tf_transformer.py             Transformer model in tensorflow
    - train_hetu_transformer.py     Trainer for hetu
    - train_tf_transformer.py       Trainer for tensorflow
```
## Usage
```bash
python train_{framework}_transformer.py
```
To change the hyperparameters, please modify `hparams.py` file.