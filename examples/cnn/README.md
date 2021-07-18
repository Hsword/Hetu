# CNN Examples
In this directory we provide simple implementations for CNN models, including both hetu and tensorflow versions for comparison.
## Structure
```
- cnn
    - models/               CNN models in HETU
    - pytorch_models/       CNN models in PyTorch
    - tf_models/            CNN models in TensorFlow
    - scripts/              Test scripts
    - main.py               Trainer for HETU
    - run_tf_horovod.py     Trainer for Horovod
    - tf_launch_server.py   Trainer for TF-PS (role: server)
    - tf_launch_worker.py   Trainer for TF-PS (role: worker)
    - tf_main.py            Trainer for TensorFlow
    - torch_main.py         Trainer for Pytorch
    - 
```
## Usage
Here are some examples of running scripts.
```bash
bash scripts/hetu_1gpu.sh mlp CIFAR10   # mlp with CIFAR10 dataset in hetu
bash scripts/hetu_8gpu.sh mlp CIFAR10   # mlp with CIFAR10 in hetu with 8-GPU (1-node)
bash scripts/hetu_16gpu.sh mlp CIFAR10  # mlp with CIFAR10 in hetu with 8-GPU (2-nodes)            
```
To train in PS setting, we also need to launch scheduler and server first. For more information about distributed training, please refer to CTR or GNN examples.

We can change the setting in scripts. See `mnist_mlp.sh` below.
```bash
#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
python ${mainpy} --model mlp --dataset CIFAR10  --validate --timing

### run in cpu
# python ${mainpy} --model mlp --dataset CIFAR10 --gpu -1 --validate --timing

```

For more details about training setting, please refer to `main.py`.
## Models
We provide following models with specific datasets.
```
CIFAR100: VGG, ResNet
CIFAR10: MLP, VGG, ResNet
MNIST: AlexNet, CNN(3-layer), LeNet, LogisticRegression, LSTM, RNN
```
