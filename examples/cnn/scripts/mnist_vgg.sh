#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
python ${mainpy} --model vgg16 --dataset CIFAR10  --validate --timing

### run in cpu
# python ${mainpy} --model vgg16 --dataset CIFAR10 --gpu -1 --validate --timing
