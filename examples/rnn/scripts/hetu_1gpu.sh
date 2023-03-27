#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py


# model: 
# e.g. bash hetu_1gpu.sh mlp CIFAR10

### validate and timing
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --validate --timing
