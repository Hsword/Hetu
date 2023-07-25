#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../tf_main.py

### validate and timing
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --validate --timing

### run in cpu
# python ${mainpy} --model tf_mlp --gpu -1 --validate --timing
