#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../torch_main.py

## validate and timing
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --validate --timing
