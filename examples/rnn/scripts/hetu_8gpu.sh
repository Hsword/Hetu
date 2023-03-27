#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py
depsdir=${workdir}/../../..

### validate and timing
heturun -w 8 python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --validate --timing --comm-mode AllReduce
