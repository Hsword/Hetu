#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py
depsdir=${workdir}/../../..

### validate and timing
# 
NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 8 -x PYTHONPATH=/home/public/third_party_tests/Athena/python /root/anaconda3/envs/zhl/bin/python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --validate --timing --comm-mode AllReduce
