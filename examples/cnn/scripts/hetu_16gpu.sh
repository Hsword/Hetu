#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py
depsdir=${workdir}/../../..
echo $depsdir
### validate and timing
$depsdir/build/_deps/openmpi-build/bin/mpirun --allow-run-as-root -np 16 -mca btl_tcp_if_include enp97s0f0 -x NCCL_SOCKET_IFNAME=enp97s0f0 -x PYTHONPATH=$depsdir/python -H daim117:8,daim118:8 /root/anaconda3/envs/zhl/bin/python ${mainpy} --model $1 --dataset $2 --learning-rate 0.000625 --validate --timing --comm-mode AllReduce

