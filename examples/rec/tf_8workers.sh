#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/run_tfworker.py

CUDA_VISIBLE_DEVICES=0 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 0 &
CUDA_VISIBLE_DEVICES=1 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 1 &
CUDA_VISIBLE_DEVICES=2 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 2 &
CUDA_VISIBLE_DEVICES=3 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 3 &
CUDA_VISIBLE_DEVICES=4 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 4 &
CUDA_VISIBLE_DEVICES=5 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 5 &
CUDA_VISIBLE_DEVICES=6 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 6 &
CUDA_VISIBLE_DEVICES=7 python ${mainpy} --config ${workdir}/../ctr/settings/tf_local_s1_w8.json --rank 7 &
wait