#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../tf_launch_worker.py

python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 0 --gpu 0 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 1 --gpu 1 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 2 --gpu 2 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 3 --gpu 3 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 4 --gpu 4 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 5 --gpu 5 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 6 --gpu 6 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --config ${workdir}/../settings/tf_dist_s1_w8.json --rank 7 --gpu 7 --timing --validate &
wait

