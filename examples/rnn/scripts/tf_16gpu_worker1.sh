#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../tf_launch_worker.py

python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 8 --gpu 0 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 9 --gpu 1 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 10 --gpu 2 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 11 --gpu 3 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 12 --gpu 4 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 13 --gpu 5 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 14 --gpu 6 --timing --validate &
python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --config ${workdir}/../settings/tf_dist_s1_w16.json --rank 15 --gpu 7 --timing --validate &
wait