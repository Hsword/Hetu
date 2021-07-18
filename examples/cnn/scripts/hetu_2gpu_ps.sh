#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
python -m hetu.launcher ${workdir}/../local_s1.yml -n 1 --sched &
python ${mainpy} --model $1 --dataset $2 --validate --timing --comm-mode PS --gpu 0 &
python ${mainpy} --model $1 --dataset $2 --validate --timing --comm-mode PS --gpu 1 &
wait