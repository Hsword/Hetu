#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
heturun -c hetu_config16ps.yml python ${mainpy} --model $1 --dataset $2 --learning-rate 0.000625 --validate --timing --comm-mode PS

