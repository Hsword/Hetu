#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py

### validate and timing
heturun -s 1 -w 2 python ${mainpy} --model $1 --dataset $2 --validate --timing --comm-mode PS
