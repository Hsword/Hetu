#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py


# model: 
# e.g. bash hetu_1gpu.sh

### validate and timing
python ${mainpy} --learning-rate 0.01 --validate --timing
