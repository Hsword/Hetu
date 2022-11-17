#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

### validate and timing
heturun -w 8 python ${mainpy} --model wdl_adult --nepoch 10 --comm allreduce
