#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../main.py
depsdir=${workdir}/../../..

### validate and timing
heturun -w 2 python ${mainpy} --learning-rate 0.00125 --validate --timing --comm-mode AllReduce
