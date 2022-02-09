#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

heturun -w 1 python ${mainpy} --model wdl_adult --val
