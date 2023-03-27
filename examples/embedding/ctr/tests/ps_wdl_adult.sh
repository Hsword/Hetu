#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

heturun -s 1 -w 4 python ${mainpy} --model wdl_adult --val --comm PS --cache lfuopt --bound 3
