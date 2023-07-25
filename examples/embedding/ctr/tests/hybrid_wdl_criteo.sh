#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_hetu.py

python ${workdir}/../models/load_data.py # download and preprocessing criteo dataset
heturun -s 1 -w 4 python ${mainpy} --model wdl_criteo --val --comm Hybrid --cache lfuopt --bound 3
