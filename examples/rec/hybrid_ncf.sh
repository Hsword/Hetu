#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/run_hetu.py

python -m hetu.launcher ${workdir}/../ctr/settings/local_s1.yml -n 1 --sched &
mpirun --allow-run-as-root -np 4 python ${mainpy} --comm Hybrid --cache lfuopt --bound 3 --config ${workdir}/../ctr/settings/local_w4.yml
