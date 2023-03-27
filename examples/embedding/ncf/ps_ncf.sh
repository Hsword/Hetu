#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/run_hetu.py

python ${mainpy} --comm PS --cache lfuopt --bound 3 --config ${workdir}/../ctr/settings/local_s1_w4.yml
