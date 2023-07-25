
#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_tf_horovod.py

horovodrun -np 8 -H localhost:8 python ${mainpy} --model $1 --dataset $2 --learning-rate 0.00125 --validate --timing
