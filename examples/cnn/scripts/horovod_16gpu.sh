
#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/../run_tf_horovod.py

# horovodrun -np 8 -H localhost:8 python ${mainpy} --model tf_mlp --dataset CIFAR10 --learning-rate 0.00125 --validate --timing

horovodrun -np 16 --start-timeout 3000 -H node1:8,node2:8 python ${mainpy} --model $1 --dataset $2 --learning-rate 0.01 --validate --timing

# ../build/_deps/openmpi-build/bin/mpirun -mca btl_tcp_if_include enp97s0f0 --bind-to none --map-by slot\
#  -x NCCL_SOCKET_IFNAME=enp97s0f0 -H node1:8,node2:8 --allow-run-as-root python run_tf_horovod.py --model
