export PYTHONPATH=$PYTHONPATH:/home/Hetu/python
NCCL_DEBUG=INFO
mpirun --allow-run-as-root -np 2 -mca btl_tcp_if_include enp97s0f0 -x NCCL_SOCKET_IFNAME=enp97s0f0 -H daim118:1, daim117:1  /root/anaconda3/envs/moe/bin/python mpi_nccl_comm.py
