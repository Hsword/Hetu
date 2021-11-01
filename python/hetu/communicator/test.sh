export PYTHONPATH=$PYTHONPATH:/home/Hetu/python
NCCL_DEBUG=INFO
mpirun --allow-run-as-root -np 4 python mpi_nccl_comm.py
