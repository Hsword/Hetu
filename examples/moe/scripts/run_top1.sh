NCCL_DEBUG=DEBUG mpirun --allow-run-as-root -np 8 -x PYTHONPATH=/home/Hetu/python python test_moe_top.py --top=1 --num_local_experts=2 --batch_size=16
