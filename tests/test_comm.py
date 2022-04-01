import numpy as np
import hetu as ht
from hetu import layers as htl
from tester import HetuTester
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time

def print_rank0(msg):
    if device_id == 0:
        logger.info(msg)


def moe(args, x, y_):
    
    x2 = ht.alltoall_op(x)
    
    return x2

# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 -x PYTHONPATH=/home/v-xiaonannie/Hetu/python /opt/conda/bin/python /home/v-xiaonannie/Hetu/tests/test_moe_op.py 

if __name__ == "__main__":

    global device_id, num_gpus
    global x_val, y_val, dispatch_input_val
    device_id = 0
    num_gpus = 8
    print_rank0("Training MoE Examples on HETU")
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.local_rank
    executor_ctx = ht.gpu(device_id % 8)
    x_val = np.random.normal(size = (16))
    x_val = ht.array(arr=x_val, ctx=executor_ctx)
    x = ht.Variable(name='x', ctx=executor_ctx, trainable=False)

    x2 = moe(x)
    eval_nodes = {'train': [loss, y, y_, train_op]}

    executor = ht.Executor(eval_nodes, ctx=executor_ctx,
                           comm_mode=args.comm_mode)
    average_time=0.0
    for i in range(30):
        print_rank0("Step %d" % i)
        start=time.time()        
        loss_val, predict_y, y_val, _  = executor.run(
            'train', eval_node_list=[loss, y, y_, train_op], feed_dict={x:x_val, y_: y_val})
        loss_numpy=loss_val.asnumpy()
        end=time.time()
        print_rank0("Train loss = %f" % loss_numpy)
        print_rank0("Step time = %s sec."%(end-start))
       
        if i+10>=30:
            average_time+=end-start

    print_rank0("Success!")
    print_rank0("Average synced step_time=%s sec."%(average_time/10))

