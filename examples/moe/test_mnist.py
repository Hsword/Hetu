import numpy as np
import hetu as ht
from hetu import init
from hetu import layers as htl
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time
myrank=0

def print_rank0(msg):
    if myrank == 0:
        logger.info(msg)

def fc(x, shape, name):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    return x

def moe(args, x, y_):
    experts = []
    for i in range(args.num_local_experts):
        experts.append(htl.Expert(embed_dim = args.model_dim, ffn_dim = args.hidden_size,\
                            dropout_rate = 0.1, activation='relu',\
                            name = "expert_%d"%(device_id* args.num_local_experts + i)))
    gate = htl.TopKGate(embed_dim = args.model_dim, num_tokens = args.batch_size * args.num_tokens, \
                        num_experts = args.num_local_experts * num_gpus, k = args.top)

    model = htl.MoELayer(gate = gate, experts = experts, num_tokens = args.num_tokens, embed_dim = args.model_dim, all2all_size=num_gpus, top=args.top)

    y, l_aux = model(x)

    y=ht.array_reshape_op(y, [args.batch_size, -1])
    y=fc(y, (784, 10), "mlp")
    #y=ht.reduce_sum_op(y, axes = 2)
    #y=ht.softmax_op(y)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    loss = loss+0.01*l_aux 
    

    return loss, y

# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 -x PYTHONPATH=/home/v-xiaonannie/Hetu/python /opt/conda/bin/python /home/v-xiaonannie/Hetu/tests/test_moe_op.py 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_tokens', type=int, default=28)
    parser.add_argument('--model_dim', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--num_local_experts', type=int, default=2)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--top', type=int, default=2)
    parser.add_argument('--l_aux_wt', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--comm-mode', default='AllReduce', help='communication mode')

    args = parser.parse_args()

    global device_id, num_gpus
    global x_val, y_val, dispatch_input_val
    device_id = 0
    num_gpus = 8
    print_rank0("Training MoE Examples on HETU")
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.local_rank
    myrank = comm.myRank.value
    print("device_id: ", device_id)
    executor_ctx = ht.gpu(device_id % 8)
    opt = ht.optim.SGDOptimizer(learning_rate=1e-6)

    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    x = ht.dataloader_op([ht.Dataloader(train_set_x, args.batch_size, 'train')])
    y_ = ht.dataloader_op([ht.Dataloader(train_set_y, args.batch_size, 'train')])
    

    loss, y = moe(args, x, y_)
    train_op = opt.minimize(loss)
    eval_nodes = {'train': [loss, y, y_, train_op]}

    executor = ht.Executor(eval_nodes, ctx=executor_ctx,
                           comm_mode=args.comm_mode)
    n_train_batches = executor.get_batch_num('train')
    average_time=0.0
    for i in range(10000):
        print_rank0("Step %d" % i)
        start=time.time()        
        loss_val, predict_y, y_val, _  = executor.run(
            'train', eval_node_list=[loss, y, y_, train_op])
        end=time.time()
        print_rank0("Loss:"+str(loss_val.asnumpy()))
        print_rank0("Step time = %s sec."%(end-start))
       
        if i+10>=30:
            average_time+=end-start

    print_rank0("Success!")
    print_rank0("Average synced step_time=%s sec."%(average_time/10))

