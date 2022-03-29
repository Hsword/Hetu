import numpy as np
import hetu as ht
from hetu import layers as htl
import argparse
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_rank0(msg):
    if device_id == 0:
        logger.info(msg)

def moe(args, x, y_, indice):
    experts = []
    for i in range(args.num_local_experts):
        experts.append(htl.Expert(embed_dim = args.model_dim, ffn_dim = args.hidden_size,\
                            dropout_rate = 0.1, activation='relu',\
                            name = "expert_%d"%(device_id* args.num_local_experts + i)))
    gate = htl.HashGate(embed_dim = args.model_dim, num_tokens = args.batch_size * args.num_tokens, \
                        num_experts = args.num_local_experts * num_gpus)

    model = htl.HashLayer(gate = gate, experts = experts, num_tokens = args.num_tokens, embed_dim = args.model_dim, all2all_size=num_gpus)

    y = model(x, indice)
            
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y

# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 -x PYTHONPATH=/home/v-xiaonannie/Hetu/python /opt/conda/bin/python /home/v-xiaonannie/Hetu/tests/test_moe_op.py 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_tokens', type=int, default=1024)
    parser.add_argument('--model_dim', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--num_local_experts', type=int, default=2)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--comm-mode', default='AllReduce', help='communication mode')

    args = parser.parse_args()

    global device_id, num_gpus
    global x_val, y_val, dispatch_input_val
    device_id = 0
    num_gpus = 8
    print_rank0("Training MoE Examples on HETU")
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.local_rank
    executor_ctx = ht.gpu(device_id % 8)
    x_val = np.random.normal(size = (args.batch_size, args.num_tokens, args.model_dim))
    x = ht.Variable(name='x', ctx=executor_ctx)
    n_classes=2048
    targets = np.random.randint(0, high=n_classes, size=(args.batch_size*args.num_tokens))
    y_val=np.eye(n_classes)[targets]

    y_ = ht.Variable(name='y_', ctx=executor_ctx)

    indice_val = np.random.randint(0, high=args.num_local_experts*num_gpus, size=(args.batch_size*args.num_tokens, 1)).astype(np.float32)
    indice = ht.Variable(name="indice", ctx=executor_ctx)
    
    opt = ht.optim.SGDOptimizer(learning_rate=1e-5)

    loss, y = moe(args, x, y_, indice)
    train_op = opt.minimize(loss)
    eval_nodes = {'train': [loss, y, y_, train_op]}

    executor = ht.Executor(eval_nodes, ctx=executor_ctx,
                           comm_mode=args.comm_mode)
    
    for i in range(1000):
        print_rank0("Step %d" % i)
        loss_val, predict_y, y_val, _ = executor.run(
            'train', eval_node_list=[loss, y, y_, train_op], feed_dict={x:x_val, y_: y_val, indice:indice_val})
        print_rank0("Train loss = %f" % loss_val.asnumpy())
    
    print_rank0("Success!")
