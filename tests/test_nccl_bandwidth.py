from hetu.communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
from hetu.stream import create_stream_handle
import hetu as ht
import argparse
import numpy as np
from time import time


def test_allreduce(arr, comm, stream, iterations=10):
    size = 4 * np.prod(arr.shape, dtype=int)
    duration = 0
    for _ in range(iterations):
        start = time()
        comm.dlarrayNcclAllReduce(
            arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, stream)
        stream.sync()
        duration += (time() - start)

    local_duration = ht.array(np.array([duration, ]), ht.cpu())
    comm.dlarrayNcclReduce(local_duration, local_duration,
                           0, executor_stream=stream)
    stream.sync()
    if comm.rank == 0:
        print("Algorithm bandwidth: %f GB/s" %
              (size * iterations / local_duration.asnumpy()[0] * comm.nrank / (2 ** 30)))


def test_p2p(arr, comm, stream, iterations=10):
    size = 4 * np.prod(arr.shape, dtype=int)
    duration = 0
    for _ in range(iterations):
        start = time()
        if comm.rank == 0:
            comm.dlarraySend(arr, ncclDataType_t.ncclFloat32, 1, stream)
        else:
            comm.dlarrayRecv(arr, ncclDataType_t.ncclFloat32, 0, stream)
        stream.sync()
        duration += (time() - start)

    print("Algorithm bandwidth: %f GB/s" %
          (size * iterations / duration / (2 ** 30)))


# mpirun --allow-run-as-root --tag-output -np 4 python test_nccl_bandwidth.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devs', type=str, default=None, help='devices')
    parser.add_argument('--p2p', action='store_true',
                        help='test p2p or allreduce')
    args = parser.parse_args()

    devices = None
    if args.devs is not None:
        devices = [int(d) for d in args.devs.split(',')]
    comm = ht.wrapped_mpi_nccl_init(devices=devices)
    if args.devs is None:
        devices = list(range(comm.nrank))

    shape = (1, 1000, 1000, 1000)
    ctx = ht.gpu(devices[comm.rank])
    stream = create_stream_handle(ctx)
    arr = ht.empty(shape, ctx=ctx)

    if comm.rank == 0:
        print('devices: {}'.format(devices))

    if args.p2p:
        assert len(devices) == 2, 'P2P must only use 2 devices.'
        test_p2p(arr, comm, stream, 10)
    else:
        test_allreduce(arr, comm, stream, 10)
