import hetu as ht
from hetu.profiler import NCCLOP


def test_nccl_profiler():
    nccl_profiler = ht.NCCLProfiler()
    results = []
    results.append(nccl_profiler.profile_sendrecv(100000, [0, 2]))
    results.append(nccl_profiler.profile_sendrecv(500000, [0, 2]))
    results.append(nccl_profiler.profile_sendrecv(1000000, [0, 2]))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)

    results = []
    results.append(nccl_profiler.profile_allreduce(100000, [0, 2]))
    results.append(nccl_profiler.profile_allreduce(100000, [0, 1]))
    results.append(nccl_profiler.profile_allreduce(100000, [0, 1, 2, 3]))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)

    results = []
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 2], primitive=NCCLOP.AllGather))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1], primitive=NCCLOP.AllGather))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1, 2, 3], primitive=NCCLOP.AllGather))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)

    results = []
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 2], primitive=NCCLOP.ReduceScatter))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1], primitive=NCCLOP.ReduceScatter))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1, 2, 3], primitive=NCCLOP.ReduceScatter))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)

    results = []
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 2], primitive=NCCLOP.Reduce))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1], primitive=NCCLOP.Reduce))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1, 2, 3], primitive=NCCLOP.Reduce))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)

    results = []
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 2], primitive=NCCLOP.Broadcast))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1], primitive=NCCLOP.Broadcast))
    results.append(nccl_profiler.profile_allreduce(
        100000, [0, 1, 2, 3], primitive=NCCLOP.Broadcast))
    if nccl_profiler.mpi_comm.rank == 0:
        print(results)


if __name__ == '__main__':
    # heturun -w 4 python test_profiler.py
    test_nccl_profiler()
