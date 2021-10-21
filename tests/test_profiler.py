import hetu as ht


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


if __name__ == '__main__':
    # mpirun --allow-run-as-root -np 4 python test_profiler.py
    test_nccl_profiler()
