from ctypes import *
import ctypes
from hetu import ndarray
import numpy as np
import os
from enum import Enum


def _load_mpi_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_mpi_runtime_api.so")
    lib = CDLL(path_to_so_file, RTLD_LOCAL)
    return lib


lib_mpi = _load_mpi_lib()


class MPIDataType_t(Enum):
    MPI_Char = 0
    MPI_Int = 1
    MPI_Uint32 = 2
    MPI_Int64 = 3
    MPI_Uint64 = 4
    MPI_Float32 = 5
    MPI_Float64 = 6


class MPIOp_t(Enum):
    MPI_OP_NULL = 0
    MPI_MAX = 1
    MPI_MIN = 2
    MPI_SUM = 3


class MPI_Communicator():

    def __init__(self):
        '''
            mpicomm: the MPI communicator, to use in MPI_Bcast, MPI_Reduce, MPI_Scatter, etc
            ncclcomm: the NCCL communicator, to use in ncclAllReduce ...
            nRanks: the total number of MPI threads
            myRanks: the rank in all MPI threads
            localRank: the rank among the MPI threads in this device
            ncclId: ncclGetUniqueId should be called once when creating a communicator
                    and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank.
            stream: the stream for NCCL communication
        '''
        self.mpicomm = c_int64(0)
        self.nRanks = c_int32(0)
        self.myRank = c_int32(0)
        self.MPI_Init()
        self.MPI_GetComm()
        self.MPI_Get_Comm_rank()
        self.MPI_Get_Comm_size()

    def MPI_Init(self):
        lib_mpi.MPIInit()

    def MPI_GetComm(self):
        lib_mpi.MPIGetComm(ctypes.byref(self.mpicomm))

    def MPI_Get_Comm_rank(self):
        lib_mpi.getMPICommRank(ctypes.byref(
            self.mpicomm), ctypes.byref(self.myRank))

    def MPI_Get_Comm_size(self):
        lib_mpi.getMPICommSize(ctypes.byref(
            self.mpicomm), ctypes.byref(self.nRanks))

    def rank(self):
        return self.myRank

    def size(self):
        return self.nRanks

    def DLArrayAllReduce(self, dlarray, datatype, reduceop):
        lib_mpi.dlarrayAllReduce(dlarray.handle, c_int(datatype.value), c_int(
            reduceop.value), ctypes.byref(self.mpicomm))

    def allReduce(self, arr):
        self.DLArrayAllReduce(arr, MPIDataType_t.MPI_Float32, MPIOp_t.MPI_SUM)

    def finish(self):
        lib_mpi.MPIFinalize()


def mpi_communicator():
    '''

    '''
    return MPI_Communicator()


# mpirun --allow-run-as-root -np 4 python2 mpi_comm.py
if __name__ == "__main__":
    comm = mpi_communicator()
    comm.MPI_GetComm()
    print("rank = %d" % (comm.rank().value))
    arr = np.ones([10]) * comm.rank().value
    arr = ndarray.array(arr)
    comm.allReduce(arr)
    print(arr.asnumpy())
    comm.finish()
