from ctypes import *
from hetu import ndarray
from hetu.stream import *
from hetu.context import DeviceGroup
import numpy as np
from enum import Enum
import os
import socket


def _load_nccl_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_mpi_nccl_runtime_api.so")
    lib = CDLL(path_to_so_file, RTLD_LOCAL)
    return lib


lib_mpi_nccl = _load_nccl_lib()
# lib_mpi_nccl = CDLL("./lib_mpi_nccl_runtime_api.so", RTLD_LOCAL)


def GroupStart():
    lib_mpi_nccl.GroupStart()


def GroupEnd():
    lib_mpi_nccl.GroupEnd()


class ncclDataType_t(Enum):
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclNumTypes = 9


class ncclRedOp_t(Enum):
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4  # only avaiable if nccl>=2.10


class ncclUniqueId(Structure):
    _fields_ = [("internal", (c_int8 * 128))]


class MPI_Communicator(object):
    def __init__(self, devices=None):
        '''
            mpicomm: the MPI communicator, to use in MPI_Bcast, MPI_Reduce, MPI_Scatter, etc
            nRanks: the total number of MPI threads
            myRanks: the rank in all MPI threads
            localRank: the rank among the MPI threads in this device
        '''
        self.mpicomm = c_int64(0)
        self.nRanks = c_int32(0)
        self.myRank = c_int32(0)
        self.localRank = c_int32(-1)
        self.device_id = c_int(0)

        self.MPI_Init()
        self.MPIGetComm()
        self.MPI_Comm_rank()
        self.MPI_Comm_size()
        self.hostHashs = (c_ulonglong * self.nRanks.value)()
        self.hostDevices = (c_int * self.nRanks.value)()
        self.getLocalRank()

        self.devices = devices
        self.device_id.value = self.getDeviceFromLocalRank(
            self.localRank.value)
        self.getGlobalDevice()
        self.hostname = socket.gethostname()

    @property
    def dev_id(self):
        return self.device_id.value

    @property
    def local_rank(self):
        return self.localRank.value

    @property
    def rank(self):
        return self.myRank.value

    @property
    def nrank(self):
        return self.nRanks.value

    def MPI_Init(self):
        lib_mpi_nccl.MPIInit()

    def MPI_Finalize(self):
        lib_mpi_nccl.MPIFinalize()

    def MPIGetComm(self):
        lib_mpi_nccl.MPIGetComm(ctypes.byref(self.mpicomm))

    def MPI_Broadcast(self, buffer, size, root=0):
        lib_mpi_nccl.MPIBcast(buffer, size, root, self.mpicomm)

    def MPI_Comm_rank(self):
        lib_mpi_nccl.getMPICommRank(ctypes.byref(
            self.mpicomm), ctypes.byref(self.myRank))

    def MPI_Comm_size(self):
        lib_mpi_nccl.getMPICommSize(ctypes.byref(
            self.mpicomm), ctypes.byref(self.nRanks))

    def getLocalRank(self):
        lib_mpi_nccl.getLocalRank(ctypes.byref(
            self.mpicomm), self.nRanks, self.myRank, ctypes.byref(self.localRank), self.hostHashs)

    def getGlobalDevice(self):
        lib_mpi_nccl.getGlobalDevice(ctypes.byref(
            self.mpicomm), self.nRanks, self.myRank, self.device_id, self.hostDevices)

    def getRankFromDevice(self, hostname, device_id):
        if hostname == 'localhost':
            hostname = socket.gethostname()
        # hash
        result = 5381
        for c in hostname:
            result = (result * 33 + ord(c)) % 1000003
        rank = 0
        while rank < self.nrank and (result != self.hostHashs[rank] or device_id != self.hostDevices[rank]):
            rank += 1
        assert rank < self.nrank, 'Device %d in host %s not found.' % (
            device_id, hostname)
        return rank

    def getDeviceFromLocalRank(self, local_rank):
        return self.devices[local_rank] if self.devices else local_rank

    def getLocalRankFromDevice(self, device_id):
        return self.devices.index(device_id) if self.devices else device_id

    def ncclInit(self, stream=None):
        return NCCL_Communicator(self, stream=stream)

    def ncclGroupInit(self, devices_context, stream=None):
        return NCCL_Communicator(self, devices_context, stream=stream)

    def __del__(self):
        self.MPI_Finalize()


class NCCL_Communicator():
    def __init__(self, comm, devices_context=None, stream=None):
        '''
            ncclcomm: the NCCL communicator, to use in ncclAllReduce ...
            ncclId: ncclGetUniqueId should be called once when creating a communicator
                    and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank.
            stream: the stream for NCCL communication
        '''
        self.mpi_communicator = comm
        self.mpicomm = comm.mpicomm
        self.nRanks = comm.nRanks
        self.myRank = comm.myRank
        self.localRank = comm.localRank
        self.device_id = comm.device_id

        if stream == None:
            self.stream = create_stream_handle(
                ndarray.gpu(self.device_id.value))
        else:
            self.stream = stream

        self.ncclId = ncclUniqueId()
        self.ncclcomm = c_int64(0)
        self.ncclSetDevice(self.device_id.value)
        if devices_context is None:
            self.ncclGetUniqueId()
            self.ncclCommInitRank()
            self.group_list = None
        elif isinstance(devices_context, tuple):
            # add for preduce case
            global_rank = self.rank
            rank = devices_context.index(self.rank)
            nrank = len(devices_context)
            self.nRanks = c_int32(nrank)
            self.myRank = c_int32(rank)
            # we don't need to know about local rank in partial reduce
            self.localRank = c_int32(rank)
            tag = hash(devices_context) % 10000007
            self.ncclGetGroupUniqueId(
                (c_int32 * nrank)(*devices_context), c_int32(global_rank), self.nRanks, c_int32(tag))
            self.ncclCommInitRank()
            self.group_list = list(devices_context)
        else:
            assert isinstance(
                devices_context, DeviceGroup), "Devices context should be a DeviceGroup."
            group_list = list(devices_context)
            if len(set(group_list)) != len(group_list):
                print("Warning: Repeated ranks are found in the group.")
                group_list = list(set(group_list))
            self.group_list = group_list

            # the group_list here is as list of ndarray.(Remote)DLContext
            global_rank = self.rank
            global_size = self.nrank
            group_rank = -1
            group_size = len(group_list)
            local_rank = -1
            rank_list = []
            assert group_size <= global_size, "Error: Too many ranks in the group."
            local_rank_cnt = 0
            for i in range(group_size):
                at_local = group_list[i].local
                hostname = 'localhost' if at_local else group_list[i].hostname
                cur_rank = self.mpi_communicator.getRankFromDevice(
                    hostname, group_list[i].device_id)
                if cur_rank == global_rank:
                    group_rank = i
                    local_rank = local_rank_cnt
                    assert self.dev_id == group_list[i].device_id
                elif at_local:
                    local_rank_cnt += 1
                rank_list.append(cur_rank)
                assert cur_rank < global_size, "Error: The range of ranks should be [0, nrank-1]."

            self.nRanks = c_int32(group_size)
            self.myRank = c_int32(group_rank)
            self.localRank = c_int32(local_rank)

            if local_rank >= 0:
                group_id = 1234
                for x in rank_list:
                    group_id += x
                    group_id *= 33
                    group_id %= 10000007
                self.ncclGetGroupUniqueId(
                    (c_int32 * group_size)(*rank_list), c_int32(global_rank), self.nRanks, c_int32(group_id))
                self.ncclCommInitRank()

    @property
    def dev_id(self):
        return self.device_id.value

    @property
    def local_rank(self):
        return self.localRank.value

    @property
    def rank(self):
        return self.myRank.value

    @property
    def nrank(self):
        return self.nRanks.value

    def ncclSetDevice(self, device_id):
        self.device_id.value = device_id
        lib_mpi_nccl.setDevice(self.device_id.value)

    def getRankFromDevice(self, ctx):
        if self.group_list is None:
            return self.mpi_communicator.getRankFromDevice(ctx.hostname, ctx.device_id)
        else:
            return self.group_list.index(ctx)

    def ncclGetUniqueId(self, senderRank=0):
        lib_mpi_nccl.getNcclUniqueId(ctypes.byref(
            self.ncclId), self.mpicomm, self.localRank, c_int(senderRank))

    def ncclGetGroupUniqueId(self, group_list, ori_rank, group_size, group_id):
        lib_mpi_nccl.getGroupNcclUniqueId(ctypes.byref(
            self.ncclId), self.mpicomm, ori_rank, group_list, group_size, group_id)

    def ncclCommInitRank(self):
        '''
            Use partial AllReduce to change here.
            self.nRanks is the number of threads to use ncclallreduce
            self.myRank is the rank among these threads. the value must in [0, self.nRank - 1]
        '''
        lib_mpi_nccl.initNcclCommRank(ctypes.byref(self.ncclcomm), self.nRanks, ctypes.byref(
            self.ncclId), self.myRank, self.localRank)

    def dlarrayNcclAllReduce(self, input_arr, output_arr, datatype, reduceop, executor_stream=None):
        lib_mpi_nccl.dlarrayAllReduce(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(
            reduceop.value), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayNcclReduce(self, input_arr, output_arr, root, datatype=ncclDataType_t.ncclFloat32, reduceop=ncclRedOp_t.ncclSum, executor_stream=None):
        lib_mpi_nccl.dlarrayReduce(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(
            reduceop.value), c_int(root), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayBroadcast(self, input_arr, output_arr, datatype, root, executor_stream=None):
        lib_mpi_nccl.dlarrayBroadcast(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(
            root), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayAllGather(self, input_arr, output_arr, datatype, executor_stream=None):
        lib_mpi_nccl.dlarrayAllGather(input_arr.handle, output_arr.handle, c_int(
            datatype.value), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayReduceScatter(self, input_arr, output_arr, datatype, reduceop, executor_stream=None):
        lib_mpi_nccl.dlarrayReduceScatter(input_arr.handle, output_arr.handle, c_int(datatype.value), c_int(
            reduceop.value), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarraySend(self, arr, datatype, target, executor_stream=None):
        lib_mpi_nccl.dlarraySend(arr.handle, c_int(datatype.value), c_int(
            target), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayRecv(self, arr, datatype, src, executor_stream=None):
        lib_mpi_nccl.dlarrayRecv(arr.handle, c_int(datatype.value), c_int(
            src), self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayHA2AGather(self, sendarr, recvarr, datatype, myrank, num_local_gpus, executor_stream=None):

        lib_mpi_nccl.dlarrayHA2AGather(sendarr.handle, recvarr.handle, c_int(datatype.value), myrank, num_local_gpus,self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayHA2AScatter(self, sendarr, recvarr, datatype, myrank, num_local_gpus, executor_stream=None):
        lib_mpi_nccl.dlarrayHA2AScatter(sendarr.handle, recvarr.handle, c_int(datatype.value), myrank, num_local_gpus, self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle)

    def dlarrayAllToAll(self, sendarr, recvarr, datatype, executor_stream=None):
        lib_mpi_nccl.dlarrayAllToAll(sendarr.handle, recvarr.handle, c_int(datatype.value), \
                                    self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle, self.nRanks)

    def dlarrayHAllToAll(self, sendarr, recvarr, datatype, num_nodes, num_local_gpus, executor_stream=None):
        lib_mpi_nccl.dlarrayHAllToAll(sendarr.handle, recvarr.handle, c_int(datatype.value),\
                                    self.ncclcomm, executor_stream.handle if executor_stream else self.stream.handle, num_nodes, num_local_gpus)

    def ncclCommDestroy(self):
        lib_mpi_nccl.commDestroyNccl(ctypes.byref(self.ncclcomm))

    def __del__(self):
        self.ncclCommDestroy()


def mpi_communicator(devices=None):
    return MPI_Communicator(devices=devices)


# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 python mpi_nccl_comm.py
if __name__ == "__main__":
    t = mpi_communicator()
    t = t.ncclInit()

    send_arr = np.ones(16)*t.localRank.value
    recv_arr = np.ones(16)*t.localRank.value
    print("before: send_arr = "+str(send_arr)+" recv_arr = "+str(recv_arr))
    send_arr = ndarray.array(send_arr, ctx=ndarray.gpu(t.device_id.value))
    recv_arr = ndarray.array(recv_arr, ctx=ndarray.gpu(t.device_id.value))
#t.dlarrayNcclAllReduce(
#       arr, arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
	
    t.dlarrayAllToAll(send_arr, recv_arr, ncclDataType_t.ncclFloat32)
    print("after:  send_arr = "+str(send_arr.asnumpy())+" recv_arr = "+str(recv_arr.asnumpy()))
