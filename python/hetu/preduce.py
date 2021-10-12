from . import ndarray
from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
from hetu import get_worker_communicate, wrapped_mpi_nccl_init, new_group_comm

import numpy as np
import ctypes

class PartialReduce:
    def __init__(self, reduce_key=0):
        # reduce_key : in pipeline case, worker on each stage use a unique key
        self._reduce_key = reduce_key
        self.ps_comm = get_worker_communicate()
        self.comm = wrapped_mpi_nccl_init()
        self._comm_map = {}
        self.rank = self.comm.rank
        self.nrank = self.comm.nrank
        self._buffer = np.ascontiguousarray(np.repeat(-1, self.nrank + 1).astype(np.int32))
        self._buffer_ptr = self._buffer.ctypes.data_as(ctypes.c_void_p)

    def get_partner(self, max_worker=-1, wait_time=1.0):
        # wait_time : the max time to wait, in millisecond
        # max_worker : if max_worker reachs, get_partner will return immediately
        #               in pipeline case, max_worker should be set properly, otherwise -1 is ok
        if max_worker < 0:
            max_worker = self.nrank
        self.ps_comm.preduce_get_partner(self._reduce_key, self.rank, max_worker, ctypes.c_float(wait_time), self._buffer_ptr)
        for i in range(self.nrank + 1):
            if self._buffer[i] < 0:
                return tuple(self._buffer[0 : i])
        assert False

    def preduce(self, array, partner, stream=None):
        # array : the array to reduce on
        # partner : the partial reduce group returned by get_partner
        # stream : the stream to run allreduce on
        if partner not in self._comm_map.keys():
            self._create_partial_comm(partner)
        comm = self._comm_map[partner]
        comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream)

    def _create_partial_comm(self, partner):
        self._comm_map[partner] = new_group_comm(partner)
