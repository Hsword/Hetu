from ctypes import *
from .. import ndarray
from ..stream import *
import numpy as np
import os


def _load_nccl_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_nccl_runtime.so")
    lib = CDLL(path_to_so_file, RTLD_LOCAL)
    return lib


lib_nccl = _load_nccl_lib()


class NCCL_Communicator():

    def __init__(self, devs, devs_number):
        self.comms = (c_int64 * devs_number)(0)
        self.streams = (c_int64 * devs_number)(0)
        self.stream_handles = []
        self.devs = (c_int * devs_number)(*devs)
        self.devs_number = c_int(devs_number)
        self.send_buff = None
        self.recv_buff = None

    def _create_streams(self):
        for i in range(self.devs_number.value):
            self.stream_handles.append(create_stream_handle(ndarray.gpu(i)))
            lib_nccl.update_stream(i, self.streams, c_int64(
                self.stream_handles[-1].handle.contents.handle))

    def _destroy_streams(self):
        self.stream_handles = []
        lib_nccl.free_streams(self.streams, self.devs, self.devs_number)

    def _init_NCCL(self):
        lib_nccl.init_NCCL(self.comms, self.devs, self.devs_number)

    def _destroy_NCCL_comms(self):
        lib_nccl.finish_NCCL(self.comms, self.devs_number)

    def _stream_sync(self):
        lib_nccl.Synchronize_streams(self.streams, self.devs, self.devs_number)

    def _allreduce(self, send_buff, recv_buff, size):
        lib_nccl.NCCL_AllReduce(
            send_buff, recv_buff, size, self.comms, self.streams, self.devs_number)

    def _alltoall(self, send_buff, recv_buff, size):
        lib_nccl.NCCL_AllToAll(
            send_buff, recv_buff, size, self.comms, self.streams, self.devs_number)

    def get_send_buff(self, array):
        self.send_buff = (c_void_p * self.devs_number.value)(*array)

    def get_recv_buff(self, array):
        self.recv_buff = (c_void_p * self.devs_number.value)(*array)

    def all_reduce(self, send_array, recv_array, size):
        self.get_send_buff(send_array)
        self.get_recv_buff(recv_array)
        self._allreduce(self.send_buff, self.recv_buff, size)

    def show_property(self):
        print("self.comms = ", self.comms)
        print("self.streams = ", self.streams)
        print("self.devs = ")
        lib_nccl.for_each(self.devs, self.devs_number)
        print("self.devs_number = ", self.devs_number.value)

    def All_Reduce_Ndarray(self, gradient_list, allreduced_gradient_list):
        gradient_buff = []
        allreduced_gradient_buff = []
        for i in range(self.devs_number.value):
            gradient_buff.append(gradient_list[i].handle.contents.data)
            allreduced_gradient_buff.append(
                allreduced_gradient_list[i].handle.contents.data)
        length = 1
        for i in range(gradient_list[0].handle.contents.ndim):
            length = length * gradient_list[0].handle.contents.shape[i]

        self.all_reduce(gradient_buff, allreduced_gradient_buff, length)


def nccl_communicator(devs, devs_number):
    return NCCL_Communicator(devs, devs_number)
