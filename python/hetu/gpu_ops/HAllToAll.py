from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from ..stream import create_event_handle
from ..gpu_links import ha2a_layout_transform, ha2a_reverse_layout_transform


class HAllToAllOp(Op):
    def __init__(self, node_A, num_nodes, num_local_gpus, ctx=None):
        super().__init__(HAllToAllOp, [node_A], ctx)
        self.num_nodes = num_nodes
        self.num_local_gpus = num_local_gpus
        self.comm = None
        self.temp_data_1 = None # for all data received from other gpus
        self.temp_data_2 = None # for scattered result
        # these two cache memory can be used in turn
        # firstly, local gpus send their data to local gpu 0 (temp_data_1)
        # local rank 0 gpu merges the data received (temp_data_2)
        # all_to_all (temp_data_1)
        # local rank 0 gpu scatters the data received (temp_data_2)
        # local rank 0 gpu sends the scattered data to each local gpu

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if isinstance(input_vals[0], ndarray.NDArray):
                self.comm.dlarrayAllToAll(
                    input_vals[0], output_val, self.dtype)
            else:
                assert False
        else:
            if self.event == None:
                self.event = create_event_handle(self.ctx)
            if isinstance(input_vals[0], ndarray.NDArray):
                if self.myrank%self.num_local_gpus==0 : # local rank 0 gpu
                    if self.temp_data_1 == None:
                        x,y = input_vals[0].shape
                        self.temp_data_1 = ndarray.array(np.zeros([x*self.num_local_gpus, y], dtype=np.float32), ctx=ndarray.gpu(0))
                        self.temp_data_2 = ndarray.array(np.zeros([x*self.num_local_gpus, y], dtype=np.float32), ctx=ndarray.gpu(0))
                    self.event.record(stream_handle)
                    self.comm.dlarrayHA2AGather(input_vals[0], self.temp_data_1, self.dtype, self.myrank, self.num_local_gpus, stream_handle)
                    self.event.record(stream_handle)
                    ha2a_layout_transform(self.temp_data_1, self.temp_data_2, self.num_nodes, self.num_local_gpus,stream_handle)
                    self.comm.dlarrayHAllToAll(self.temp_data_2, self.temp_data_1, self.dtype, self.num_nodes, self.num_local_gpus, stream_handle)
                    self.event.record(stream_handle)
                    ha2a_reverse_layout_transform(self.temp_data_1, self.temp_data_2, self.num_nodes, self.num_local_gpus, stream_handle)
                    self.comm.dlarrayHA2AScatter(self.temp_data_2, output_val, self.dtype, self.myrank, self.num_local_gpus, stream_handle)
                    self.event.record(stream_handle)
                    
                else: # other gpus
                    self.event.record(stream_handle)
                    self.comm.dlarrayHA2AGather(input_vals[0], input_vals[0], self.dtype, self.myrank, self.num_local_gpus, stream_handle)
                    self.event.record(stream_handle)
                    self.event.record(stream_handle)
                    self.comm.dlarrayHA2AScatter(input_vals[0], output_val, self.dtype, self.myrank, self.num_local_gpus, stream_handle)
                    self.event.record(stream_handle)
                    
            else:
                assert False

    def gradient(self, output_grad):
        return [halltoall_op(output_grad,self.num_nodes, self.num_local_gpus, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_hook(self, config):
        from ..communicator.mpi_nccl_comm import ncclDataType_t
        self.ctx = self.inputs[0].ctx
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)

        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        self.inputs[0].inplace = False
        self.dtype = ncclDataType_t.ncclFloat32
        self.comm = config.nccl_comm
        self.local_rank = self.comm.local_rank
        self.myrank = self.comm.myRank.value
        self.local_gpu_0 = (int)(self.myrank//self.num_local_gpus) * self.num_local_gpus


def halltoall_op(node, num_nodes, num_local_gpus, ctx=None):
    """AllToAll Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return HAllToAllOp(node, num_nodes, num_local_gpus, ctx=ctx)
