from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from ..stream import create_event_handle


class AllToAllOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(AllToAllOp, [node_A], ctx)
        self.comm = None

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
                self.comm.dlarrayAllToAll(
                    input_vals[0], output_val, self.dtype, stream_handle)
                self.event.record(stream_handle)
            else:
                assert False

    def gradient(self, output_grad):
        return [alltoall_op(output_grad, ctx=self.raw_ctx)]
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

def alltoall_op(node, ctx=None):
    """AllToAll Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AllToAllOp(node, ctx=ctx)
