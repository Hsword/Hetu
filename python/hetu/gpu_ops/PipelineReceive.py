from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t, GroupEnd
from ..stream import create_event_handle, create_stream_handle


class PipelineReceiveOp(Op):
    def __init__(self, source, comm, stream=None, ctx=None):
        assert ctx, "PipelineReceiveOp must be initialized with the ctx argument!"
        super().__init__(PipelineReceiveOp, [], ctx)
        self.const_attr = source
        self.comm = comm
        self.comm_stream = stream
        self.desc = self.name + \
            '(%s receive from %s)' % (str(self.ctx.device_id), str(source))
        self.shape = None
        self.shape_is_received = False

    def compute(self, input_vals, output_val, stream_handle=None, group_call=False):
        assert not self.on_cpu, "PipelineReceiveOp only support P2P communication between gpus"
        assert self.comm_stream, "communicate stream should not be None"
        self.comm.dlarrayRecv(output_val, ncclDataType_t.ncclFloat32, self.const_attr, stream_handle)

        if group_call:
            GroupEnd()

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        if not self.shape_is_received:
            # receive
            shape_arr = ndarray.array([0, 0, 0, 0], self.ctx)
            self.comm.dlarrayRecv(shape_arr,
                                  ncclDataType_t.ncclFloat32,
                                  self.const_attr,
                                  self.comm_stream)

            # remove padding and save
            shape_arr = [int(x) for x in list(shape_arr.asnumpy()) if x != 0]
            self.shape = tuple(shape_arr)
            self.shape_is_received = True

        return self.shape

    def forward_hook(self, config):
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu


def pipeline_receive_op(source, comm, stream=None, ctx=None):
    """Make a new instance of PipelineReceiveOp and call the instance.

    Parameters:
    ----
    source : scalar value
        The gpu index for source.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PipelineReceiveOp(source, comm, stream=stream, ctx=ctx)
