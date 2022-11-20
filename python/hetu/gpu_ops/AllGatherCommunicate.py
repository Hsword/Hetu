from __future__ import absolute_import
from .Node import Op
from ..ndarray import is_gpu_ctx
from ..stream import create_event_handle


class AllGatherCommunicateOp(Op):
    def __init__(self, nodeA, comm):
        super().__init__(AllGatherCommunicateOp, [nodeA], nodeA.ctx)
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.comm = comm

    def compute(self, input_vals, output_val, stream_handle=None):
        self.comm.dlarrayAllGather(
            input_vals[0], output_val, self.dtype, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        input_shape = input_shapes[0]
        output_shape = list(input_shape)
        output_shape[0] *= self.comm.nrank
        return tuple(output_shape)

    def forward_hook(self, config):
        from ..communicator.mpi_nccl_comm import ncclDataType_t
        self.ctx = self.inputs[0].ctx
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
        self.event = create_event_handle(self.ctx)

        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        self.inputs[0].inplace = False
        self.dtype = ncclDataType_t.ncclFloat32
        assert self.on_gpu


def allgatherCommunicate_op(node, comm):
    """Make a new instance of AllGatherCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do allgather

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AllGatherCommunicateOp(node, comm)
