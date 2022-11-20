from __future__ import absolute_import
from .Node import Op
from ..ndarray import is_gpu_ctx
from ..stream import create_event_handle


class BroadcastCommunicateOp(Op):
    def __init__(self, nodeA, comm, root, ctx):
        inputs = [] if root != comm.rank else [nodeA]
        super().__init__(BroadcastCommunicateOp, inputs, ctx)
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.comm = comm
        self.root = root
        self.event = create_event_handle(self.ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        input_val = input_vals[0] if len(self.inputs) == 1 else output_val
        self.comm.dlarrayBroadcast(
            input_val, output_val, self.dtype, self.root, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        from ..ndarray import array, empty
        import numpy as np
        if self.comm.rank == self.root:
            shape = input_shapes[0]
            if len(shape) < 4:
                shape = [0] * (4 - len(shape)) + list(shape)
            new_res = array(np.array(shape), ctx=self.ctx)
        else:
            new_res = empty((4,), ctx=self.ctx)
        self.comm.dlarrayBroadcast(new_res, new_res, self.dtype, self.root, self.p2p_stream)
        self.p2p_stream.sync()
        shape = new_res.asnumpy()
        return tuple(int(x) for x in list(shape) if x > 0)

    def forward_hook(self, config):
        from ..communicator.mpi_nccl_comm import ncclDataType_t
        if self.on_gpu and len(self.inputs) == 1 and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
            # disable inplace if not lazy execution
            # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
            self.inputs[0].inplace = False
        self.p2p_stream = config.p2p_stream
        self.dtype = ncclDataType_t.ncclFloat32
        assert self.on_gpu


def broadcastCommunicate_op(node, comm, root, ctx):
    """Make a new instance of BroadcastCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do broadcast

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BroadcastCommunicateOp(node, comm, root, ctx)
