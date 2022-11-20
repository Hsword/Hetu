from __future__ import absolute_import
from .Node import Op
from ..ndarray import is_gpu_ctx
from ..stream import create_event_handle


class ReduceCommunicateOp(Op):
    def __init__(self, nodeA, comm, root):
        super().__init__(ReduceCommunicateOp, [nodeA], nodeA.ctx)
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.comm = comm
        self.root = root

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.root != self.comm.rank:
            output_val = input_vals[0]
        self.comm.dlarrayNcclReduce(
            input_vals[0], output_val, self.root, executor_stream=stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        if self.root == self.comm.rank:
            return input_shapes[0]
        else:
            return None

    def forward_hook(self, config):
        self.ctx = self.inputs[0].ctx
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
        self.event = create_event_handle(self.ctx)

        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        self.inputs[0].inplace = False
        assert self.on_gpu


def reduceCommunicate_op(node, comm, rank):
    """Make a new instance of ReduceCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do reduce

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceCommunicateOp(node, comm, rank)
