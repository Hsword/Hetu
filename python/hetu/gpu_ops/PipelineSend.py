from __future__ import absolute_import
from .Node import Op


class PipelineSendOp(Op):
    def __init__(self, node_A, destination, comm, ctx=None):
        # infer shape is done by executor
        from ..communicator.mpi_nccl_comm import ncclDataType_t
        super().__init__(PipelineSendOp, [node_A], ctx)
        if isinstance(destination, int):
            destination = (destination,)
        else:
            destination = tuple(destination)
        self.const_attr = destination
        self.comm = comm
        self.dtype = ncclDataType_t.ncclFloat32
        self.index = 0
        self.ntargets = len(destination)
        self.use_indexed_slices = node_A.use_indexed_slices

    @property
    def desc(self):
        return self.name + \
            '(send node %s to rank %s)' % (
                self.inputs[0].name, str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.use_indexed_slices:
            self.comm.dlarraySend(
                input_vals[0].indices, self.dtype, self.const_attr[self.index], stream_handle)
            self.comm.dlarraySend(
                input_vals[0].values, self.dtype, self.const_attr[self.index], stream_handle)
        else:
            self.comm.dlarraySend(
                input_vals[0], self.dtype, self.const_attr[self.index], stream_handle)
        self.step_index()

    def step_index(self):
        self.index = (self.index + 1) % self.ntargets

    def forward_hook(self, config):
        from ..ndarray import is_gpu_ctx
        from ..stream import create_event_handle
        self.ctx = self.inputs[0].ctx
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        # add event to the previous node, ensure that the send is
        # blocked until previous computations are finished
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
        self.event = create_event_handle(self.ctx)


def pipeline_send_op(node, destination, comm, ctx=None):
    """Make a new instance of PipelineSendOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to be send.
    destination : scalar value
        The gpu index for destination.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PipelineSendOp(node, destination, comm, ctx=ctx)
