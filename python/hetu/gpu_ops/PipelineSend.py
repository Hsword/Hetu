from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t, GroupStart
from ..stream import create_event_handle, create_stream_handle


class PipelineSendOp(Op):
    def __init__(self, node_A, destination, comm, stream=None, ctx=None):
        super().__init__(PipelineSendOp, [node_A], ctx)
        self.const_attr = destination
        self.comm = comm
        self.comm_stream = stream
        self.desc = self.name + \
            '(send node %s to rank %s)' % (node_A.name, str(destination))
        self.shape = None
        self.shape_is_sent = False

    def compute(self, input_vals, output_val, stream_handle=None, group_call=False):
        assert not self.on_cpu, "PipelineSendOp only support P2P communication between gpus"
        if group_call:
            GroupStart()

        self.comm.dlarraySend(input_vals[0], ncclDataType_t.ncclFloat32, self.const_attr, stream_handle)

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        shape = input_shapes[0]
        self.shape = shape
        if not self.shape_is_sent:
            self.shape_is_sent = True
            # pad shape so that len=4
            if len(shape) < 4:
                shape = [0] * (4 - len(shape)) + list(shape)
            # construct and send
            payload = ndarray.array(shape, self.ctx)
            self.comm.dlarraySend(payload,
                                  ncclDataType_t.ncclFloat32,
                                  self.const_attr,
                                  self.comm_stream)
        return shape

    def forward_hook(self, config):
        self.ctx = self.inputs[0].ctx
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        # add event to the previous node, ensure that the send is
        # blocked until previous computations are finished
        if self.on_gpu and self.inputs[0].event is None:
            self.inputs[0].event = create_event_handle(self.ctx)
        self.event = create_event_handle(self.ctx)


def pipeline_send_op(node, destination, comm, stream=None, ctx=None):
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
    return PipelineSendOp(node, destination, comm, stream=stream, ctx=ctx)
