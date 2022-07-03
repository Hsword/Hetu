from __future__ import absolute_import
from .Node import Op


class PipelineReceiveOp(Op):
    def __init__(self, source, comm, use_indexed_slices=False, ctx=None):
        # infer shape is done by executor
        from ..communicator.mpi_nccl_comm import ncclDataType_t
        super().__init__(PipelineReceiveOp, [], ctx)
        if isinstance(source, int):
            source = (source,)
        else:
            source = tuple(source)
        self.const_attr = source
        self.comm = comm
        self.dtype = ncclDataType_t.ncclFloat32
        self.index = 0
        self.ntargets = len(source)
        self.use_indexed_slices = use_indexed_slices

    @property
    def desc(self):
        return self.name + \
            '(%s receive from %s)' % (
                str(self.ctx.device_id), str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.use_indexed_slices:
            self.comm.dlarrayRecv(
                output_val.indices, self.dtype, self.const_attr[self.index], stream_handle)
            self.comm.dlarrayRecv(
                output_val.values, self.dtype, self.const_attr[self.index], stream_handle)
        else:
            self.comm.dlarrayRecv(
                output_val, self.dtype, self.const_attr[self.index], stream_handle)
        self.step_index()

    def step_index(self):
        self.index = (self.index + 1) % self.ntargets

    def forward_hook(self, config):
        from ..ndarray import is_gpu_ctx
        from ..stream import create_event_handle
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.event = create_event_handle(self.ctx)


def pipeline_receive_op(source, comm, use_indexed_slices=False, ctx=None):
    """Make a new instance of PipelineReceiveOp and call the instance.

    Parameters:
    ----
    source : scalar value
        The gpu index for source.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PipelineReceiveOp(source, comm, use_indexed_slices=use_indexed_slices, ctx=ctx)
