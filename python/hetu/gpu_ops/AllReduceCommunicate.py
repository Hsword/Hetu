from __future__ import absolute_import
from .Node import Op
from ..ndarray import is_gpu_ctx, NDArray, IndexedSlices, empty
from ..stream import create_event_handle


class AllReduceCommunicateOp(Op):
    def __init__(self, nodeA, comm):
        super().__init__(AllReduceCommunicateOp, [nodeA], nodeA.ctx)
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.comm = comm
        self.use_indexed_slices = nodeA.use_indexed_slices

    def compute(self, input_vals, output_val, stream_handle=None):
        if isinstance(input_vals[0], NDArray):
            self.comm.dlarrayNcclAllReduce(
                input_vals[0], output_val, self.dtype, self.reduce_op, stream_handle)
        elif isinstance(input_vals[0], IndexedSlices):
            self.comm.dlarrayAllGather(
                input_vals[0].indices, output_val.indices, self.dtype, stream_handle)
            self.comm.dlarrayAllGather(
                input_vals[0].values, output_val.values, self.dtype, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]

    def forward_hook(self, config):
        from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
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
        self.reduce_op = ncclRedOp_t.ncclSum


def allreduceCommunicate_op(node, comm):
    """Make a new instance of AllReduceCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do allreduce

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AllReduceCommunicateOp(node, comm)


class AllReduceCommunicateP2POp(AllReduceCommunicateOp):
    def __init__(self, nodeA, comm):
        Op.__init__(self, AllReduceCommunicateP2POp, [nodeA], nodeA.ctx)
        self.on_gpu = is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        self.comm = comm
        self.use_indexed_slices = nodeA.use_indexed_slices


def allreduceCommunicatep2p_op(node, comm):
    return AllReduceCommunicateP2POp(node, comm)


class GroupAllReduceCommunicateOp(Op):
    def __init__(self, nodeA, group_comm):
        super().__init__(GroupAllReduceCommunicateOp, [nodeA], nodeA.ctx)
        self.group_comm = group_comm

    def compute(self, input_vals, output_val, stream_handle=None):
        from ..communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
        input_vals[0].copyto(output_val)
        self.group_comm.dlarrayNcclAllReduce(
            output_val, output_val, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def groupallreduceCommunicate_op(node, group_comm):
    """Make a new instance of GroupAllReduceCommunicateOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to do groupallreduce

    Returns:
    ----
    A new Node instance created by Op.

    """
    return GroupAllReduceCommunicateOp(node, group_comm)
