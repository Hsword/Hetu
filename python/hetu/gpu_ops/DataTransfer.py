from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .. import stream


class DataH2DOp(Op):
    # not support sparse matrix!!!
    # for sparse matrix, please set Variable's ctx to gpu and pass value in feed_dict
    def __init__(self, node_A, ctx):
        super().__init__(DataH2DOp, [node_A], ctx)
        assert ndarray.is_gpu_ctx(ctx)
        assert not ndarray.is_gpu_ctx(node_A.ctx)
        self.event = None
        self.on_cpu = False
        self.on_gpu = True
        self.dtype = node_A.dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        if stream_handle:
            if self.event is None:
                self.event = stream.create_event_handle(self.ctx)
            output_val.async_h2d(input_vals[0], stream_handle, self.event)
        else:
            input_vals[0].copyto(output_val)

    def gradient(self, output_grad):
        if output_grad.use_indexed_slices:
            return [datad2h_sparse_op(output_grad)]
        else:
            return [datad2h_op(output_grad)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        pass


class DataD2HOp(Op):
    def __init__(self, node_A):
        assert not node_A.use_indexed_slices
        super().__init__(DataD2HOp, [node_A], ndarray.cpu(0))
        assert ndarray.is_gpu_ctx(node_A.ctx)
        self.event = None
        self.on_cpu = True
        self.on_gpu = False
        self.dtype = node_A.dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        if stream_handle:
            if self.event is None:
                self.event = stream.create_event_handle(self.inputs[0].ctx)
            output_val.async_d2h(input_vals[0], stream_handle, self.event)
        else:
            input_vals[0].copyto(output_val)

    def gradient(self, output_grad):
        return [datah2d_op(output_grad, ctx=self.inputs[0].ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        pass


class DataD2HSparseOp(Op):
    # here sparse means indexed slices
    def __init__(self, node_A):
        assert node_A.use_indexed_slices
        super().__init__(DataD2HSparseOp, [node_A], ndarray.cpu(0))
        assert ndarray.is_gpu_ctx(node_A.ctx)
        self.event = None
        self.on_cpu = True
        self.on_gpu = False
        self.use_indexed_slices = True

    def compute(self, input_vals, output_val, stream_handle=None):
        assert isinstance(input_vals[0], ndarray.IndexedSlices)
        assert isinstance(output_val, ndarray.IndexedSlices)
        # TODO: include all these parts into memory allocation management!!!
        # TODO: also consider how to deduplicate
        if stream_handle:
            if self.event is None:
                self.event = stream.create_event_handle(self.inputs[0].ctx)
            output_val.indices.async_d2h(
                input_vals[0].indices, stream_handle, self.event)
            output_val.values.async_d2h(
                input_vals[0].values, stream_handle, self.event)
        else:
            input_vals[0].indices.copyto(output_val.indices)
            input_vals[0].values.copyto(output_val.values)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        pass


class DataH2DSparseOp(Op):
    # here sparse means indexed slices
    def __init__(self, node_A, ctx):
        super().__init__(DataH2DSparseOp, [node_A], ctx)
        assert not ndarray.is_gpu_ctx(node_A.ctx)
        self.event = None
        self.on_cpu = False
        self.on_gpu = True
        self.use_indexed_slices = True

    def compute(self, input_vals, output_val, stream_handle=None):
        assert isinstance(input_vals[0], ndarray.IndexedSlices)
        assert isinstance(output_val, ndarray.IndexedSlices)
        # TODO: include all these parts into memory allocation management!!!
        # TODO: also consider how to deduplicate
        if output_val.indices is None or output_val.indices.shape != input_vals[0].indices.shape:
            output_val.indices = ndarray.empty(
                input_vals[0].indices.shape, ctx=self.ctx)
            output_val.values = ndarray.empty(
                input_vals[0].values.shape, ctx=self.ctx)
        if stream_handle:
            if self.event is None:
                self.event = stream.create_event_handle(self.ctx)
            output_val.indices.async_h2d(
                input_vals[0].indices, stream_handle, self.event)
            output_val.values.async_h2d(
                input_vals[0].values, stream_handle, self.event)
        else:
            input_vals[0].indices.copyto(output_val.indices)
            input_vals[0].values.copyto(output_val.values)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        pass


def datah2d_op(node, ctx):
    """Transfer data from host(CPU) to device(GPU).

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DataH2DOp(node, ctx=ctx)


def datad2h_op(node):
    """Transfer data from device(GPU) to host(CPU).

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DataD2HOp(node)


def datad2h_sparse_op(node):
    """Transfer sparse data from device(GPU) to host(CPU).

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DataD2HSparseOp(node)


def datah2d_sparse_op(node, ctx):
    """Transfer sparse data from host(CPU) to device(GPU).

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DataH2DSparseOp(node, ctx)
