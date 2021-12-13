from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import scatter

class ScatterOp(Op):
    def __init__(self, node_target, node_index, node_src, ctx=None):
        super().__init__(ScatterOp, [node_target, node_index, node_src], ctx)

    def compute(self, target, dim, index, src, stream_handle=None):
        scatter(target, dim, index, src)

    def gradient(self, output_grad):
        pass

    def infer_shape(self, input_shapes):
        pass

def scatter_op(node1, node2, node3, ctx=None):
    return ScatterOp(node1, node2, node3, ctx=ctx)
