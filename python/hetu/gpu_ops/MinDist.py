from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import minimum_distance_vector
from ..ndarray import empty


class MinDistOp(Op):
    # put argmin result in codebook, return the sum of all the two-norm distance
    def __init__(self, query, key, codebook, indices, mode='euclidean', ctx=None):
        mode = mode[:2]
        assert mode in ('eu', 'in')  # euclidean distance and inner product
        super().__init__(MinDistOp, [query, key, codebook, indices], ctx)
        self.grad_node = None
        self.mode = mode

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            minimum_distance_vector(
                input_vals[0], input_vals[1], input_vals[2], input_vals[3], output_val, self.mode, stream_handle)

    def gradient(self, output_grad):
        if self.mode == 'eu':
            from .EmbeddingLookUp import embedding_lookup_gradient_op
            self.grad_node = embedding_lookup_gradient_op(
                output_grad, None, None, ctx=self.raw_ctx)
            return [None, self.grad_node, None, None]
        else:
            return [None, None, None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 4
        self.index = empty(input_shapes[0][:-1], ctx=self.ctx)
        if self.grad_node is not None:
            self.grad_node.embed_shape = input_shapes[0]
            self.grad_node.index = self.index
        return input_shapes[0]


def min_dist_op(query, key, codebook, indices, mode='eu', ctx=None):
    return MinDistOp(query, key, codebook, indices, mode, ctx=None)
