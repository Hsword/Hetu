from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
import numpy as np
from ..gpu_links import sparse_embedding_lookup


class SparseEmbeddingLookUp(Op):
    def __init__(self, embedding, index, ctx=None):
        super().__init__(SparseEmbeddingLookUp, [embedding, index], ctx)
        embedding.is_embed = True

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            sparse_embedding_lookup(input_vals[0], input_vals[1],
                                    output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        output_shape = list(input_shapes[1])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)


def sparse_embedding_lookup_op(embedding, index, ctx=None):
    return SparseEmbeddingLookUp(embedding, index, ctx=ctx)
