from __future__ import absolute_import
import ctypes
from time import time
from .Node import Op
from ..cpu_links import assign_embedding_with_indexedslices as cpu_assign_embedding_with_indexedslices
from ..gpu_links import assign_embedding_with_indexedslices, assign_quantized_embedding, \
    assign_quantized_embedding_unified


class AssignWithIndexedSlicesOp(Op):
    def __init__(self, embed, newparam, ctx=None):
        super().__init__(AssignWithIndexedSlicesOp, [embed, newparam], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            cpu_assign_embedding_with_indexedslices(
                input_vals[0], input_vals[1])
        else:
            assign_embedding_with_indexedslices(
                input_vals[0], input_vals[1], stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None


def assign_with_indexedslices_op(embed, newparam, ctx=None):
    return AssignWithIndexedSlicesOp(embed, newparam, ctx=ctx)


class AssignQuantizedEmbeddingOp(Op):
    def __init__(self, embed, newparam, digit, scale=None, minele=None, middle=None, qparam=None, ctx=None):
        inputs = [embed, newparam]
        self.digit = digit
        if qparam is not None:
            inputs.append(qparam)
        else:
            self.scale = scale
            if minele is not None:
                self.minele = minele
            else:
                self.minele = middle - 2 ** (digit - 1) * scale
        self.seed = ctypes.c_ulonglong(0)
        super().__init__(AssignQuantizedEmbeddingOp, inputs, ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        self.seed.value = int(time())
        if len(input_vals) == 3:
            if self.on_cpu:
                raise NotImplementedError
            else:
                assign_quantized_embedding(
                    input_vals[0], input_vals[1], input_vals[2], self.digit, self.seed, stream_handle)
        else:
            if self.on_cpu:
                raise NotImplementedError
            else:
                assign_quantized_embedding_unified(
                    input_vals[0], input_vals[1], self.scale, self.minele, self.digit, self.seed, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None


def assign_quantized_embedding_op(embed, newparam, digit, scale=None, minele=None, middle=None, qparam=None, ctx=None):
    return AssignQuantizedEmbeddingOp(embed, newparam, digit, scale=scale, minele=minele, middle=middle, qparam=qparam, ctx=ctx)
