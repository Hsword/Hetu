from __future__ import absolute_import
from .Node import Op
from ..stream import create_event_handle
from ..cpu_links import assign_embedding_with_indexedslices as cpu_assign_embedding_with_indexedslices
from ..gpu_links import assign_embedding_with_indexedslices, assign_quantized_embedding, \
    assign_quantized_embedding_unified


class AssignWithIndexedSlicesOp(Op):
    def __init__(self, embed, unique, newparam, ctx=None):
        super().__init__(AssignWithIndexedSlicesOp,
                         [embed, unique, newparam], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            cpu_assign_embedding_with_indexedslices(
                input_vals[0], input_vals[1], input_vals[2])
        else:
            assign_embedding_with_indexedslices(
                input_vals[0], input_vals[1], input_vals[2], stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        self.ctx = self.inputs[0].ctx
        self.on_gpu = self.inputs[0].on_gpu
        self.on_cpu = self.inputs[0].on_cpu
        if self in config.eval_node_list and self.on_gpu and self.event is None:
            self.event = create_event_handle(self.ctx)


def assign_with_indexedslices_op(embed, unique, newparam, ctx=None):
    return AssignWithIndexedSlicesOp(embed, unique, newparam, ctx=ctx)


class AssignQuantizedEmbeddingOp(Op):
    def __init__(self, embed, unique, newparam, digit, scale=None, minele=None, middle=None, qparam=None, ctx=None):
        inputs = [embed, unique, newparam]
        self.digit = digit
        if qparam is not None:
            inputs.append(qparam)
        else:
            self.scale = scale
            if minele is not None:
                self.minele = minele
            else:
                self.minele = middle - 2 ** (digit - 1) * scale
        super().__init__(AssignQuantizedEmbeddingOp, inputs, ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if len(input_vals) == 4:
            if self.on_cpu:
                raise NotImplementedError
            else:
                assign_quantized_embedding(
                    input_vals[0], input_vals[1], input_vals[2], input_vals[3], self.digit, stream_handle)
        else:
            if self.on_cpu:
                raise NotImplementedError
            else:
                assign_quantized_embedding_unified(
                    input_vals[0], input_vals[1], input_vals[2], self.scale, self.minele, self.digit, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        self.ctx = self.inputs[0].ctx
        self.on_gpu = self.inputs[0].on_gpu
        self.on_cpu = self.inputs[0].on_cpu
        if self in config.eval_node_list and self.on_gpu and self.event is None:
            self.event = create_event_handle(self.ctx)


def assign_quantized_embedding_op(embed, unique, newparam, digit, scale=None, minele=None, middle=None, qparam=None, ctx=None):
    return AssignQuantizedEmbeddingOp(embed, unique, newparam, digit, scale=scale, minele=minele, middle=middle, qparam=qparam, ctx=ctx)
