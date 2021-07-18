from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["ReduceMeanOp"], onnx_op=["ReduceMean"])
@hetu_op(["ReduceSumOp"], onnx_op=["ReduceSum"])
class ReduceMean(general.PassOp):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        keepdims = node.get_attr_value('keepdims', None)
        assert keepdims is not None
        node.set_attr("keepdims", keepdims[0])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)
