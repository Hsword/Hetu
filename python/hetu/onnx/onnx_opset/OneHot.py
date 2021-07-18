from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["OneHotOp"], onnx_op=["OneHot"])
class OneHot:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        assert False, "until there is no onehot op in onnx"

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        depth = node.get_attr_value('num_classes', None)
        assert depth is not None
        input_shape = ctx.get_shape(node._inputs[0])

        depth_node = ctx.make_const(util.make_name(
            node.name), np.array([depth], dtype=np.int64))
        values_node = ctx.make_const(util.make_name(
            node.name), np.array([0, 1], dtype=np.int64))
        node.input_tensor_names = node.input_tensor_names +\
            depth_node.output_tensor_names +\
            values_node.output_tensor_names

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_9(ctx, node, **kwargs)
