from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["Array_ReshapeOp"], onnx_op=["Reshape"])
class AveragePool:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        shape = node.get_attr_value('output_shape', None)
        assert shape is not None, "Failed: ReshapeOp does not have a const shape"
        node.set_attr("shape", shape)

    @classmethod
    def version_5(cls, ctx, node, **kwargs):
        shape = node.get_attr_value('output_shape', None)
        assert shape is not None, "Failed: ReshapeOp does not have a const shape"
        shape_node = ctx.make_const(
            util.make_name("shape"), np.array(shape, None)
        )
        node.input_tensor_names = node.input_tensor_names + shape_node.output_tensor_names
