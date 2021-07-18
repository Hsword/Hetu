from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["AddByConstOp"], onnx_op=["Add"])
class Add:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        convert_const_to_node(ctx, node)

        shape0 = ctx.get_shape(node._inputs[0])
        shape1 = ctx.get_shape(node._inputs[1])
        if shape0 != shape1:
            node.set_attr('broadcast', 1)
            if shape0 and shape1 and len(shape0) < len(shape1):
                tmp = node._inputs[0]
                ctx.replace_input(node, node._inputs[0], node._inputs[1], 0)
                ctx.replace_input(node, node._inputs[1], tmp, 1)
        else:
            node.set_attr('broadcast', 0)

    @classmethod
    def version_6(cls, ctx, node, **kwargs):

        convert_const_to_node(ctx, node)

        shape0 = ctx.get_shape(node._inputs[0])
        shape1 = ctx.get_shape(node._inputs[1])
        if shape0 and shape1 and len(shape0) < len(shape1):
            tmp = node._inputs[0]
            ctx.replace_input(node, node._inputs[0], node._inputs[1], 0)
            ctx.replace_input(node, node._inputs[1], tmp, 1)


def convert_const_to_node(ctx, node):
    const = node.get_attr_value('const_attr', None)
    assert const is not None, "Failed: const_attr is none when AddConst op mapping"
    const_node = ctx.make_const(util.make_name(
        'const'), np.array([const], dtype=np.float32), raw=False)
    node.input_tensor_names = node.input_tensor_names+[const_node.name]
