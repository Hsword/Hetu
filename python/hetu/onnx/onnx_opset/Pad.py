from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["PadOp"], onnx_op=["Pad"])
class Pad:
    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        pads = node.get_attr_value('paddings', None)
        assert pads is not None
        node.set_attr('pads', pads)

        support_modes = ['constant', 'reflect', 'edge']
        mode = node.get_attr_value('mode', 'constant').lower()
        assert mode in support_modes
        node.set_attr('mode', mode)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        pads = node.get_attr_value('paddings', None)
        assert pads is not None
        paddings = np.array(pads).astype(np.int64)
        paddings_node = ctx.make_const(util.make_name(node.name), paddings)
        node.input_tensor_names = node.input_tensor_names + \
            paddings_node.output_tensor_names

        support_modes = ['constant', 'reflect', 'edge']
        mode = node.get_attr_value('mode', 'constant').lower()
        assert mode in support_modes
        node.set_attr('mode', mode)

        constant_value = node.get_attr_value('constant_values', None)
        constant_value = np.array([constant_value]).astype(np.float32)
        constant_value_node = ctx.make_const(
            util.make_name(node.name), constant_value,)
        node.input_tensor_names = node.input_tensor_names + \
            constant_value_node.output_tensor_names
