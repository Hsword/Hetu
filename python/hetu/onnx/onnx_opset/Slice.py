from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["SliceOp"], onnx_op=["Slice"])
class Slice:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        starts = node.get_attr_value('begin_pos')
        size = node.get_attr_value('output_shape')
        ends = np.zeros_like(starts)
        for i, s in enumerate(size):
            ends[i] = starts[i] + size[i]

        #hetu:output_shape (input_size[0],size)
        # ends[0]=starts[0]

        node.set_attr('starts', starts)
        node.set_attr('ends', ends)

        starts_node = ctx.make_const(
            util.make_name("starts"), np.array(starts, None)
        )
        ends_node = ctx.make_const(
            util.make_name("ends"), np.array(ends, None)
        )
        node.input_tensor_names = node.input_tensor_names + \
            starts_node.output_tensor_names + ends_node.output_tensor_names
