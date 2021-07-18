from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["MatMulOp"], onnx_op=["MatMul"])
class MatMul:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        trans_a = node.get_attr_value('matmul_attr_trans_A', 0)
        trans_b = node.get_attr_value('matmul_attr_trans_B', 0)
        # fixme:only supported matrixs have two dims now
        if trans_a != 0:
            ctx.insert_new_node_on_input(
                node, 'Transpose', node._inputs[0], perm=[1, 0])
        if trans_b != 0:
            ctx.insert_new_node_on_input(
                node, 'Transpose', node._inputs[1], perm=[1, 0])
