
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hetu.onnx.handler import hetu_op

from onnx import helper
from hetu.onnx.onnx_opset import general


@hetu_op(["PlaceholderOp"], onnx_op=["Placeholder"])
class PlaceholderOp:
    @classmethod
    def version_1(clsc, ctx, node, **kwargs):
        val = node.get_attr_value('value')
        if(val is not None):
            node.op_type = "Const"


@hetu_op(["defined_in"])
class Defined_In(general.PassOp):
    pass
