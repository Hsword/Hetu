from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["DropoutOp"], onnx_op=["Dropout"])
class Dropout():
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        assert False, 'todo:'
        pass

    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls.version_1(ctx, node, **kwargs)

    @classmethod
    def version_12(cls, ctx, node, **kwargs):
        radio = node.get_attr_value('keep_prob', None)
        # node.set_attr('radio',radio)
        assert radio is not None
        radio_node = ctx.make_const(util.make_name(node.name+'_radio_'),
                                    np.array([1-radio], dtype=np.float32), is_0D_tensor=True)
        node.input_tensor_names += radio_node.output_tensor_names

        training_mode = np.bool_(True)
        training_mode_node = ctx.make_const(util.make_name(node.name+'_training_mode_'),
                                            np.array([training_mode], dtype=np.bool), is_0D_tensor=True)
        node.input_tensor_names += training_mode_node.output_tensor_names

        # seed=node.get_attr_value('seed',None)
        # node.set_attr('seed',seed)
