from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from onnx import onnx_pb, numpy_helper
from hetu.onnx import constants, util, graph
from hetu.onnx.handler import hetu_op
from hetu.onnx.onnx_opset import general


@hetu_op(["Batch_NormalizationOp"], onnx_op=["BatchNormalization"])
class BatchNormalization:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        epsilon = node.get_attr_value('eps', 0.01)
        node.set_attr("epsilon", epsilon)
        momentum = node.get_attr_value('momentum', 0.99)

        mean = node.get_attr_value('save_mean', None)
        var = node.get_attr_value('save_var', None)
        assert mean is not None and var is not None

        mean = numpy_helper.to_array(mean)
        var = numpy_helper.to_array(var)
        mean = np.reshape(mean, [-1])
        var = np.reshape(var, [-1])
        new_mean_node_name = util.make_name(node.name+'_mean_')
        new_mean_node = ctx.make_const(new_mean_node_name, mean)
        node.input_tensor_names += new_mean_node.output_tensor_names

        new_val_node_name = util.make_name(node.name+'_var_')
        new_val_node = ctx.make_const(new_val_node_name, var)
        node.input_tensor_names += new_val_node.output_tensor_names
