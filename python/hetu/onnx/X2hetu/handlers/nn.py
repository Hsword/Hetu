import numpy as np

from hetu.onnx.X2hetu.handler import BackendHandler, onnx_op

import hetu as ht


@onnx_op('Conv')
class Conv2d(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        in_weights = tensor_dict[node.input_tensor_names[1]]
        in_weights_shape = list(in_weights.shape)
        paddings = node.get_attr_value('pads')
        strides = node.get_attr_value('strides')
        assert len(set(paddings)) == 1 and len(set(strides)) == 1

        y = ht.conv2d_op(x, in_weights, padding=paddings[0], stride=strides[0])
        tensor_dict[node.output_tensor_names[0]] = y
        return y


@onnx_op('Relu')
class Relu(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        inputs = [tensor_dict.get(inp, None)
                  for inp in node.input_tensor_names]
        assert len(inputs) == 1
        y = ht.relu_op(inputs[0])
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)


@onnx_op('AveragePool')
class AveragePool(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        inputs = [tensor_dict.get(inp, None)
                  for inp in node.input_tensor_names]
        assert len(inputs) == 1
        kernel_shape = node.get_attr_value('kernel_shape')
        strides = node.get_attr_value('strides')
        assert len(kernel_shape) == 2 and len(strides) == 2
        assert strides[0] == strides[1], 'strides 0 and 1 must be equal now!'

        # todo,here padding set to 0. check.
        y = ht.avg_pool2d_op(inputs[0], kernel_H=kernel_shape[0],
                             kernel_W=kernel_shape[1], padding=0, stride=strides[0])
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)
