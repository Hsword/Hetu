import numpy as np

from hetu.onnx.X2hetu.handler import BackendHandler, onnx_op

import hetu as ht


@onnx_op('Identity')
class Identify(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        inputs = [tensor_dict.get(inp, None)
                  for inp in node.input_tensor_names]
        assert len(inputs) == 1
        y = inputs[0]
        tensor_dict[node.output_tensor_names[0]] = y
        return y


@onnx_op('Reshape')
class Reshape(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        output_shape = tensor_dict[node.input_tensor_names[1]]

        y = ht.array_reshape_op(x, output_shape)
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_5(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_14(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)


@onnx_op('Transpose')
class Transpose(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        perm = node.get_attr_value('perm')
        y = ht.transpose_op(x, perm=perm)
        tensor_dict[node.output_tensor_names[0]] = y

        return y

    @classmethod
    def version_5(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_14(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)


@onnx_op('Slice')
class Slice(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        ends = node.get_attr_value('ends')
        starts = node.get_attr_value('starts')
        # for hetu ,size is (-1,size) ,it is output_shape
        assert len(ends) == len(starts)
        size = [ends[i] - starts[i] for i in range(len(ends))]

        size[0] = -1
        y = ht.slice_op(x, begin=starts, size=size)
        tensor_dict[node.output_tensor_names[0]] = y
        return y


@onnx_op('Concat')
class Concat(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        a = tensor_dict[node.input_tensor_names[0]]
        b = tensor_dict[node.input_tensor_names[1]]
        axis = node.get_attr_value('axis')

        y = ht.concat_op(a, b, axis=axis)
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_4(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)
