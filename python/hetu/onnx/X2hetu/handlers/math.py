import numpy as np

from hetu.onnx.X2hetu.handler import BackendHandler, onnx_op
import hetu as ht


@onnx_op('Cast')
class Cast(BackendHandler):

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        output_shape = tensor_dict[node.input_tensor_names[0]].tensor_value
        assert output_shape is not None

        tensor_dict[node.output_tensor_names[0]] = output_shape

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)


@onnx_op('Add')
class Add(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        assert False, 'not yet implemented! addd version 1'

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        assert False, 'not yet implemented! add version 6'

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        inputs = [tensor_dict.get(inp, None)
                  for inp in node.input_tensor_names]
        assert len(inputs) == 2
        y = ht.add_op(inputs[0], inputs[1])
        tensor_dict[node.output_tensor_names[0]] = y
        return y


@onnx_op('MatMul')
class MatMul(BackendHandler):

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        a = tensor_dict[node.input_tensor_names[0]]
        b = tensor_dict[node.input_tensor_names[1]]

        y = ht.matmul_op(a, b, )
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        assert False, 'not yet implemented! matmul version 9'
        pass


@onnx_op('Gemm')
class Gemm(BackendHandler):

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        a = tensor_dict[node.input_tensor_names[0]]
        b = tensor_dict[node.input_tensor_names[1]]
        transA = False if node.get_attr_value("transA", 0) == 0 else True
        transB = False if node.get_attr_value("transB", 0) == 0 else True

        y = ht.matmul_op(a, b, trans_A=transA, trans_B=transB)
        if len(node.input_tensor_names) > 2:
            z = tensor_dict[node.input_tensor_names[2]]
            y = ht.add_op(y, z)
        tensor_dict[node.output_tensor_names[0]] = y
        return y

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        cls.version_1(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        assert False, 'not yet implemented! matmul version 9'
        pass
