from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect
import os
import shutil

from onnx import defs


class BackendHandler:

    ONNX_OP = None

    DOMAIN = defs.ONNX_DOMAIN
    VERSION = 0
    SINCE_VERSION = 0
    PARTIAL_SUPPORT = False
    PS_DESCRIPTION = ""
    ONEFLOW_BLOBNAME_MAP = {}
    ONEFLOW_CODE_GEN = []
    OP_OUTPUS = []

    @classmethod
    def check_cls(cls):
        if not cls.ONNX_OP:
            print('doesn`t have ONNX_OP')

    @staticmethod
    def onnx_op(op):
        return BackendHandler.property_register("ONNX_OP", op)

    @classmethod
    def handle(cls, node, tensor_dict, **kwargs):

        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            return ver_handle(node, tensor_dict, **kwargs)
        raise ValueError(
            'node "{}" of version {} is not supported'.format(
                node.op_type, cls.SINCE_VERSION
            )
        )

    @classmethod
    def run_onnx_node(cls,
                      node, tensor_dict,
                      inputs=None,
                      attrs=None,
                      name='',
                      **kwargs,
                      ):
        if inputs is None:
            inputs = [tensor_dict.get(inp, None)
                      for inp in node.input_tensor_names]
        if attrs is None:
            attrs = copy.deepcopy(node._attrs)
        if name != "":
            attrs["name"] = name
        for inp in node.input_tensor_names:
            if tensor_dict[inp] not in cls.ONEFLOW_BLOBNAME_MAP:
                cls.ONEFLOW_BLOBNAME_MAP[tensor_dict[inp]] = inp
        cls.OP_OUTPUS = []
        for oup in node.output_tensor_names:
            cls.OP_OUTPUS.append(oup)
        # todo
        # y = cls._run_flow_func(flow_func, inputs, attrs)
        # if type(y) == list():
        #     for x in cls.OP_OUTPUS:
        #         if y[x] not in cls.ONEFLOW_BLOBNAME_MAP:
        #             cls.ONEFLOW_BLOBNAME_MAP[y[x]] = x
        # else:
        #     if y not in cls.ONEFLOW_BLOBNAME_MAP:
        #         cls.ONEFLOW_BLOBNAME_MAP[y] = cls.OP_OUTPUS[0]
        return None  # y

    @staticmethod
    def property_register(name, value):
        def deco(cls):
            setattr(cls, name, value)
            return cls

        return deco


onnx_op = BackendHandler.onnx_op
