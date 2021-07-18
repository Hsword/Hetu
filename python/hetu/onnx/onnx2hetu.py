from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


from typing import Text, Optional, Dict, Callable, List

import hetu.onnx.onnx_opset
from hetu import Variable
from hetu.onnx import util, constants, handler
from hetu.onnx.graph import Graph

from hetu.onnx.X2hetu.handler import BackendHandler
from hetu.onnx.X2hetu.handlers import *
from hetu.onnx.graph import Node as OnnxNode

import numpy as np
import ctypes
import os
import collections
try:
    import onnx
    from onnx import (helper, onnx_pb, numpy_helper)
    from onnx.helper import make_opsetid
    from onnx import defs
except:
    print('ONNX export support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')


def load_onnx(onnx_path=None,):
    assert onnx_path is not None, 'onnx path is None!'
    onnx_model = onnx.load(onnx_path)
    return from_onnx(onnx_model)
    pass


def from_onnx(onnx_model: onnx.ModelProto,):

    initialized = [
        init.name
        for init in onnx_model.graph.initializer
    ]
    input_names = []
    for x in onnx_model.graph.input:
        if x.name not in initialized:
            input_names.append(x.name)

    # for x in onnx_model.graph.initializer:
    #     print(x.name)
    # for i,node in enumerate(onnx_model.graph.node):
    #     print(i,node.name,node.op_type,[[inp] for inp in node.input],[[inp] for inp in node.output])

    d = prepare(onnx_model)
    output_names = [x.name for x in onnx_model.graph.output]
    assert len(input_names) == 1 and len(
        output_names) == 1, 'only support length of input and output is 1 now.'
    return d[input_names[0]], d[output_names[0]]

    # if len(output_names) == 1:
    #     return d[output_names[0]]
    # return {output_name: d[output_name] for output_name in output_names}

    # output = onnx_model.graph.output
    # print(output)
    pass


def get_all_backend_handlers(opset_dict):

    handlers = {}
    for handler in BackendHandler.__subclasses__():
        handler.check_cls()

        domain = handler.DOMAIN
        version = opset_dict[domain]
        handler.VERSION = version

        since_version = 1
        if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
            try:
                since_version = defs.get_schema(
                    handler.ONNX_OP,
                    domain=handler.DOMAIN,
                    max_inclusive_version=version,
                ).since_version
            except RuntimeError:
                print(
                    "Fail to get since_version of {} in domain `{}` "
                    "with max_inclusive_version={}. Set to 1.".format(
                        handler.ONNX_OP, handler.DOMAIN, version
                    )
                )
        else:
            print(
                "Unknown op {} in domain `{}`.".format(
                    handler.ONNX_OP, handler.DOMAIN or "ai.onnx"
                )
            )
        handler.SINCE_VERSION = since_version
        handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
    return handlers


class HetuBackend(object):

    @classmethod
    def prepare(cls,
                model,
                ):
        return cls.onnx_model_2_hetu(model)

    @classmethod
    def onnx_model_2_hetu(cls, model,):
        # if model.ir_version < 3:
        #     opset_import = [make_opsetid(defs.ONNX_DOMAIN,1)]
        assert model.ir_version >= 3
        opset_import = model.opset_import
        return cls._onnx_graph_2_hetu(
            model.graph, opset_import,
        )
        pass

    @classmethod
    def _onnx_graph_2_hetu(cls, graph_def, opset,):
        handlers = cls._get_handlers(opset)

        if graph_def.initializer:
            initialized = {
                init.name: onnx.numpy_helper.to_array(init)
                for init in graph_def.initializer
            }
            input_dict_items = cls._onnx_initializer_to_input_dict_items(
                graph_def.initializer,
                initialized,
            )

        else:
            input_dict_items = []
            initialized = {}

        for node in graph_def.node:
            node = OnnxNode(node)
            # todo:should check.
            if node.op_type == 'Constant':
                initialized[node.output_tensor_names[0]] = numpy_helper.to_array(
                    node.attrs["value"]
                )
        # creating placeholders for currently unknown inputs
        for value_info in graph_def.input:
            if value_info.name in initialized.keys():
                continue
            shape = list(
                d.dim_value if (
                    d.dim_value > 0 and d.dim_param == "") else None
                for d in value_info.type.tensor_type.shape.dim
            )
            # todo,check here ,shape not use

            input_dict_items.append((value_info.name,
                                     Variable(name=value_info.name),
                                     ))
        tensor_dict = dict(input_dict_items)
        for node in graph_def.node:
            onnx_node = OnnxNode(node)
            # print(onnx_node.name,onnx_node.op_type)
            output_ops = cls._onnx_node_to_hetu_op(
                onnx_node,
                tensor_dict,
                initialized,
                handlers,
                opset=opset,
            )

        return tensor_dict

    @classmethod
    def _onnx_node_to_hetu_op(cls, node, tensor_dict, init_dict, handlers=None, opset=None,):
        # handlers = handlers or
        handler = handlers[node.domain].get(node.op_type, None)
        if handler:
            output = handler.handle(
                node, tensor_dict, init_dict=init_dict,
            )
            if not isinstance(output, (list, tuple)):
                output = [output]
            return output
        else:
            raise ValueError("{} is not supported".format(node.op_type))

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer,
                                              initialized,):

        def get_flow_shape(shape):
            if len(shape) == 0:
                return (1,)
            return shape
        return [
            (
                init.name,
                Variable(name=init.name, value=initialized[init.name]),

            )
            for init in initializer
        ]

    @classmethod
    def _get_handlers(cls, opset):
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)


prepare = HetuBackend.prepare
