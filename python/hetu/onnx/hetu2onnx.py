from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


from typing import Text, Optional, Dict, Callable, List

import hetu.onnx.onnx_opset
from hetu import Executor
from hetu.onnx import util, constants, handler
from hetu.onnx.graph import Graph


import numpy as np
import ctypes
import os
import collections
try:
    import onnx
    from onnx import (helper, onnx_pb, numpy_helper)
except:
    print('ONNX export support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')


def export(
        executor: Executor,
        inputs: list,
        outputs: list,
        onnx_save_dir: Text,
        job_name: Text = None):
    assert len(inputs) > 0
    assert len(outputs) > 0
    onnx_graph = ProcessHetuGraph(
        executor, inputs, outputs,
    )
    if job_name is None:
        job_name = 'HetutoOnnx'
    model_proto = onnx_graph.make_model(
        job_name, onnx_save_dir,
    )
    with open(onnx_save_dir, 'wb') as f:
        try:
            f.write(model_proto.SerializeToString())
        except ValueError as e:
            raise ValueError(
                "Error occured when running model_proto.SerializeToString.")

    # node_list,input_list,output_list=HetuToOnnxNaive(executor,inputs,outputs)
    # graph_proto=helper.make_graph(node_list,"test",input_list,output_list)
    # onnx.checker.check_graph(graph_proto)
    # model_def=helper.make_model(graph_proto,producer_name="test_onnx")
    # onnx.checker.check_model(model_def)
    # onnx.save(model_def,onnx_save_dir)


def ProcessHetuGraph(
        executor,
        inputs,
        outputs,
        opset=None,
):
    opset = util.FindOpset(opset)  # opset=12 on my pc
    if opset > util.get_max_supported_opset_version():
        print("Onnx package %s is too low to support opset %s!",
              util.get_onnx_version(), opset)

    (onnx_nodes, dtypes, shapes, output_names) = HetuToOnnxPrevious(
        executor, inputs, outputs)

    g = Graph(onnx_nodes, shapes, dtypes, opset, output_names=output_names)

    ops_mapping = handler.hetu_op.create_mapping(g._opset)
    HetuOnnxMapping(g, ops_mapping)
    g.topology_sort(g._nodes)
    return g


def HetuOnnxMapping(g, ops_mapping):
    mapped_op = collections.Counter()
    ops = list(g._nodes)

    for node in ops:
        op = node.op_type
        map_info = ops_mapping.get(op)

        assert map_info is not None, "op [%s:%s] is not supported" % (
            node.name, op)

        mapped_op[op] += 1
        func, onnx_op, kwargs = map_info
        if onnx_op is not None:
            node.op_type = onnx_op
        try:
            func(g, node, **kwargs)
        except Exception as ex:
            assert False, "Failed to convert node %s" % (node.name)

    return mapped_op


def HetuToOnnxPrevious(executor, inputs, outputs):

    def get_op_name(node):
        if node in inputs:
            return node.name+'-'+str(inputs.index(node))
        if node in outputs:
            return node.name+'-'+str(outputs.index(node))
        return node.name+'_'+str(node.id)

    def get_op_shape(node):
        return executor.node_to_shape_map[node]

    topo_nodes = executor.topo_order

    dtypes = {}
    node_list = []
    output_names = []
    shapes = {}
    for node in topo_nodes:
        attrs = {}
        kvs = {**node.__dict__}
        for k, v in kvs.items():
            if k in constants.NEEDLESS_ATTRS or v is None:
                continue
            # for batchnormop ,in gpu ,type(save_var and save_mean) is narray.ndarray ,not support in onnx.
            #
            if isinstance(v, hetu.ndarray.NDArray):
                # print(k,v)

                v = v.asnumpy()
            # for placeholder
            if k == 'tensor_value' and v is not None:
                v = numpy_helper.from_array(v, name=get_op_name(node))
                k = 'value'
            if k == 'name':
                v = get_op_name(node)
            if isinstance(v, np.ndarray):
                v = numpy_helper.from_array(v, name=get_op_name(node))
            if node.op_type == 'PadOp' and k == 'paddings':
                assert isinstance(v, list)
                v = np.array(v).transpose().flatten()
            if isinstance(v, ctypes.c_ulong):
                v = v.value
            attrs[k] = v

        try:
            if node in inputs:
                attrs['op_type'] = 'defined_in'
            attrs['inputs'] = [get_op_name(no) for no in attrs['inputs']]
            attrs['outputs'] = [get_op_name(node)]
            if node in outputs:
                try:
                    assert len(attrs['outputs']) == 1, "Failed: output node %s must have one output" % (
                        node.name)

                    defined_out_name = 'defined_out' + \
                        attrs['outputs'][0][attrs['outputs'][0].rfind(':'):]
                    onnx_node = helper.make_node('Identity', inputs=attrs['outputs'],
                                                 outputs=[defined_out_name],
                                                 name=defined_out_name)
                    # fixme:hetu tensor have only one dtype as np.float now.
                    if node.op_type == 'OneHotOp':
                        dtype = np.int64
                    else:
                        dtype = np.float32
                    dtypes[onnx_node.name] = util.numpy_to_onnx_dtype(dtype)
                    if onnx_node.name not in shapes:
                        shapes[onnx_node.name] = get_op_shape(node)
                    output_names.append(onnx_node.name)
                    node_list.append(onnx_node)
                except Exception as ex:
                    print("convert failed for %s to defined_out, ex=%s" %
                          (node.name, ex))
                    raise

        except Exception as ex:
            print("format inputs failed for %s, ex=%s" % (node.name, ex))
            raise
        assert attrs.__contains__('op_type')
        assert attrs.__contains__('inputs')
        assert attrs.__contains__('outputs')

        # fixme:hetu tensor have only one dtype as np.float now.
        # fixme:only variableop add dtype attr now.
        if attrs.__contains__('dtype'):
            dtype = attrs['dtype']
            # same name of 'dtype' in onnx:make_node. so del it first.
            del attrs['dtype']
        else:
            dtype = np.float32
        dtypes[attrs['name']] = util.numpy_to_onnx_dtype(dtype)

        if attrs['name'] not in shapes:
            shapes[attrs['name']] = get_op_shape(node)
        try:
            onnx_node = helper.make_node(**attrs,)
            node_list.append(onnx_node)
        except Exception as ex:
            print(attrs)
            print("convert failed for %s, ex=%s" % (node.name, ex))
            raise
    return node_list, dtypes, shapes, output_names
