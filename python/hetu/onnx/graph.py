from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import copy
import logging
import six
import numpy as np
from os.path import join as pathjoin

from onnx import (
    helper,
    numpy_helper,
    OperatorSetIdProto,
    AttributeProto,
    TensorProto,
    onnx_pb,
)

from hetu.onnx import util
from hetu.onnx.util import FindOpset
from hetu.onnx import constants


class Node(object):
    def __init__(self, node, graph=None):
        self._op = node
        self._graph = graph
        self._inputs = list(node.input)
        self._outputs = list(node.output)
        self._attrs = {}
        if graph is not None:
            graph.set_node_by_name(self)
        for a in node.attribute:
            self._attrs[a.name] = a

    @property
    def input_tensor_names(self):
        return self._inputs

    @input_tensor_names.setter
    def input_tensor_names(self, val):
        self._inputs = copy.deepcopy(val)

    @property
    def output_tensor_names(self):
        return copy.deepcopy(self._outputs)

    @property
    def name(self):
        return self._op.name

    @property
    def op_type(self):
        return self._op.op_type

    @op_type.setter
    def op_type(self, val):
        self._op.op_type = val

    @property
    def is_graph_input(self):
        return self.op_type in ['defined_in']

    @property
    def is_graph_output(self):
        return self.op_type in ['defined_out']

    @property
    def input_nodes(self):
        return [self._graph.get_node_by_outputname(n) for n in self._inputs]

    @property
    def op(self):
        return self._op

    def set_attr(self, name, val):
        self._attrs[name] = helper.make_attribute(name, val,)

    def get_attr(self, name, default=None):
        return self._attrs.get(name, default)

    def get_attr_value(self, name, default=None):
        attr = self.get_attr(name)
        if attr:
            attr_val = helper.get_attribute_value(attr)
            if isinstance(attr_val, bytes):
                attr_val = attr_val.decode('utf-8')
            return attr_val
        return default

    @property
    def is_const(self):
        return self.op_type in ['Const'] or \
            (self.op_type in ['PlaceholderOp']
             and self._attrs.get('value') is not None)

    def get_tensor_value(self, as_list=True):
        assert self.is_const, "Failed: Node {} must be Const".format(self.name)
        t = self.get_attr('value')
        t = numpy_helper.to_array(helper.get_attribute_value(t))
        if as_list:
            t = t.tolist()
        return t

    def onnx_attrs(self):
        schema = util.get_schema(self.op_type, self._graph._opset)
        onnx_attrs = {}
        for name, attr in self._attrs.items():
            if name == 'value':
                onnx_attrs[name] = self._attrs['value']
            elif schema is None or schema.has_attribute(name):
                onnx_attrs[name] = attr
        return onnx_attrs

    def update_node_proto(self):
        nodes = list(self._op.input)
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self._inputs)
        nodes = list(self._op.output)
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self._outputs)

        del self._op.attribute[:]

        attr = list(self.onnx_attrs().values())
        if attr:
            self._op.attribute.extend(attr)

    # add for X2hetu

    @property
    def domain(self):
        """Return Op type."""
        return self._op.domain


class Graph(object):

    def __init__(self, nodes, shapes=None, dtypes=None, opset=None, output_names=None):

        self._nodes = []
        self._nodename_to_node = {}
        self._outputname_to_nodename = {}
        self._dtypes = dtypes
        self._shapes = shapes
        self._opset = FindOpset(opset)
        self._outputs = output_names if output_names is not None else []
        ops = [Node(node, self) for node in nodes]
        self.update_graph_nodes(ops)

    def update_graph_nodes(self, ops):

        remained_dtypes = {}
        remained_shapes = {}
        self._outputname_to_nodename = {}
        for op in ops:
            for op_output in op.output_tensor_names:
                if op_output in self._dtypes:
                    remained_dtypes[op_output] = self._dtypes[op_output]
                if op_output in self._shapes:
                    remained_shapes[op_output] = self._shapes[op_output]
                self._outputname_to_nodename[op_output] = op.name

        self._nodes = ops
        self._nodename_to_node = {op.name: op for op in ops}
        self._dtypes = remained_dtypes
        self._shapes = remained_shapes

    def set_node_by_name(self, node):
        self._nodename_to_node[node.name] = node
        for outputname in node._outputs:
            self._outputname_to_nodename[outputname] = node.name

    def get_node_by_outputname(self, outputname):
        nodename = self._outputname_to_nodename.get(outputname)
        if nodename:
            return self._nodename_to_node.get(nodename)
        return None

    def get_shape(self, name):
        return self._shapes.get(name)

    def set_shape(self, name, val):
        self._shapes[name] = val

    def get_dtype(self, name):
        return self._dtypes.get(name)

    def set_dtype(self, name, val):
        self._dtypes[name] = val

    def update_node_shape_dtype(self, node):
        if node.is_const or node.is_graph_input:
            return
        initializers = []
        for i, inp in enumerate(node.input_nodes):
            if inp.is_const:
                tensor = util.TensorProtoFromNumpy(inp.get_tensor_value(as_list=False),
                                                   name=inp.output_tensor_names[0])
                initializers.append(tensor)
        input_shapes = [self.get_shape(i) for i in node.input_tensor_names]
        input_dtypes = [self.get_dtype(i) for i in node.input_tensor_names]
        shapes, dtypes = util.InferOnnxShapeDtype(
            node, self._opset, input_shapes, input_dtypes, initializers)
        if not shapes or not dtypes:
            return
        for output, shape, dtype in zip(node.output_tensor_names, shapes, dtypes):
            self.set_dtype(output, dtype)
            self.set_shape(output, shape)

    def make_const(self, name, np_val, raw=False, is_0D_tensor=False):

        shape = [] if is_0D_tensor else np_val.shape
        if raw:
            onnx_tensor = None
            # fixme:   Not yet implemented
            pass
        else:
            onnx_tensor = helper.make_tensor(
                name,
                util.numpy_to_onnx_dtype(np_val.dtype),
                shape,
                np_val,
                raw=False,
            )
        dtype = onnx_tensor.data_type
        node = self.make_node(
            "Const",
            [],
            outputs=[name],
            name=name,
            attr={"value": onnx_tensor},
            dtypes=[dtype],
        )
        self.set_shape(name, shape)
        self.set_dtype(name, dtype)
        return node

    def make_node(self, op_type, inputs, attr=None, output_count=1, outputs=None,
                  name=None, shapes=None, dtypes=None):
        if attr is None:
            attr = {}
        if shapes is None:
            shapes = []
        if dtypes is None:
            dtypes = []
        if name is None:
            name = util.make_name(op_type)
        if outputs is None:
            outputs = [name+':'+str(i) for i in range(output_count)]
        output_count = len(outputs)

        onnx_node = helper.make_node(
            op_type, inputs, outputs, name=name, **attr)

        node = Node(onnx_node, self)

        if shapes:
            assert len(
                shapes) == output_count, "Failed: output shapes count not equal to output count when make_node"
            for i in range(output_count):
                self.set_shape(node._outputs[i], shapes[i])
        if dtypes:
            assert len(
                dtypes) == output_count, "Failed: output dtypes count not equal to output count when make_node"
            for i in range(output_count):
                self.set_dtype(node._outputs[i], dtypes[i])
        if not shapes or not dtypes:
            self.update_node_shape_dtype(node)
        self._nodes.append(node)
        return node

    def insert_new_node_on_input(self, node, op_type, input_name, name=None, **kwargs):
        if name is None:
            name = util.make_name(node.name)
        new_output = util.make_name(name)
        if not isinstance(input_name, list):
            input_name = [input_name]
        new_node = self.make_node(
            op_type,
            input_name,
            attr=kwargs,
            outputs=[new_output],
            name=name,
        )
        for i, n in enumerate(node.input_tensor_names):
            if n == input_name[0]:
                node.input_tensor_names[i] = new_output
                break
        return new_node

    def insert_new_node_on_output(self, op_type, output_name, name, **kwargs):
        new_output = util.make_name(name)
        new_node = self.make_node(
            op_type,
            [output_name],
            attr=kwargs,
            outputs=[new_output],
            name=name,
        )
        for node in self._nodes:
            if node == new_node:
                continue
            for i, input_name in enumerate(node.input_tensor_names):
                if input_name == output_name:
                    node.input_tensor_names[i] = new_output
        return new_node

    def replace_input(self, node, old_input, new_input, input_index=None):
        if input_index is None:
            for i, input_name in enumerate(node._inputs):
                if input_name == old_input:
                    node._inputs[i] = new_input
        elif node._inputs[input_index] == old_input:
            node._inputs[input_index] = new_input
        else:
            raise RuntimeError("Failed:Unable to replace input %r into %r for node %r." % (
                old_input, new_input, node.name))

    def topology_sort(self, ops):
        def _push_stack(stack, node, in_stack):
            stack.append(node)
            if node in in_stack:
                raise ValueError("Graph has cycles.")
            in_stack[node] = True

        def _get_unvisited_child(g, node, not_visited):
            for child in g[node]:
                if child in not_visited:
                    return child
            return -1

        ops.sort(key=lambda op: op.name)
        n = len(ops)
        g = [[] for _ in range(n)]
        op_name_to_index = {}
        for i, op in enumerate(ops):
            op_name_to_index[op.name] = i
        for i, op in enumerate(ops):
            all_input = list(op.input_tensor_names)
            for inp in sorted(all_input):
                j = self.get_node_by_outputname(inp)
                g[op_name_to_index[j.name]].append(i)

        label = [-1 for _ in range(n)]
        stack = []
        in_stack = dict()
        not_visited = dict.fromkeys(range(n))
        label_counter = n-1
        while not_visited:
            node = list(not_visited.keys())[0]
            _push_stack(stack, node, in_stack)
            while stack:
                node = _get_unvisited_child(g, stack[-1], not_visited)
                if node != -1:
                    _push_stack(stack, node, in_stack)
                else:
                    node = stack.pop()
                    in_stack.pop(node)
                    not_visited.pop(node)
                    label[node] = label_counter
                    label_counter -= 1
        ret = [x for _, x in sorted(zip(label, ops))]
        self.update_graph_nodes(ret)

    def make_model(self, graph_doc, onnx_filename, graph_name='hetu.python.onnx'):
        graph = self.make_graph(
            graph_doc, onnx_filename, graph_name=graph_name,
        )
        model_proto = helper.make_model(graph)
        return model_proto

    def make_graph(self, doc, onnx_filename, graph_name='hetu.python.onnx'):

        for node in self._nodes:
            node.update_node_proto()

        ops = []
        const_ops = []
        input_ops = []
        for op in self._nodes:
            if op.is_const:
                const_ops.append(op)
                continue
            if op.is_graph_input:
                input_ops.append(op)
                continue
            ops.append(op)
        initializers = []
        for op in const_ops:
            tensor_name = op.output_tensor_names[0]
            tensor = util.TensorProtoFromNumpy(
                op.get_tensor_value(as_list=False),
                tensor_name,
                export_path=onnx_filename,
            )
            initializers.append(tensor)

        # sorted inputs by input id.  input_tensor_name like this:   A:0,B:1
        # fixme:mybe outputs should be sort also
        input_ids = [op.output_tensor_names[0] for op in input_ops]
        input_ids = sorted(input_ids, key=lambda x: int(x.split('-')[-1]))

        if self._opset < 9:
            input_ids += [op.output_tensor_names[0] for op in const_ops]
        input_tensor_values = self.MakeOnnxGraphIO(input_ids)
        output_tensor_values = self.MakeOnnxGraphIO(self._outputs)

        graph = helper.make_graph(
            [op.op for op in ops],
            graph_name,
            input_tensor_values,
            output_tensor_values,
            initializer=initializers,
            doc_string=doc,
        )

        return graph

    def MakeOnnxGraphIO(self, ids):
        tensor_value_infos = []
        for name in ids:
            dtype = self.get_dtype(name)
            shape = self.get_shape(name)
            v = util.MakeOnnxInputsOutputs(name, dtype, shape)
            tensor_value_infos.append(v)
        return tensor_value_infos

    def copy_shape(self, input_name, output_name):
        shape = self.get_shape(input_name)
        if shape is not None:
            self.set_shape(output_name, shape)
