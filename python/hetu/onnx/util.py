from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import os
import onnx
from onnx import onnx_pb, helper, defs, numpy_helper, TensorProto, OperatorSetIdProto, shape_inference
from hetu.onnx import constants, util
from collections import defaultdict, OrderedDict


#
# mapping dtypes from hetu to onnx
#
# fixme:bug,(int64) type is error
# fixme:unused now
HETU_TO_ONNX_DTYPES = {
    np.float: onnx_pb.TensorProto.FLOAT,
}


#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.BOOL: np.bool,
}


def numpy_to_onnx_dtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype "+np_dtype+" for mapping")


def onnx_to_numpy_dtype(onnx_dtype):
    return ONNX_TO_NUMPY_DTYPE[onnx_dtype]


def map_hetu_dtype(dtype):
    if dtype:
        dtype = HETU_TO_ONNX_DTYPES[dtype]
    return dtype


ONNX_UNKNOWN_DIMENSION = -1

INSERT_NAME_ID = 1


def make_name(name):
    global INSERT_NAME_ID
    INSERT_NAME_ID += 1
    return "{}_{}".format(name, INSERT_NAME_ID)


def FindOpset(opset):
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
    return opset


def get_onnx_version():
    return onnx.__version__


def MakeOnnxInputsOutputs(name, elem_type, shape, **kwargs):
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED

    return helper.make_tensor_value_info(
        name, elem_type, shape)


def GenerateValidFilename(s):
    return "".join([c if c.isalpha() or c.isdigit() else "_" for c in s])


def TensorProtoFromNumpy(
    arr: np.ndarray, name=None, external_data=False, export_path=None
):
    if name is None:
        name = make_name("tensor_")
    tp = numpy_helper.from_array(arr, name)
    # value with size < 1024 bytes will remain in .onnx file
    # (like what pytorch does)
    if (not external_data) or arr.nbytes < 1024:
        return tp
    assert tp.HasField("raw_data")
    tp.ClearField("raw_data")
    export_dir = os.path.dirname(export_path)
    filename = GenerateValidFilename(name)
    with open(os.path.join(export_dir, filename), "wb") as f:
        arr.tofile(f)
    tp.data_location = onnx_pb.TensorProto.EXTERNAL
    external_data = tp.external_data.add()
    external_data.key = "location"
    external_data.value = filename
    return tp


class OnnxOpSchema(object):

    def __init__(self, name, domain, since_version, attributes):

        self._name = name
        self._domain = domain
        self._attributes = attributes
        self._since_version = since_version

    @property
    def attributes(self):
        return self._attributes

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name

    @property
    def since_version(self):
        return self._since_version

    @staticmethod
    def FromOnnxSchema(onnx_schema):
        name = onnx_schema.name
        domain = onnx_schema.domain
        since_version = int(onnx_schema.since_version)
        attributes = onnx_schema.attributes
        return OnnxOpSchema(name, domain, since_version, attributes)

    def has_attribute(self, attr):
        return attr in self.attributes


def _RegisterAllSchemasWithHistory():
    """Register all schemas with history"""
    onnx_schemas = defs.get_all_schemas_with_history()
    name_domain_version_schema_map = defaultdict(lambda: defaultdict(dict))
    for s in onnx_schemas:
        schema = OnnxOpSchema.FromOnnxSchema(s)
        name_domain_version_schema_map[schema.name][schema.domain][
            schema.since_version
        ] = schema

    ordered_map = defaultdict(lambda: defaultdict(OrderedDict))
    for name, domain_version_schema_map in name_domain_version_schema_map.items():
        for domain, version_schema_map in domain_version_schema_map.items():
            ordered_map[name][domain] = OrderedDict(
                sorted(version_schema_map.items(), key=lambda x: -x[0])
            )
    return ordered_map


def _ParseDomainOpsetVersions(schemas):
    """ Get max opset version among all schemas within each domain. """
    domain_opset_versions = dict()
    for domain_version_schema_map in schemas.values():
        for domain, version_schema_map in domain_version_schema_map.items():
            # version_schema_map is sorted by since_version in descend order
            max_version = next(iter(version_schema_map))
            if domain not in domain_opset_versions:
                domain_opset_versions[domain] = int(max_version)
            else:
                domain_opset_versions[domain] = max(
                    domain_opset_versions[domain], int(max_version)
                )
    return domain_opset_versions


_schemas = _RegisterAllSchemasWithHistory()

_domain_opset_versions = _ParseDomainOpsetVersions(_schemas)


def get_schema(name, max_inclusive_opset_version, domain=None):
    """Get schema by name within specific version."""
    domain = domain or constants.ONNX_DOMAIN
    domain_version_schema_map = _schemas[name]
    version_schema_map = domain_version_schema_map[domain]
    for version, schema in version_schema_map.items():
        if version <= max_inclusive_opset_version:
            return schema
    return None


def get_max_supported_opset_version(domain=None):
    """Get max supported opset version by current onnx package given a domain."""
    domain = domain or constants.ONNX_DOMAIN
    return _domain_opset_versions.get(domain, None)


def InferOnnxShapeDtype(
    node, opset_version, input_shapes, input_dtypes, initializers=None
):
    """
    Infer shapes and dtypes for outputs of the node.
    Sometimes, shape inference needs the values of node's inputs, so initializers are used.
    """

    def BuildOnnxOp(node):
        """Build onnx op"""
        onnx_node = helper.make_node(
            node.op_type,
            node.input_tensor_names,
            node.output_tensor_names,
            name=node.name,
        )

        # # deal with attributes
        # attr = []
        # attr_graphs = node.get_body_graphs()
        # if attr_graphs:
        #     for attr_name, sub_graph in attr_graphs.items():
        #         copied_sub_graph = copy.deepcopy(sub_graph)
        #         graph_proto = copied_sub_graph.MakeGraph(
        #             "graph for " + node.name + " " + attr_name
        #         )
        #         attr.append(helper.make_attribute(attr_name, graph_proto))
        # attr.extend(node.attrs_onnx.values())
        # if attr:
        #     onnx_node.attribute.extend(attr)
        return onnx_node

    inputs = []
    outputs = []
    for inp, shape, dtype in zip(node.input_tensor_names, input_shapes, input_dtypes):
        inputs.append(util.MakeOnnxInputsOutputs(inp, dtype, shape))
    for output in node.output_tensor_names:
        outputs.append(util.MakeOnnxInputsOutputs(
            output, TensorProto.UNDEFINED, None))
    graph_proto = helper.make_graph(
        [BuildOnnxOp(node)], "infer-graph", inputs, outputs, initializer=initializers
    )
    imp = OperatorSetIdProto()
    imp.version = opset_version
    model_proto = helper.make_model(graph_proto, opset_imports=[imp])

    inferred_model = None
    try:
        inferred_model = shape_inference.infer_shapes(model_proto)
    except Exception:
        print('error')
        return None, None
    shapes = {}
    dtypes = {}
    for output in inferred_model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("elem_type"):
            dtypes[output.name] = tensor_type.elem_type
        else:
            dtypes[output.name] = TensorProto.UNDEFINED
        # 0 in shapes of onnx means unknown which is -1 in our convertor
        # fixme:how to do if the dim is -1 originally
        if tensor_type.HasField("shape"):
            shapes[output.name] = [
                dim.dim_value if dim.dim_value != 0 else util.ONNX_UNKNOWN_DIMENSION
                for dim in tensor_type.shape.dim
            ]
        else:
            shapes[output.name] = None
    output_shapes = []
    output_dtypes = []
    for output in node.output_tensor_names:
        if output in shapes:
            output_shapes.append(shapes[output])
        else:
            output_shapes.append(None)
        if output in dtypes:
            output_dtypes.append(dtypes[output])
        else:
            output_dtypes.append(TensorProto.UNDEFINED)
    return output_shapes, output_dtypes
