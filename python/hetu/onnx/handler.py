from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import inspect

from hetu.onnx import constants


class hetu_op:

    _OPSETS = collections.OrderedDict()
    _MAPPING = None

    def __init__(self, name, onnx_op=None, domain=constants.ONNX_DOMAIN, **kwargs):

        if not isinstance(name, list):
            name = [name]
        self.name = name
        if not isinstance(onnx_op, list):
            onnx_op = [onnx_op]*len(name)
        self.onnx_op = onnx_op
        self.domain = domain
        self.kwargs = kwargs

    def __call__(self, func):
        opset = hetu_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            hetu_op._OPSETS[self.domain] = opset
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("version_"):
                version = int(k.replace("version_", ""))
                while version >= len(opset):
                    opset.append({})
                opset_dict = opset[version]
                for i, name in enumerate(self.name):
                    opset_dict[name] = (v, self.onnx_op[i], self.kwargs)
        return func

    @staticmethod
    def get_opsets():
        return hetu_op._OPSETS

    @staticmethod
    def create_mapping(max_onnx_opset_version):
        mapping = {constants.ONNX_DOMAIN: max_onnx_opset_version}
        ops_mapping = {}
        for domain, opsets in hetu_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        ops_mapping.update(op_map)

        hetu_op._MAPPING = ops_mapping
        return ops_mapping

    @staticmethod
    def find_effective_op(name):
        """Find the effective version of an op create_mapping.
           This is used if we need to compose ops from other ops where we'd need to find the
           op that is doing to be used in the final graph, for example there is a custom op
           that overrides a onnx op ...

        :param name: The operator name.
        """
        map_info = hetu_op._MAPPING.get(name)
        if map_info is None:
            return None
        return map_info
