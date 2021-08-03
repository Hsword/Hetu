from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from .. import stream


def Variable(name, value=None, initializer=None, trainable=True, dtype=np.float32, ctx=None):
    """
        Defined a variable.
        Trainable: Parameter
        Not Trainable: Constant
    """
    placeholder_node = placeholder_op(
        name, value, initializer, trainable, dtype, ctx)
    return placeholder_node


class PlaceholderOp(Op):
    def __init__(self, name, value=None, initializer=None, trainable=True, dtype=np.float32, ctx=None):
        super().__init__(PlaceholderOp, [], ctx)
        self.name = name
        self.is_embed = False
        self.shape = None
        if value is None and initializer is None:
            trainable = False
        elif value is not None:
            assert initializer is None, 'Value already specified, initializer should be None.'
            assert isinstance(value, (np.ndarray, ndarray.NDArray)),\
                'Value data type %s not valid.' % str(type(value))
            self.shape = value.shape
        else:
            assert initializer is not None, 'Value not specified, initializer should not be None.'
            self.shape = initializer.shape
        self.tensor_value = value
        self.initializer = initializer
        self.trainable = trainable
        self.dtype = dtype
        self.reshaped = False

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.shape, "placeholder %s values provided by feed_dict" % self.name

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        assert self.shape, "placeholder %s shape provided by feed_shape" % self.name
        return self.shape

    def forward_hook(self, config):
        pass

    def backward_hook(self, config):
        if self.ctx is None:
            self.ctx = config.context
        if (config.node_strategy.get(self, config.comm_mode) == 'PS' or (config.node_strategy.get(self, config.comm_mode) == "Hybrid" and self.is_embed)) and self.trainable:
            self.ctx = ndarray.cpu(0)
            if config.cstable_policy is not None and self.is_embed:
                self.event = stream.CSEvent(config.ps_comm, self.id)
            else:
                self.event = stream.PSEvent(config.ps_comm, self.id)
        else:
            if self.initializer:
                self.initializer(self, config.seed,
                                 config.np_rand, config.comp_stream)
                self.initializer = None
            elif self.tensor_value is not None:
                value = self.tensor_value
                assert isinstance(value, (np.ndarray, ndarray.NDArray)), \
                    'Parameters should be initialized as numpy.ndarray or ndarray.NDArray .'
                if isinstance(value, np.ndarray):
                    value = ndarray.array(value, self.ctx)
                elif value.ctx != self.ctx:
                    new_value = ndarray.empty(value.shape, self.ctx)
                    value.copyto(new_value)
                    value = new_value
                self.tensor_value = value
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu

    def reshape_in_mp(self, cur_part, parts):
        if self.reshaped:
            return
        if self.shape is None:
            # TODO: support reshape in input nodes
            return
        self.reshaped = True
        # this function only used in context launch to enable variable initialized in model parallel
        ori_shape = list(self.shape)
        for i, pts in enumerate(parts):
            assert ori_shape[i] % pts == 0
            ori_shape[i] //= pts
        self.shape = tuple(ori_shape)
        if self.initializer is not None:
            self.initializer.shape = self.shape
        elif self.tensor_value is not None:
            slcs = []
            for i in range(len(ori_shape)):
                st = ori_shape[i] * cur_part[i]
                en = st + ori_shape[i]
                slcs.append(slice(st, en))
            ori_value = self.tensor_value
            if not isinstance(ori_value, np.ndarray):
                ori_value = ori_value.asnumpy()
            ori_value = ori_value[tuple(slcs)]
            assert ori_value.shape == self.shape
            self.tensor_value = ori_value


def placeholder_op(name, value=None, initializer=None, trainable=True, dtype=np.float32, ctx=None):
    """Node of variable placeholder.

    Parameters:
    ----
    None

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PlaceholderOp(name, value, initializer, trainable, dtype, ctx)
