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
            assert isinstance(value, (np.ndarray, ndarray.NDArray, ndarray.ND_Sparse_Array)),\
                'Value data type %s not valid.' % str(type(value))
            self.shape = value.shape
            if isinstance(value, (np.ndarray, ndarray.NDArray)):
                assert value.dtype == dtype
            else:
                assert dtype == np.float32
        else:
            assert initializer is not None, 'Value not specified, initializer should not be None.'
            self.shape = initializer.shape
        self.tensor_value = value
        self.initializer = initializer
        self.trainable = trainable
        self.dtype = dtype
        self.reshaped = False
        self.embedding_offsets = None

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
                if self.is_embed:
                    # save initializer and seed for possible further use
                    self.used_initializer = self.initializer
                    from ..random import get_seed_status
                    self.init_seed = get_seed_status()
                self.initializer(self, config.comp_stream)
                self.initializer = None
            elif self.tensor_value is not None:
                value = self.tensor_value
                assert isinstance(value, (np.ndarray, ndarray.NDArray, ndarray.ND_Sparse_Array)), \
                    'Parameters should be initialized as numpy.ndarray or ndarray.NDArray .'
                if isinstance(value, np.ndarray):
                    value = ndarray.array(value, self.ctx, dtype=self.dtype)
                elif value.ctx != self.ctx:
                    assert not isinstance(value, ndarray.ND_Sparse_Array)
                    new_value = ndarray.empty(
                        value.shape, self.ctx, dtype=self.dtype)
                    value.copyto(new_value)
                    value = new_value
                self.tensor_value = value
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu

    def reshape_in_mp(self, cur_part, parts):
        self.cur_part = cur_part
        self.parts = parts
        if self.reshaped:
            return
        self.reshaped = True
        if self.shape is None:
            return
        # this function only used in context launch to enable variable initialized in model parallel
        ori_shape = list(self.shape)
        for i, pts in parts.items():
            assert ori_shape[i] % pts == 0
            ori_shape[i] //= pts
        self.shape = tuple(ori_shape)
        if self.initializer is not None:
            self.initializer.shape = self.shape
        elif self.tensor_value is not None:
            self.tensor_value = self.reshape_tensor(self.tensor_value)
            assert self.shape == self.tensor_value.shape

    def reshape_tensor(self, tensor):
        if self.embedding_offsets is not None:
            offset, length = self.embedding_offsets
            tensor -= offset
            tensor[tensor < 0] = -1
            tensor[tensor >= length] = -1
        if self.parts == {}:
            return tensor
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.asnumpy()
        ori_shape = tensor.shape
        slcs = []
        for i in range(len(ori_shape)):
            if i in self.parts:
                part = ori_shape[i] // self.parts[i]
                st = part * self.cur_part[i]
                en = st + part
            else:
                st = 0
                en = ori_shape[i]
            slcs.append(slice(st, en))
        tensor = tensor[tuple(slcs)]
        return tensor


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
