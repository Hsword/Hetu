from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
import numpy as np
from ..gpu_links import robe_hash, mod_hash, compo_hash, learn_hash
from random import randrange

class RobeHashOp(Op):
    def __init__(self, node, roarsz, ctx=None):
        super().__init__(RobeHashOp, [node], ctx)
        self.roarsz = roarsz #roarsz means ROBE array size
        self.Bh=623987128#randrange(998244353)
        self.Ch=423878919#randrange(998244353)
        self.MO=998244353
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = ( ( np.array(
                input_vals[0].asnumpy(), dtype=np.int32) * self.Bh + self.Ch) % self.MO + self.MO) % self.MO % self.roarsz
        else:
            robe_hash(input_vals[0], output_val, self.roarsz, self.Bh, self.Ch, self.MO, stream_handle)

    def gradient(self, output_grad):
        return [None]
    
    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

class ModHashOp(Op):
    def __init__(self, node, nembed, ctx=None):
        super().__init__(ModHashOp, [node], ctx)
        self.nembed = nembed
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.array(
                input_vals[0].asnumpy(), dtype=np.int32) % self.nembed
        else:
            mod_hash(input_vals[0], output_val, self.nembed, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class CompoHashOp(Op):
    def __init__(self, node, ntable, nembed, ctx=None):
        super().__init__(CompoHashOp, [node], ctx)
        self.ntable = ntable
        self.nembed = nembed
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            x = np.array(input_vals[0].asnumpy(), dtype=np.int32)
            results = []
            for i in range(self.ntable - 1):
                results.append(x % self.nembed)
                x //= self.nembed
            results.append(x)
            output_val[:] = np.stack(results, axis=-1)
        else:
            compo_hash(input_vals[0], output_val,
                       self.ntable, self.nembed, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        output_shape = list(input_shapes[0])
        output_shape.append(self.ntable)
        return tuple(output_shape)


class LearnHashOp(Op):
    def __init__(self, node, slope, bias, prime, nbucket, dist, ctx=None):
        assert dist in ['uniform', 'normal']
        super().__init__(LearnHashOp, [node, slope, bias, prime], ctx)
        self.nbucket = nbucket
        self.dist = dist
        self.eps = 1e-12

    def compute(self, input_vals, output_val, stream_handle=None):
        num_hash = input_vals[1].shape[0]
        if self.on_cpu:
            x = np.expand_dims(
                np.array(input_vals[0].asnumpy(), dtype=np.int32), -1)
            results = (input_vals[1].asnumpy(
            ) * x + input_vals[2].asnumpy()) % input_vals[3].asnumpy() % self.nbucket
            scale_pos = results / (self.nbucket - 1)
            scale_both = scale_pos * 2 - 1
            if self.dist != 'uniform':
                i = 0
                while i < num_hash:
                    j = i + 1
                    left_content = np.sqrt(-2 *
                                           np.log(np.maximum(scale_pos[..., i], self.eps)))
                    right_content = 2 * np.pi * scale_pos[..., j]
                    scale_both[..., i] = left_content * np.cos(right_content)
                    scale_both[..., j] = left_content * np.sin(right_content)
                    i = i + 2
            output_val[:] = scale_both
        else:
            normal = (self.dist == 'normal')
            learn_hash(input_vals[0], input_vals[1], input_vals[2], input_vals[3],
                       output_val, self.nbucket, normal, self.eps, stream_handle)

    def gradient(self, output_grad):
        return [None, None, None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 4
        assert input_shapes[1] == input_shapes[2] == input_shapes[3]
        output_shape = list(input_shapes[0])
        output_shape.append(input_shapes[1][0])
        return tuple(output_shape)

def robe_hash_op(node, roarsz, ctx=None):
    return RobeHashOp(node, roarsz, ctx=ctx)


def mod_hash_op(node, nembed, ctx=None):
    return ModHashOp(node, nembed, ctx=ctx)


def compo_hash_op(node, ntable, nembed, ctx=None):
    return CompoHashOp(node, ntable, nembed, ctx=ctx)


def learn_hash_op(node, slope, bias, prime, nbucket, dist, ctx=None):
    return LearnHashOp(node, slope, bias, prime, nbucket, dist, ctx=ctx)
