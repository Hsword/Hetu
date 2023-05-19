from __future__ import absolute_import
from .Node import Op
import numpy as np
from ..gpu_links import robe_hash, robe_sign, mod_hash, mod_hash_negative, div_hash, compo_hash, learn_hash


class RobeHashOp(Op):
    def __init__(self, indices, rands, length, dim, Z, use_slot_coef, ctx=None):
        super().__init__(RobeHashOp, [indices, rands], ctx)
        self.length = length
        self.dim = dim
        self.Z = Z
        assert self.dim % self.Z == 0
        self.use_slot_coef = use_slot_coef
        self.dtype = np.int32
        assert indices.dtype == np.int32 and rands.dtype == np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            # (Ah*e + Bh*x + Ch*c + Dh) mod P mod |M|
            random_numbers = input_vals[1].asnumpy()
            # Bh*x + Dh
            result = random_numbers[3].astype(np.int64) * \
                input_vals[0].asnumpy() + random_numbers[1]
            if self.use_slot_coef:
                # Ah*e
                slot_offset = np.arange(
                    input_vals[0].shape[-1], dtype=np.int64)
                result = result + random_numbers[4] * slot_offset
            Z_offset = random_numbers[2] * np.arange(self.Z, dtype=np.int64).repeat(
                self.dim // self.Z)
            # Ch*c
            result = result[..., None] + Z_offset
            result = result + \
                np.arange(
                    self.dim // self.Z)[None, ...].repeat(self.Z, 0).reshape(-1)
            # mod P mod |M|
            result = result % random_numbers[0] % self.length
            output_val[:] = result.astype(np.int32)
        else:
            robe_hash(input_vals[0], input_vals[1], output_val, self.length,
                      self.dim, self.Z, self.use_slot_coef, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2 and len(input_shapes[0]) == 2
        output_shape = tuple(input_shapes[0]) + (self.dim,)
        return output_shape


class RobeSignOp(Op):
    def __init__(self, indices, rands, dim, use_slot_coef, ctx=None):
        super().__init__(RobeSignOp, [indices, rands], ctx)
        self.dim = dim
        self.use_slot_coef = use_slot_coef
        assert indices.dtype == np.int32 and rands.dtype == np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            # ((Ag*e + Bg*x + Cg*i + Dg) mod P mod 2) * 2 - 1
            random_numbers = input_vals[1].asnumpy()
            # Bg*x + Dg
            result = random_numbers[7].astype(np.int64) * \
                input_vals[0].asnumpy() + random_numbers[5]
            if self.use_slot_coef:
                # Ag*e
                slot_offset = np.arange(
                    input_vals[0].shape[-1], dtype=np.int64)
                result = result + random_numbers[8] * slot_offset
            # Cg*i
            result = result[..., None] + random_numbers[6] * \
                np.arange(self.dim, dtype=np.int64)
            # mod P mod 2
            result = result % random_numbers[0] % 2
            result = result * 2 - 1
            output_val[:] = result.astype(np.float32)
        else:
            robe_sign(input_vals[0], input_vals[1], output_val,
                      self.dim, self.use_slot_coef, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2 and len(input_shapes[0]) == 2
        output_shape = tuple(input_shapes[0]) + (self.dim,)
        return output_shape


class ModHashOp(Op):
    def __init__(self, node, nembed, ctx=None):
        super().__init__(ModHashOp, [node], ctx)
        self.nembed = nembed
        assert node.dtype == np.int32
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


class ModHashNegativeOp(Op):
    def __init__(self, node, nembed, ctx=None):
        super().__init__(ModHashNegativeOp, [node], ctx)
        self.nembed = nembed
        assert node.dtype == np.int32
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            values = input_vals[0].asnumpy()
            values = -(values + 1)
            assert values.dtype == np.int32
            is_positive = (values >= 0)
            positive_parts = values[is_positive] % self.nembed
            values[is_positive] = positive_parts
            output_val[:] = values
        else:
            mod_hash_negative(
                input_vals[0], output_val, self.nembed, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class DivHashOp(Op):
    def __init__(self, node, nembed, ctx=None):
        super().__init__(DivHashOp, [node], ctx)
        self.nembed = nembed
        assert node.dtype == np.int32
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.array(
                input_vals[0].asnumpy(), dtype=np.int32) // self.nembed
        else:
            div_hash(input_vals[0], output_val, self.nembed, stream_handle)

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
        assert node.dtype == np.int32
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
        assert node.dtype == slope.dtype == bias.dtype == prime.dtype == np.int32

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


def robe_hash_op(indices, rands, length, dim, Z, use_slot_coef=True, ctx=None):
    return RobeHashOp(indices, rands, length, dim, Z, use_slot_coef, ctx=ctx)


def robe_sign_op(indices, rands, dim, use_slot_coef=True, ctx=None):
    return RobeSignOp(indices, rands, dim, use_slot_coef, ctx=ctx)


def mod_hash_op(node, nembed, ctx=None):
    return ModHashOp(node, nembed, ctx=ctx)


def mod_hash_negative_op(node, nembed, ctx=None):
    return ModHashNegativeOp(node, nembed, ctx=ctx)


def div_hash_op(node, nembed, ctx=None):
    return DivHashOp(node, nembed, ctx=ctx)


def compo_hash_op(node, ntable, nembed, ctx=None):
    return CompoHashOp(node, ntable, nembed, ctx=ctx)


def learn_hash_op(node, slope, bias, prime, nbucket, dist, ctx=None):
    return LearnHashOp(node, slope, bias, prime, nbucket, dist, ctx=ctx)
