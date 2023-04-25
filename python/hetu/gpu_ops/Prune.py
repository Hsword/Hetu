
from __future__ import absolute_import
import numpy as np

from hetu.ndarray import empty

from .Node import Op
from ..gpu_links import num_less_than, set_less_than


class PruneLowMagnitudeOp(Op):
    def __init__(self, node, rate, ctx=None):
        super().__init__(PruneLowMagnitudeOp, [node], ctx)
        self.rate_updater = rate
        self.niter = 0

    def compute(self, input_vals, output_val, stream_handle=None):
        input_val = input_vals[0]
        if self.on_cpu:
            raise NotImplementedError
        else:
            # get threshold by binary search
            self.niter += 1
            cur_rate = self.rate_updater(self.niter)
            if cur_rate <= 0:
                return
            l, r = 0., 1e2
            cnt = 0
            while l < r:
                cnt += 1
                mid = (l + r) / 2
                num_less_than(input_val, self.buffer, self.output,
                              mid, self.axes, stream_handle)
                stream_handle.sync()
                sparse_items = self.output.asnumpy()[0]
                sparse_rate = sparse_items / self.nparam
                if abs(sparse_rate - cur_rate) < 1e-3 * (1 - cur_rate):
                    break
                elif sparse_rate > cur_rate:
                    r = mid
                else:
                    l = mid
                if cnt > 100:
                    break
            # prune param with lower magnitude than mid
            set_less_than(input_val, mid, stream_handle)
        return mid

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        input_shape = input_shapes[0]
        self.nparam = np.prod(input_shape, dtype=int)
        self.buffer = empty(input_shape, ctx=self.ctx)
        self.output = empty((1,), ctx=self.ctx)
        self.axes = tuple(range(len(input_shape)))
        return None


def prune_low_magnitude_op(node, rate, ctx=None):
    return PruneLowMagnitudeOp(node, rate, ctx=ctx)
