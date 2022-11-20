from .base import BaseLayer
import hetu as ht


class Slice(BaseLayer):
    def __init__(self, begin, size):
        self.begin = begin
        self.size = size

    def __call__(self, x):
        return ht.slice_op(x, self.begin, self.size)
