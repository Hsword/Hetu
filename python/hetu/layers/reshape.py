from .base import BaseLayer
import hetu as ht


class Reshape(BaseLayer):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return ht.array_reshape_op(x, self.shape)
