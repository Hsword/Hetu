from .base import BaseLayer
import hetu as ht


class Relu(BaseLayer):
    def __call__(self, x):
        return ht.relu_op(x)
