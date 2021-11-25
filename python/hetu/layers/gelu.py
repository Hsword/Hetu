from .base import BaseLayer
import hetu as ht


class Gelu(BaseLayer):
    def __call__(self, x):
        return ht.gelu_op(x)