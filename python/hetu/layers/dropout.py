from .base import BaseLayer
import hetu as ht


class DropOut(BaseLayer):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if self.p == 0:
            return x
        return ht.dropout_op(x, 1-self.p)
