from .base import BaseLayer
import hetu as ht


class SumLayers(BaseLayer):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        if len(self.layers) == 1:
            return self.layers[0](x)
        else:
            return ht.sum_op([layer(x) for layer in self.layers])
