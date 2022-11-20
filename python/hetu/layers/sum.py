from .base import BaseLayer
import hetu as ht


class SumLayers(BaseLayer):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, xs):
        if not isinstance(xs, list):
            xs = [xs] * len(self.layers)
        assert len(xs) == len(self.layers)
        if len(self.layers) == 1:
            return self.layers[0](xs[0])
        else:
            return ht.sum_op([layer(x) for layer, x in zip(self.layers, xs)])
