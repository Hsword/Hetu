from .base import BaseLayer
import hetu as ht


class Concatenate(BaseLayer):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return ht.concatenate_op(args, axis=self.axis)


class ConcatenateLayers(BaseLayer):
    def __init__(self, layers, axis=0):
        self.layers = layers
        self.axis = axis

    def __call__(self, x):
        if len(self.layers) == 1:
            return self.layers[0](x)
        else:
            return ht.concatenate_op([layer(x) for layer in self.layers], axis=self.axis)
