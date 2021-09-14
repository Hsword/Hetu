from .base import BaseLayer


class Sequence(BaseLayer):
    def __init__(self, *args):
        self.layers = args

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
