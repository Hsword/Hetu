from .base import BaseLayer


class Identity(BaseLayer):
    def __call__(self, x):
        return x
