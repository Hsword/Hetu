from ..gpu_ops.Node import Op


class BaseLayer(object):
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError
