from __future__ import absolute_import
from .Node import Op


class DispatchOp(Op):
    def __init__(self, node, parts):
        super().__init__(DispatchOp, [node], None)
        self.parts = parts

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, "This Op should be replaced in preprocessing phase."

    def gradient(self, output_grad):
        return [dispatch_gradient(output_grad, self.inputs[0])]

    def infer_shape(self, input_shapes):
        assert False, "This Op should be replaced in preprocessing phase."


class DispatchGradientOp(Op):
    def __init__(self, node, forward_input):
        super().__init__(DispatchGradientOp, [node, forward_input], None)

    def compute(self, input_vals, output_val, stream_handle=None):
        assert False, "This Op should be replaced in preprocessing phase."

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert False, "This Op should be replaced in preprocessing phase."


def dispatch(node, parts={}):
    """Dispatch a node into several parts, so the nodes following up can use model parallel.

    Parameters:
    ----
    node : Node
        The input Node.
    parts: tuple
        Indicates number of partitions in each dimension.
    Returns:
    ----
    A new Node instance created by Op.

    """
    return DispatchOp(node, parts)


def dispatch_gradient(node, forward_input):
    """Gradient node for Dispatch.

    Parameters:
    ----
    node : Node
        The input Node.
    forward_input: Node
        The original input node in forward phase.
    Returns:
    ----
    A new Node instance created by Op.

    """
    return DispatchGradientOp(node, forward_input)
