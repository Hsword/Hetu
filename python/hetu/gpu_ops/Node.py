from __future__ import absolute_import, annotations
from .. import ndarray
from .. import stream
from ..context import get_current_context, DeviceGroup
from copy import copy, deepcopy
import numpy as np
from typing import TYPE_CHECKING
G_NODE_ID = 0

if TYPE_CHECKING:
    from ..ndarray import DLContext, NDArray, ND_Sparse_Array, IndexedSlices
    from ..stream import Event, PSEvent, Stream
    from ..context import NodeStatus
    from .DataTransfer import DataH2DOp, DataD2HOp, DataD2HSparseOp
    from .executor import HetuConfig
    from typing import Type, Optional, Tuple, Union, Dict, List
    Array = Union[NDArray, ND_Sparse_Array, IndexedSlices]


class Op(object):
    """Basic unit of the computation graph."""

    def __init__(
        self,
        op_type: Type[Op],
        inputs: List[Op],
        ctx: Union[None, DLContext, DeviceGroup] = None
    ) -> None:
        self.inputs: List[Op] = inputs
        self.raw_ctx: Optional[DeviceGroup] = get_current_context(
        ) if ctx is None else DeviceGroup(ctx)
        self.ctx: Union[None, DLContext, DeviceGroup] = ctx
        self.const_attr: Optional[int] = None
        self.dtype: Optional[Type] = np.float32
        self.inplace: bool = False
        self.lazy_execution: bool = False
        self.event: Union[None, Event, PSEvent] = None
        self.use_indexed_slices: bool = False
        self.op_type: str = op_type.__name__
        global G_NODE_ID
        self.id: int = G_NODE_ID
        G_NODE_ID = G_NODE_ID + 1
        self.name: str = self.op_type + str(self.id)

    @property
    def desc(self) -> str:
        return self.name + \
            '(' + ', '.join([inp.name for inp in self.inputs]) + ')'

    def __add__(self, other: Union[Op, int]) -> Op:
        from .AddElewise import add_op
        from .AddConst import addbyconst_op

        # here the operator does NOT specify context
        # please explicitly specify the context in gradients!!
        if isinstance(other, Op):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = addbyconst_op(self, other)
        return new_node

    def __mul__(self, other: Union[Op, int]) -> Op:
        from .MultiplyElewise import mul_op
        from .MultiplyConst import mul_byconst_op

        if isinstance(other, Op):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo: Dict) -> Op:
        # this function should be used before graph splits and hooks (in distributed strategies)
        # use copy for DeviceGroup and NDArray (shared data); use deepcopy for nodes and optimizer
        if id(self) not in memo:
            new_op = copy(self)
            memo[id(self)] = new_op
            for k, v in self.__dict__.items():
                if k in ('inputs', 'grad_nodes'):
                    new_op.__dict__[k] = [
                        deepcopy(n, memo) for n in v]
                elif k in ('grad_node', 'forward_node', 'optimizer'):
                    new_op.__dict__[k] = deepcopy(v, memo)
        return memo[id(self)]

    def compute(
        self,
        input_vals: List[Array],
        output_val: Array,
        stream_handle: Optional[Stream] = None
    ) -> None:
        """Given values of input nodes, compute the output value.
        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        """
        raise NotImplementedError

    def gradient(
        self,
        output_grad: Op,
    ) -> List[Op]:
        """Given output gradient, compute partial gradient to each input node.
        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions
        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(
        self,
        input_shapes: List[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        """Given shapes of input nodes, compute shape of output node.
        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.
        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.
        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError

    def naive_infer_shape(
        self,
        input_shapes: List[Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        return self.infer_shape(input_shapes)

    def add_transfer_op(
        self,
        src_node: Op,
        dst_ctx: DLContext,
        h2d_ops: Dict[Op, DataH2DOp],
        d2h_ops: Dict[Op, Union[DataD2HOp, DataD2HSparseOp]],
    ) -> Op:
        from .DataTransfer import datah2d_op, datad2h_op, datah2d_sparse_op, datad2h_sparse_op

        def add_h2d(prev_node: Op, cur_ctx: DLContext) -> DataH2DOp:
            if prev_node not in h2d_ops:
                if prev_node.use_indexed_slices:
                    h2d_ops[prev_node] = datah2d_sparse_op(prev_node, cur_ctx)
                else:
                    h2d_ops[prev_node] = datah2d_op(prev_node, cur_ctx)
            return h2d_ops[prev_node]

        def add_d2h(prev_node: Op) -> Union[DataD2HOp, DataD2HSparseOp]:
            if prev_node not in d2h_ops:
                if prev_node.use_indexed_slices:
                    d2h_ops[prev_node] = datad2h_sparse_op(prev_node)
                else:
                    d2h_ops[prev_node] = datad2h_op(prev_node)
                if prev_node.event is None:
                    # here we should ensure the computation complete before d2h
                    prev_node.event = stream.create_event_handle(prev_node.ctx)
            return d2h_ops[prev_node]
        src_ctx = src_node.ctx
        result = src_node
        if src_ctx != dst_ctx:
            if ndarray.is_gpu_ctx(dst_ctx):
                if ndarray.is_gpu_ctx(src_ctx):
                    assert False, 'Please use NCCL to P2P communicate!'
                else:
                    result = add_h2d(result, dst_ctx)
            else:
                result = add_d2h(result)
        return result

    def forward_hook(self, config: HetuConfig) -> None:
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        if not self.lazy_execution:
            for node in self.inputs:
                if node.op_type not in ["Array_ReshapeOp", "Array_Reshape_GradientOp"]:
                    node.inplace = False

        # insert data transfer op if needed
        input_ctxs = set([n.ctx for n in self.inputs])
        assert None not in input_ctxs, 'Inputs contexts should already be determined.'
        if self.ctx is None:
            self.ctx = config.context
        elif isinstance(self.ctx, DeviceGroup):
            self.ctx = self.ctx.get_only()
        for i in range(len(self.inputs)):
            self.inputs[i] = self.add_transfer_op(
                self.inputs[i], self.ctx, config.h2d_ops, config.d2h_ops)
        self.on_gpu = ndarray.is_gpu_ctx(self.ctx)
        self.on_cpu = not self.on_gpu
        if self in config.eval_node_list and self.on_gpu and self.event is None:
            self.event = stream.create_event_handle(self.ctx)

    def backward_hook(self, config: HetuConfig) -> None:
        pass

    def forward_deduce_states(
        self,
        input_statuses: List[NodeStatus],
        status: NodeStatus,
        deduce_order: bool
    ) -> None:
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            status.copy_from(nst, deduce_order)

    def backward_deduce_states(
        self,
        status: NodeStatus,
        input_statuses: List[NodeStatus],
        deduce_order: bool
    ) -> None:
        assert len(input_statuses) == len(self.inputs)
        for nst in input_statuses:
            nst.copy_from(status, deduce_order)

    def deduce_generated_backward_nodes_states(self, input_statuses: List[NodeStatus], status: NodeStatus, index: Optional[int]):
        if index == -1:
            return status.remove_partial()
        else:
            return input_statuses[index].remove_partial()

    def enable_distributed_partial(self) -> bool:
        if not hasattr(self, '_enable_partial'):
            # node with bias has different output if using partial!!!!!
            # TODO: solve the problem above
            from .MatrixMult import MatMulOp
            from .Linear import LinearOp
            from .Conv2d import Conv2dOp, Conv2d_Gradient_of_DataOp, Conv2d_Gradient_of_FilterOp
            from .Conv2dAddBias import Conv2dAddBiasOp
            from .ReduceMean import ReduceMeanOp
            from .ReduceSum import ReduceSumOp
            from .EmbeddingLookUp import EmbeddingLookUp_Gradient, EmbeddingLookUp
            from .LayerNorm import Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp
            from .BatchNorm import Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp
            from .OnesLike import OnesLikeOp
            from .Division import DivOp, DivConstOp  # only for bert, the loss is divided
            from .AddElewise import AddOp  # only for bert, the loss is added
            from .MultiplyElewise import MulOp  # only for bert, the loss is multiplied
            from .Sum import SumOp
            from .SoftmaxCrossEntropySparse import SoftmaxCrossEntropySparseOp
            self._enable_partial = isinstance(self, (
                MatMulOp, LinearOp, Conv2dOp, Conv2d_Gradient_of_DataOp,
                Conv2d_Gradient_of_FilterOp, Conv2dAddBiasOp,
                ReduceMeanOp, ReduceSumOp, EmbeddingLookUp_Gradient, EmbeddingLookUp, SoftmaxCrossEntropySparseOp,
                Layer_Normalization_Gradient_of_ScaleOp, Layer_Normalization_Gradient_of_BiasOp,
                Batch_Normalization_Gradient_of_ScaleOp, Batch_Normalization_Gradient_of_BiasOp,
                OnesLikeOp, SumOp, DivOp, DivConstOp, AddOp, MulOp))
        return self._enable_partial

    def reset_status(self) -> None:
        # reset ori_status, tar_status, grad_set, etc.
        pass
