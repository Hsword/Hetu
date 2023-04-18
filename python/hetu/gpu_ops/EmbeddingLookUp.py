from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
import numpy as np
from ..cpu_links import \
    embedding_lookup as cpu_embedding_lookup, \
    reduce_indexedslice as cpu_reduce_indexedslice
from ..gpu_links import embedding_lookup, \
    reduce_indexedslice_get_workspace_size, \
    reduce_indexedslice


class EmbeddingLookUp(Op):
    def __init__(self, embedding, index, ctx=None):
        super().__init__(EmbeddingLookUp, [embedding, index], ctx)
        embedding.is_embed = True
        self.grad_node = None
        self.dtype = embedding.dtype
        assert index.dtype == np.int32

    def _compute_cpu_dnnl(self, input_vals, output_val, stream_handle=None):
        cpu_embedding_lookup(input_vals[0], input_vals[1], output_val)

    def _compute_cpu_numpy(self, input_vals, output_val, stream_handle=None):
        flatten_index = input_vals[1].asnumpy().reshape(-1).astype(np.int32)
        output_val[:] = input_vals[0].asnumpy(
        )[flatten_index].reshape(output_val.shape)

    def _compute_gpu(self, input_vals, output_val, stream_handle=None):
        embedding_lookup(input_vals[0], input_vals[1],
                         output_val, stream_handle)

    def _compute_sparsepull_from_ps(self, input_vals, output_val, stream_handle=None):
        self.event.sync()
        if self.bsp == 0:
            self.comm.BarrierWorker()
        self.comm.SparsePull(
            self.ps_id, input_vals[1].handle, output_val.handle)
        self.event.update()

    def _compute_sparsepull_from_cache(self, input_vals, output_val, stream_handle=None):
        self.event.sync()
        if self.bsp == 0:
            self.comm.BarrierWorker()
        ts = self.inputs[0].cache.embedding_lookup(input_vals[1], output_val)
        self.event.update_ts(ts)

    def gradient(self, output_grad):
        # both is acceptable for normal embeddings;
        # opt op for quantize and autodim
        from .Unique import unique_indices_op, unique_indices_offsets_op, deduplicate_lookup_op, deduplicate_grad_op
        unique = unique_indices_op(self.inputs[1], ctx=self.raw_ctx)
        idoffsets = unique_indices_offsets_op(unique, ctx=self.raw_ctx)
        deduplookup = deduplicate_lookup_op(self, idoffsets, ctx=self.raw_ctx)
        dedupgrad = deduplicate_grad_op(
            output_grad, idoffsets, ctx=self.raw_ctx)
        # self.grad_node = embedding_lookup_gradient_op(
        #     output_grad, self.inputs[1], None, ctx=self.raw_ctx)
        return [(unique, deduplookup, dedupgrad), None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        if self.grad_node is not None:
            self.grad_node.embed_shape = input_shapes[0]
        output_shape = list(input_shapes[1])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)

    def forward_hook(self, config):
        super().forward_hook(config)
        # insert data transfer op if needed
        if config.use_sparse_pull or config.cstable_policy:
            self.event = self.inputs[0].event
            if not config.prefetch:
                self.bsp = config.bsp
                self.comm = config.ps_comm
                if config.cstable_policy:
                    self.compute = self._compute_sparsepull_from_cache
                else:
                    self.ps_id = self.inputs[0].id
                    self.compute = self._compute_sparsepull_from_ps
        else:
            if self.on_cpu and DNNL_LIB['cpu_EmbeddingLookup']:
                self.compute = self._compute_cpu_dnnl
            elif self.on_cpu:
                self.compute = self._compute_cpu_numpy
            else:
                self.compute = self._compute_gpu

    def backward_hook(self, config):
        # insert data transfer op if needed
        local_comm_mode = config.node_strategy.get(self, config.comm_mode)
        embedding_comm_mode = config.node_strategy.get(
            self.inputs[0], config.comm_mode)
        assert local_comm_mode in (embedding_comm_mode, None), \
            'Embedding lookup communication mode invalid. Should conform with embedding parameter.'
        if local_comm_mode in ('PS', 'Hybrid'):
            cpu_ctx = ndarray.cpu(0)
            self.ctx = cpu_ctx
            for n in self.inputs:
                n.ctx = cpu_ctx

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        tb_st, id_st = input_statuses
        if tb_st.valid(deduce_order) and id_st.valid(deduce_order):
            if deduce_order:
                lorder = tb_st.order
                rorder = id_st.order
                partial_index = lorder.index(0) if 0 in lorder else None
                split_index = rorder.index(0) if 0 in rorder else None
                if partial_index == 0:
                    new_order = [-2]
                elif split_index == 0:
                    new_order = [0]
                else:
                    new_order = []
                if status.duplicate > 1:
                    new_order.append(-1)
                if partial_index == 1:
                    new_order.append(-2)
                elif split_index == 1:
                    new_order.append(0)
                status.set_order(tuple(new_order))
            else:
                partial = tb_st.state.get(0, 1)
                new_state = {0: id_st.state.get(0, 1)}
                status.set_state(new_state, None, partial)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        # now we only support split batch dimension
        # TODO: enable dynamic dimension to express the last dimension in state/order
        assert len(input_statuses) == len(self.inputs)
        if status.valid(deduce_order):
            status.check_state(1, deduce_order)
        if deduce_order:
            if status.valid_all():
                input_statuses[0].set_order(
                    status.combine_order((0, -1), (-2, 0)))
                input_statuses[1].set_order(status.combine_order((-2, -1)))
        else:
            if status.valid_state():
                input_statuses[0].set_state(
                    *status.combine_state((0, -1), (-2, 0)))
                input_statuses[1].set_state(*status.combine_state((-2, -1)))

    def deduce_generated_backward_nodes_states(self, input_statuses, status, index):
        assert index <= 0
        if index == -1:
            return status.remove_partial()
        else:
            from ..context import NodeStatus
            new_status = NodeStatus(
                dev_num=status.dev_num, partial_or_node=True)
            new_status.set_state(*status.exchange_state(-2, 0))
            new_status.set_order(status.exchange_order(-2, 0))
            return new_status


class EmbeddingLookUp_Gradient(Op):
    def __init__(self, vectors, index, embed_shape, ctx=None):
        inputs = [vectors]
        if isinstance(index, Op):
            inputs.append(index)
            self.index = None
        else:
            self.index = index
        super().__init__(EmbeddingLookUp_Gradient,
                         inputs, ctx)
        self.embed_shape = embed_shape
        self.use_indexed_slices = True
        self.dedup_args = None

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.index is None:
            index = input_vals[1]
        else:
            index = self.index
        if self.on_gpu:
            reduce_indexedslice(index, input_vals[0], output_val.indices, output_val.values,
                                self.dedup_args['sp'], self.dedup_args['size'], self.dedup_args['eb'], stream_handle)
        else:
            cpu_reduce_indexedslice(
                index, input_vals[0], output_val.indices, output_val.values)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert self.embed_shape
        if self.on_gpu:
            if self.index is None:
                ind_shape = input_shapes[1]
            else:
                ind_shape = self.index.shape
            ind_size = int(np.prod(ind_shape))
            ws_size = reduce_indexedslice_get_workspace_size(ind_size)
            all_ws_size = 2 * ind_size + 2 + (ws_size + 3) // 4
            self.dedup_args = {
                'sp': ndarray.empty((all_ws_size, ), ctx=self.ctx),
                'size': ws_size,
                'eb': 32,
            }
        else:
            self.dedup_args = {}
        return self.embed_shape

    def backward_hook(self, config):
        # insert data transfer op if needed
        if config.comm_mode == 'PS' or config.comm_mode == "Hybrid":
            self.ctx = ndarray.cpu(0)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if status.valid(deduce_order):
            input_statuses[0].set_from_combine(
                status, deduce_order, (0, -1), (-2, 0))
            input_statuses[1].set_from_combine(
                status, deduce_order, (0, -1), (-2, 0))
        elif input_statuses[0].valid(deduce_order):
            input_statuses[1].copy_from(input_statuses[0], deduce_order)
        elif input_statuses[1].valid(deduce_order):
            input_statuses[0].copy_from(input_statuses[1], deduce_order)


def embedding_lookup_op(embedding, index, ctx=None):
    """Make a new instance of EmbeddingLookUp and call the instance.

    Parameters:
    ----
    embedding : Node
        The Node of Embedding.
    index : Node
        The index to be looked up.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return EmbeddingLookUp(embedding, index, ctx=ctx)


def embedding_lookup_gradient_op(vectors, index, embed_shape, ctx=None):
    """Make a new instance of EmbeddingLookUp_Gradient and call the instance.

    Parameters:
    ----
    vectors : Node
        Vectors which looked up from Embedding.
    index : Node
        The index to be looked up.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return EmbeddingLookUp_Gradient(vectors, index, embed_shape, ctx=ctx)
