from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from .._base import DNNL_LIB
import numpy as np
from ..cpu_links import embedding_lookup as cpu_embedding_lookup
from ..gpu_links import embedding_lookup


class EmbeddingLookUp(Op):
    def __init__(self, embedding, index, ctx=None):
        super().__init__(EmbeddingLookUp, [embedding, index], ctx)
        embedding.is_embed = True

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
        self.grad_node = embedding_lookup_gradient_op(
            output_grad, self.inputs[1], None, ctx=self.inputs[0].ctx)
        return [self.grad_node, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        if hasattr(self, 'grad_node'):
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
        assert local_comm_mode == config.node_strategy.get(self.inputs[0], config.comm_mode), \
            'Embedding lookup communication mode invalid. Should conform with embedding parameter.'
        if local_comm_mode in ('PS', 'Hybrid'):
            cpu_ctx = ndarray.cpu(0)
            self.ctx = cpu_ctx
            for n in self.inputs:
                n.ctx = cpu_ctx


class EmbeddingLookUp_Gradient(Op):
    def __init__(self, vectors, index, embed_shape, ctx=None):
        super().__init__(EmbeddingLookUp_Gradient, [vectors, index], ctx)
        self.embed_shape = embed_shape

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.embed_shape
        output_val.update(
            values=input_vals[0], indices=input_vals[1], dense_shape=self.embed_shape)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert self.embed_shape
        return self.embed_shape

    def backward_hook(self, config):
        # insert data transfer op if needed
        if config.comm_mode == 'PS' or config.comm_mode == "Hybrid":
            self.ctx = ndarray.cpu(0)


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
