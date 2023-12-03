import numpy as np
import ctypes
import hetu as ht
from .ndarray import NDArray, IndexedSlices
from .lr_scheduler import FixedScheduler
from .gpu_ops.Node import Op
from .gpu_ops.AssignWithIndexedSlices import assign_with_indexedslices_op, assign_quantized_embedding_op
from .gpu_ops.ParameterServerCommunicate import ParameterServerCommunicateOp
from .gpu_ops.Variable import PlaceholderOp
from .gpu_links.OptimizerLink import sgd_update, sgd_update_indexedslices, \
    momentum_update, adagrad_update, adagrad_update_indexedslices, \
    adam_update, adam_update_indexedslices, \
    adamw_update, lamb_update, betats_update
from .cpu_links.dnnl_op import sgd_update as cpu_sgd_update, \
    sgd_update_indexedslices as cpu_sgd_update_indexedslices, \
    momentum_update as cpu_momentum_update, \
    adagrad_update as cpu_adagrad_update, \
    adagrad_update_indexedslices as cpu_adagrad_update_indexedslices,\
    adam_update as cpu_adam_update, \
    adam_update_indexedslices as cpu_adam_update_indexedslices, \
    betats_update as cpu_betats_update
from ._base import DNNL_LIB


class Optimizer(object):
    """Optimizers."""

    def __init__(self, learning_rate, l2reg=0):
        if isinstance(learning_rate, FixedScheduler):
            self.lr_sched = learning_rate
        else:
            assert learning_rate >= 0, \
                "learning rate must be non-negative"
            self.lr_sched = FixedScheduler(learning_rate)
        # now we don't support l2 regularizer for PS mode parameters
        # TODO: support l2 regularizer for PS mode parameters (after PS mode has optimizer on Servers)
        assert l2reg >= 0, 'L2 regularizer should be positive or 0.'
        self.l2reg = l2reg
        self.opt_op_type = None
        self.sparse_opt_op_type = None

    @property
    def learning_rate(self):
        return self.lr_sched.get()

    @staticmethod
    def get_var_list(loss):
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            if (isinstance(node, PlaceholderOp) and node.trainable):
                var_list.append(node)
            for n in node.inputs:
                topo_sort_dfs(n, visited, var_list)

        visited = set()
        trainable_vars = []
        if isinstance(loss, list):
            for l in loss:
                topo_sort_dfs(l, visited, trainable_vars)
        else:
            topo_sort_dfs(loss, visited, trainable_vars)
        return trainable_vars

    def minimize(self, loss, var_list=None):
        """Return optimizer ops to update parameters.

        Parameters
        ----------
        loss: loss node that we are minimizing.
        var_list: list of nodes that we are taking derivative wrt.

        Returns
        -------
        A list of optimizer nodes.

        """
        self.loss = loss
        if not var_list:
            var_list = self.get_var_list(loss)
        grads, self.backward2forward, self.forward2backward = ht.gradients(
            loss, var_list, return_all=True)
        opt_nodes = []
        if isinstance(self, (AdamOptimizer, AdamWOptimizer, LambOptimizer)):
            names = {}
            gidx = 0
            for p in var_list:
                pctx = p.ctx
                if pctx not in names:
                    names[pctx] = gidx
                    gidx += 1
            self.betatss = {ctx: _raw_init_states(
                (2,), f'betats_{names[ctx]}', ctx, ht.init.constant, fill_value=1.0) for ctx in names}
            self.betats_update_ops = {ctx: betats_update_op(
                betats, self.beta1, self.beta2, ctx) for ctx, betats in self.betatss.items()}
        for param, grad in zip(var_list, grads):
            if isinstance(grad, tuple):
                assert param.is_embed
                unique, deduplookup, dedupgrad = grad
                opt_op = self.sparse_opt_op_type(
                    self, param, unique, deduplookup, dedupgrad)
                if param.dtype != np.float32:
                    lookup = deduplookup.inputs[0]
                    from .gpu_ops.QuantizeEmbedding import QuantizedEmbeddingLookUpOp, UnifiedQuantizedEmbeddingLookUpOp
                    from .gpu_ops.QuantizeALPTEmb import ALPTEmbeddingLookUpOp
                    if isinstance(lookup, UnifiedQuantizedEmbeddingLookUpOp):
                        assign_op = assign_quantized_embedding_op(
                            param, unique, opt_op, lookup.digit, scale=lookup.scale, minele=lookup.minele)
                    elif isinstance(lookup, QuantizedEmbeddingLookUpOp):
                        assign_op = assign_quantized_embedding_op(
                            param, unique, opt_op, lookup.digit, qparam=lookup.inputs[2])
                    elif isinstance(lookup, ALPTEmbeddingLookUpOp):
                        assign_op = assign_with_indexedslices_op(
                            param, unique, opt_op)
                    else:
                        assert False
                else:
                    assign_op = assign_with_indexedslices_op(
                        param, unique, opt_op)
                opt_nodes.append(assign_op)
            else:
                opt_nodes.append(self.opt_op_type(param, grad, self))
        return opt_nodes


class OptimizerOp(Op):
    def __init__(self, param, grad, optimizer, *states):
        self.optimizer = optimizer
        self.learning_rate = optimizer.learning_rate
        self.l2reg = optimizer.l2reg
        super().__init__(type(self), [param, grad] + list(states))

    def compute(self, input_vals, output_val, stream_handle=None):
        raise NotImplementedError

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        param = self.inputs[0]
        grad = self.inputs[1]
        paramctx = param.ctx
        grad.inplace = False

        self.ctx = paramctx
        self.on_gpu = param.on_gpu
        self.on_cpu = param.on_cpu
        self.comm_mode = config.comm_mode
        if self.comm_mode != 'PS':
            # Though the gradients for transfer ops are well defined,
            # we called gradients in optimizer op before transfer ops are added.
            # So here we also add tranfer ops for gradients update.
            # Could be optimized later.
            if not isinstance(grad, ParameterServerCommunicateOp):
                self.inputs[1] = super().add_transfer_op(
                    grad, paramctx, config.h2d_ops, config.d2h_ops)

    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        cur_node = self.inputs[1]
        cur_param = self.inputs[0]
        if "expert" not in cur_param.name:
            # expert parameter no use allreduce
            current_strategy = config.node_strategy.get(
                cur_param, self.comm_mode)
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not cur_node.use_indexed_slices):
                cur_node = ht.allreduceCommunicate_op(
                    cur_node, config.param_allreduce_group.get(cur_param, config.nccl_comm))
                if config.layer_indices is not None and cur_node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[cur_node] + 1
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and cur_node.use_indexed_slices):
                cur_node = ht.parameterServerCommunicate_op(
                    cur_node, cur_param, self.optimizer.get_config())
                if config.layer_indices is not None and cur_node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[cur_node] + 1
            self.inputs[1] = cur_node


class OptimizerSparseOp(Op):
    def __init__(self, optimizer, param, unique, dedup_lookup, dedup_grad, *states):
        self.optimizer = optimizer
        self.learning_rate = optimizer.learning_rate
        self.l2reg = optimizer.l2reg
        super().__init__(type(self), [
            param, unique, dedup_lookup, dedup_grad] + list(states))

    def compute(self, input_vals, output_val, stream_handle=None):
        raise NotImplementedError

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[2]


class SGDUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        super().__init__(param, grad, optimizer)

    def compute(self, input_vals, output_val, stream_handle=None):
        tensor, grad = input_vals
        if self.on_gpu:
            sgd_update(
                tensor, grad, self.learning_rate, self.l2reg, stream_handle)
        else:
            if DNNL_LIB['cpu_SGDOptimizerSparseUpdate'] and DNNL_LIB['cpu_SGDOptimizerUpdate']:
                cpu_sgd_update(
                    tensor, grad, self.learning_rate, self.l2reg)
            else:
                # not implement regularization
                if isinstance(grad, IndexedSlices):
                    np_indices = grad.indices.asnumpy()
                    np_tensor = tensor.asnumpy()
                    np_values = grad.values.asnumpy().reshape(
                        (-1, np_tensor.shape[-1]))
                    for j, ind in enumerate(np_indices.reshape(-1)):
                        if ind < 0:
                            continue
                        np_tensor[ind] -= self.learning_rate * np_values[j]
                    tensor[:] = np_tensor
                else:
                    prev_param = tensor.asnumpy()
                    grad = grad.asnumpy(
                    ) + self.l2reg * prev_param if self.l2reg > 0 else grad.asnumpy()
                    tensor[:] = prev_param - \
                        self.learning_rate * grad


class SGDSparseUpdateOp(OptimizerSparseOp):
    def __init__(self, optimizer, param, unique, dedup_lookup, dedup_grad):
        super().__init__(optimizer, param, unique, dedup_lookup, dedup_grad)

    def compute(self, input_vals, output_val, stream_handle=None):
        _, unique, dedup_lookup, dedup_grad = input_vals
        if self.on_gpu:
            sgd_update_indexedslices(
                unique, dedup_grad, dedup_lookup, output_val, self.learning_rate, stream_handle)
        else:
            if DNNL_LIB['cpu_SGDUpdateIndexedSlices']:
                cpu_sgd_update_indexedslices(
                    unique, dedup_grad, dedup_lookup, output_val, self.learning_rate)
            else:
                # not implement regularization
                output_val[:] = dedup_lookup.asnumpy(
                ) - self.learning_rate * dedup_grad.asnumpy()


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, l2reg=0):
        super(SGDOptimizer, self).__init__(learning_rate, l2reg)
        self.opt_op_type = SGDUpdateOp
        self.sparse_opt_op_type = SGDSparseUpdateOp

    def get_config(self):
        return (ctypes.c_int(0), (ctypes.c_float * 1)(self.learning_rate), ctypes.c_int(1))


def _init_states(param, name, func=None, **kargs):
    return _raw_init_states(
        param.shape,
        f'{param.name}_{name}',
        param.ctx,
        func,
        **kargs,
    )


def _raw_init_states(shape, name, ctx, func=None, **kargs):
    if func is None:
        from .initializers import zeros
        func = zeros
    state = func(
        shape,
        name=name,
        trainable=False,
        ctx=ctx,
        **kargs,
    )
    return state


class MomentumUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        velocity = _init_states(param, 'momentum_v')
        self.momentum = optimizer.momentum
        self.nesterov = optimizer.nesterov
        super().__init__(param, grad, optimizer, velocity)

    def compute(self, input_vals, output_val, stream_handle=None):
        tensor, grad, velocity = input_vals
        if self.on_gpu:
            momentum_update(tensor, grad, velocity, self.learning_rate, self.momentum,
                            self.nesterov, self.l2reg, stream_handle)
        else:
            if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                cpu_momentum_update(tensor, grad, velocity,
                                    self.learning_rate, self.momentum, self.nesterov, self.l2reg)
            else:
                if isinstance(grad, IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = tensor.asnumpy()
                    grad = grad.asnumpy(
                    ) + self.l2reg * prev_param if self.l2reg > 0 else grad.asnumpy()
                    velo = velocity.asnumpy()
                    if self.nesterov:
                        lr_grads = -self.learning_rate * grad
                        velocity[:] = self.momentum * \
                            (velo + lr_grads)
                        tensor[:] = prev_param + velo + lr_grads
                    else:
                        velocity[:] = self.momentum * \
                            velo - self.learning_rate * grad
                        tensor[:] = prev_param + velo


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, l2reg=0):
        super(MomentumOptimizer, self).__init__(learning_rate, l2reg)
        self.momentum = momentum
        self.nesterov = nesterov
        self.opt_op_type = MomentumUpdateOp

    def get_config(self):
        return (ctypes.c_int(self.nesterov + 1), (ctypes.c_float * 2)(self.learning_rate, self.momentum), ctypes.c_int(2))


class AdaGradUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        from .initializers import constant
        accum = _init_states(param, 'adagrad_accum', constant,
                             fill_value=optimizer.initial_accumulator_value)
        self.eps = optimizer.eps
        super().__init__(param, grad, optimizer, accum)

    def compute(self, input_vals, output_val, stream_handle=None):
        tensor, grad, accum = input_vals
        if self.on_gpu:
            adagrad_update(tensor, grad, accum,
                           self.learning_rate, self.eps, self.l2reg, stream_handle)
        else:
            if DNNL_LIB['cpu_AdaGradOptimizerSparseUpdate'] and DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                cpu_adagrad_update(
                    tensor, grad, accum, self.learning_rate, self.l2reg, self.eps)
            else:
                if isinstance(grad, IndexedSlices):
                    np_indices = grad.indices.asnumpy()
                    np_tensor = tensor.asnumpy()
                    np_values = grad.values.asnumpy().reshape(
                        (-1, np_tensor.shape[-1]))
                    np_acc = accum.asnumpy()
                    np_unique_indices, inverse = np.unique(
                        np_indices, return_inverse=True)
                    new_value = np.zeros(
                        (len(np_unique_indices), np_values.shape[-1]), dtype=np.float32)
                    for j, ind in enumerate(inverse):
                        new_value[ind] += np_values[j]
                    for j, ind in enumerate(np_unique_indices.reshape(-1)):
                        if ind < 0:
                            continue
                        np_acc[ind] += (new_value[j] ** 2)
                        np_tensor[ind] -= self.learning_rate * \
                            new_value[j] / \
                            (np.sqrt(np_acc[ind]) + self.eps)
                    accum[:] = np_acc
                    tensor[:] = np_tensor
                else:
                    prev_param = tensor.asnumpy()
                    grad = grad.asnumpy(
                    ) + self.l2reg * prev_param if self.l2reg > 0 else grad.asnumpy()
                    accum[:] = accum.asnumpy(
                    ) + np.power(grad, 2)
                    tensor[:] = \
                        prev_param - self.learning_rate * grad / \
                        (np.sqrt(
                            accum.asnumpy()) + self.eps)


class AdaGradSparseUpdateOp(OptimizerSparseOp):
    def __init__(self, optimizer, param, unique, dedup_lookup, dedup_grad):
        from .initializers import constant
        accum = _init_states(param, 'adagrad_accum', constant,
                             fill_value=optimizer.initial_accumulator_value)
        self.eps = optimizer.eps
        states = [accum]
        super().__init__(optimizer, param, unique, dedup_lookup, dedup_grad, *states)

    def compute(self, input_vals, output_val, stream_handle=None):
        _, unique, dedup_lookup, dedup_grad, accum = input_vals
        if self.on_gpu:
            adagrad_update_indexedslices(
                unique, dedup_grad,
                dedup_lookup, output_val,
                self.learning_rate, accum, self.eps, stream_handle)
        else:
            if DNNL_LIB['cpu_AdaGradUpdateIndexedSlices']:
                cpu_adagrad_update_indexedslices(
                    unique, dedup_grad,
                    dedup_lookup, output_val,
                    self.learning_rate, accum, self.eps)
            else:
                grad = dedup_grad.asnumpy()
                accum[:] = accum.asnumpy(
                ) + np.power(grad, 2)
                output_val[:] = \
                    dedup_lookup.asnumpy() - self.learning_rate * grad / \
                    (np.sqrt(
                        accum.asnumpy()) + self.eps)


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7, l2reg=0):
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__(learning_rate, l2reg)
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.opt_op_type = AdaGradUpdateOp
        self.sparse_opt_op_type = AdaGradSparseUpdateOp

    def get_config(self):
        return (ctypes.c_int(3), (ctypes.c_float * 3)(self.learning_rate, self.initial_accumulator_value, self.eps), ctypes.c_int(3))


class BetatsUpdateOp(Op):
    # assistant op for betats in adam, adamw, lamb
    def __init__(self, betats, beta1, beta2, ctx=None):
        super().__init__(BetatsUpdateOp, [betats], ctx)
        self.beta1 = beta1
        self.beta2 = beta2

    def infer_shape(self, input_shapes):
        return None

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            betats_update(input_vals[0], self.beta1, self.beta2, stream_handle)
        else:
            if DNNL_LIB['cpu_BetatsUpdate']:
                cpu_betats_update(input_vals[0], self.beta1, self.beta2)
            else:
                betats = input_vals[0]
                np_betats = betats.asnumpy()
                np_betats[0] *= self.beta1
                np_betats[1] *= self.beta2
                betats[:] = np_betats


def betats_update_op(betats, beta1, beta2, ctx=None):
    return BetatsUpdateOp(betats, beta1, beta2, ctx)


class AdamUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        from .initializers import zeros
        m = _init_states(param, 'adam_m', zeros)
        v = _init_states(param, 'adam_v', zeros)
        pctx = param.ctx
        states = [m, v, optimizer.betatss[pctx],
                  optimizer.betats_update_ops[pctx]]
        if optimizer.amsgrad:
            maxv = _init_states(param, 'adam_maxv', zeros)
            states.append(maxv)
        self.amsgrad = optimizer.amsgrad
        self.beta1 = optimizer.beta1
        self.beta2 = optimizer.beta2
        self.epsilon = optimizer.epsilon
        super().__init__(param, grad, optimizer, *states)

    def compute(self, input_vals, output_val, stream_handle=None):
        tensor, grad, m, v, betats = input_vals[:5]
        if self.amsgrad:
            maxv = input_vals[-1]
        else:
            maxv = None
        if self.on_gpu:
            adam_update(tensor, grad, m, v, maxv, self.learning_rate, self.beta1,
                        self.beta2, betats, self.epsilon, self.l2reg, stream_handle)
        else:
            if DNNL_LIB['cpu_AdamOptimizerSparseUpdate'] and DNNL_LIB['cpu_AdamOptimizerUpdate']:
                cpu_adam_update(tensor, grad, m, v, maxv, self.learning_rate,
                                self.beta1, self.beta2, betats, self.l2reg, self.epsilon)
            else:
                cur_beta_ts = betats.asnumpy()
                cur_beta1_t, cur_beta2_t = cur_beta_ts[0], cur_beta_ts[1]
                if isinstance(grad, IndexedSlices):
                    np_indices = grad.indices.asnumpy()
                    np_tensor = tensor.asnumpy()
                    np_values = grad.values.asnumpy().reshape(
                        (-1, np_tensor.shape[-1]))
                    np_m = m.asnumpy()
                    np_v = v.asnumpy()
                    if self.amsgrad:
                        np_maxv = maxv.asnumpy()
                    np_unique_indices, inverse = np.unique(
                        np_indices, return_inverse=True)
                    new_value = np.zeros(
                        (len(np_unique_indices), np_values.shape[-1]), dtype=np.float32)
                    for j, ind in enumerate(inverse):
                        new_value[ind] += np_values[j]
                    for j, ind in enumerate(np_unique_indices.reshape(-1)):
                        if ind < 0:
                            continue
                        np_m[ind] = self.beta1 * np_m[ind] + \
                            (1 - self.beta1) * new_value[j]
                        np_v[ind] = self.beta2 * np_v[ind] + \
                            (1 - self.beta2) * new_value[j] * new_value[j]
                        mc = np_m[ind] / (1 - cur_beta1_t)
                        vc = np_v[ind] / (1 - cur_beta2_t)
                        if self.amsgrad:
                            np_maxv[ind] = np.maximum(vc, np_maxv[ind])
                            np_tensor[ind] -= self.learning_rate * \
                                mc / (np.sqrt(np_maxv[ind]) + self.epsilon)
                        else:
                            np_tensor[ind] -= self.learning_rate * \
                                mc / (np.sqrt(np_maxv[ind]) + self.epsilon)
                    m[:] = np_m
                    v[:] = np_v
                    tensor[:] = np_tensor
                    if self.amsgrad:
                        maxv[:] = np_maxv
                else:
                    prev_param = tensor.asnumpy()
                    grad = grad.asnumpy(
                    ) + self.l2reg * prev_param if self.l2reg > 0 else grad.asnumpy()
                    m[:] = self.beta1 * \
                        m.asnumpy() + (1 - self.beta1) * grad
                    v[:] = self.beta2 * v.asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = m.asnumpy() / (1 - cur_beta1_t)
                    vc = v.asnumpy() / (1 - cur_beta2_t)
                    if self.amsgrad:
                        cur_maxv = np.maximum(vc, maxv.asnumpy())
                        tensor[:] = prev_param - \
                            self.learning_rate * mc / \
                            (np.sqrt(cur_maxv) + self.epsilon)
                        maxv[:] = cur_maxv
                    else:
                        tensor[:] = prev_param - \
                            self.learning_rate * mc / \
                            (np.sqrt(vc) + self.epsilon)


class AdamSparseUpdateOp(OptimizerSparseOp):
    def __init__(self, optimizer, param, unique, dedup_lookup, dedup_grad):
        from .initializers import zeros
        m = _init_states(param, 'adam_m', zeros)
        v = _init_states(param, 'adam_v', zeros)
        pctx = param.ctx
        states = [m, v, optimizer.betatss[pctx],
                  optimizer.betats_update_ops[pctx]]
        if optimizer.amsgrad:
            maxv = _init_states(param, 'adam_maxv', zeros)
            states.append(maxv)
        self.amsgrad = optimizer.amsgrad
        self.beta1 = optimizer.beta1
        self.beta2 = optimizer.beta2
        self.epsilon = optimizer.epsilon
        super().__init__(optimizer, param, unique, dedup_lookup, dedup_grad, *states)

    def compute(self, input_vals, output_val, stream_handle=None):
        _, unique, dedup_lookup, dedup_grad = input_vals[:4]
        m, v, betats = input_vals[4:7]
        if self.amsgrad:
            maxv = input_vals[-1]
        else:
            maxv = None
        if self.on_gpu:
            adam_update_indexedslices(
                unique, dedup_grad,
                dedup_lookup, output_val,
                self.learning_rate, m, v, maxv, self.beta1,
                self.beta2, betats, self.epsilon, stream_handle)
        else:
            if DNNL_LIB['cpu_AdamUpdateIndexedSlices']:
                cpu_adam_update_indexedslices(
                    unique, dedup_grad,
                    dedup_lookup, output_val,
                    self.learning_rate, m, v, maxv, self.beta1,
                    self.beta2, betats, self.epsilon)
            else:
                grad = dedup_grad.asnumpy()
                m[:] = self.beta1 * \
                    m.asnumpy() + (1 - self.beta1) * grad
                v[:] = self.beta2 * v.asnumpy() + \
                    (1 - self.beta2) * grad * grad
                cur_betats = betats.asnumpy()
                mc = m.asnumpy() / (1 - cur_betats[0])
                vc = v.asnumpy() / (1 - cur_betats[1])
                if self.amsgrad:
                    cur_maxv = np.maximum(vc, maxv.asnumpy())
                    output_val[:] = dedup_lookup.asnumpy() - \
                        self.learning_rate * mc / \
                        (np.sqrt(cur_maxv) + self.epsilon)
                    maxv[:] = cur_maxv
                else:
                    output_val[:] = dedup_lookup.asnumpy() - \
                        self.learning_rate * mc / (np.sqrt(vc) + self.epsilon)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0, amsgrad=False):
        super(AdamOptimizer, self).__init__(learning_rate, l2reg)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.opt_op_type = AdamUpdateOp
        self.sparse_opt_op_type = AdamSparseUpdateOp

    def get_config(self):
        return (ctypes.c_int(4), (ctypes.c_float * 4)(self.learning_rate, self.beta1, self.beta2, self.epsilon), ctypes.c_int(4))


class AMSGradOptimizer(AdamOptimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, l2reg=0):
        super().__init__(learning_rate, beta1, beta2, epsilon, l2reg, amsgrad=True)


class AdamWUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        from .initializers import zeros
        m = _init_states(param, 'adamw_m', zeros)
        v = _init_states(param, 'adamw_v', zeros)
        self.beta1 = optimizer.beta1
        self.beta2 = optimizer.beta2
        self.beta1_t = optimizer.beta1_t
        self.beta2_t = optimizer.beta2_t
        self.epsilon = optimizer.epsilon
        self.weight_decay = optimizer.weight_decay
        super().__init__(param, grad, optimizer, m, v)

    def compute(self, input_vals, output_val, stream_handle=None):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        tensor, grad, m, v = input_vals
        if self.on_gpu:
            assert isinstance(tensor, NDArray)
            assert isinstance(grad, (NDArray, IndexedSlices))
            assert isinstance(m, NDArray)
            assert isinstance(v, NDArray)
            adamw_update(tensor, grad, m, v, self.learning_rate, self.beta1,
                         self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
        else:
            if isinstance(grad, IndexedSlices):
                raise NotImplementedError
            else:
                prev_param = tensor.asnumpy()
                grad = grad.asnumpy()
                m[:] = self.beta1 * \
                    m.asnumpy() + (1 - self.beta1) * grad
                v[:] = self.beta2 * v.asnumpy() + \
                    (1 - self.beta2) * grad * grad
                mc = m.asnumpy() / (1 - self.beta1_t)
                vc = v.asnumpy() / (1 - self.beta2_t)
                update = mc / (np.sqrt(vc) + self.epsilon)
                tensor[:] = prev_param - \
                    self.learning_rate * \
                    (update + self.weight_decay * tensor)


class AdamWOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.opt_op_type = AdamWUpdateOp

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))


class LambUpdateOp(OptimizerOp):
    def __init__(self, param, grad, optimizer):
        from .initializers import zeros
        m = _init_states(param, 'lamb_m', zeros)
        v = _init_states(param, 'lamb_v', zeros)
        self.beta1 = optimizer.beta1
        self.beta2 = optimizer.beta2
        self.beta1_t = optimizer.beta1_t
        self.beta2_t = optimizer.beta2_t
        self.epsilon = optimizer.epsilon
        self.weight_decay = optimizer.weight_decay
        super().__init__(param, grad, optimizer, m, v)

    def compute(self, input_vals, output_val, stream_handle=None):
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        tensor, grad, m, v = input_vals
        if self.on_gpu:
            assert isinstance(tensor, NDArray)
            assert isinstance(grad, (NDArray, IndexedSlices))
            assert isinstance(m, NDArray)
            assert isinstance(v, NDArray)
            lamb_update(tensor, grad, m, v, self.learning_rate, self.beta1,
                        self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
        else:
            if isinstance(grad, IndexedSlices):
                raise NotImplementedError
            else:
                prev_param = tensor.asnumpy()
                grad = grad.asnumpy()
                m[:] = self.beta1 * \
                    m.asnumpy() + (1 - self.beta1) * grad
                v[:] = self.beta2 * v.asnumpy() + \
                    (1 - self.beta2) * grad * grad
                mc = m.asnumpy() / (1 - self.beta1_t)
                vc = v.asnumpy() / (1 - self.beta2_t)
                update = mc / (np.sqrt(vc) + self.epsilon)
                norm2_param = np.sqrt(np.sum(np.power(tensor, 2)))
                norm2_update = np.sqrt(np.sum(np.power(update, 2)))
                tensor[:] = prev_param - \
                    self.learning_rate * norm2_param / norm2_update * \
                    (update + self.weight_decay * tensor)


class LambOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(LambOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.opt_op_type = LambUpdateOp

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))
