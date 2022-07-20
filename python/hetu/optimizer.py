import numpy as np
import ctypes
from copy import copy, deepcopy
import hetu as ht
from .ndarray import NDArray, IndexedSlices, array
from .lr_scheduler import FixedScheduler
from .gpu_ops.Node import Op
from .gpu_ops.ParameterServerCommunicate import ParameterServerCommunicateOp
from .gpu_ops.Variable import PlaceholderOp
from .gpu_links.OptimizerLink import sgd_update, momentum_update, adagrad_update, adam_update, adamw_update, lamb_update
from .cpu_links.dnnl_op import sgd_update as cpu_sgd_update, momentum_update as cpu_momentum_update, adagrad_update as cpu_adagrad_update, adam_update as cpu_adam_update
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
        self.params = None
        self.tensors = None
        self.initiated = False

    @property
    def learning_rate(self):
        return self.lr_sched.get()

    @staticmethod
    def get_var_list(loss):
        # from .layers.base import OpLayer
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            # if (isinstance(node, PlaceholderOp) and node.trainable) or isinstance(node, OpLayer):
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

    def initiate_states(self, config):
        assert not self.initiated, "Optimizer already initiated."
        if self.tensors is None:
            self.tensors = [config.placeholder_to_arr_map[node]
                            for node in self.params]
        self.initiated = True

    def uninitiate_states(self):
        # for profiler use, delete tensors
        self.tensors = []
        self.initiated = False

    def update_tensors_version(self, tensor_map):
        self.tensors = [tensor_map[node] for node in self.params]

    def minimize(self, loss, var_list=None):
        """Return an optimizer op to update parameters.

        Parameters
        ----------
        loss: loss node that we are minimizing.
        var_list: list of nodes that we are taking derivative wrt.

        Returns
        -------
        An optimizer node.

        """
        self.loss = loss
        if not var_list:
            var_list = self.get_var_list(loss)
        self.params = var_list
        grads, self.backward2forward, self.forward2backward = ht.gradients(
            loss, self.params, return_all=True)
        optimizer_node = OptimizerOp(grads, self)
        return optimizer_node

    def __deepcopy__(self, memo):
        assert not self.initiated, 'Should not deep copy optimizer if already initiated!'
        new_opt = copy(self)
        new_opt.loss = deepcopy(self.loss, memo)
        new_opt.params = [deepcopy(node, memo) for node in self.params]
        new_opt.backward2forward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.backward2forward.items()])
        new_opt.forward2backward = dict([(deepcopy(k, memo), (deepcopy(n, memo) for n in v))
                                         for k, v in self.forward2backward.items()])
        return new_opt


class OptimizerOp(Op):
    def __init__(self, grads, optimizer):
        self.name = "Optimizer_%s" % (optimizer.name)
        self.optimizer = optimizer
        super().__init__(OptimizerOp, grads, None)

    def compute(self, input_vals, output_val, stream_handle=None, new_tensors_map=None):
        assert output_val is None
        # For PS op, this input_vals is None
        # PS mode doesn't need local update
        if new_tensors_map is not None:
            self.optimizer.update_tensors_version(new_tensors_map)
        if self.comm_mode != 'PS':
            self.optimizer.update(input_vals, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return None

    def forward_hook(self, config):
        # disable inplace if not lazy execution
        # previously we use array reshape lazy callback to do this, which is deprecated (not efficient)
        for node in self.inputs:
            node.inplace = False

        self.optimizer.initiate_states(config)
        self.on_cpu = self.on_gpu = None
        self.comm_mode = config.comm_mode
        # some things todo.
        if self.comm_mode != 'PS':
            for i in range(len(self.inputs)):
                # Though the gradients for transfer ops are well defined,
                # we called gradients in optimizer op before transfer ops are added.
                # So here we also add tranfer ops for gradients update.
                # Could be optimized later.
                if not isinstance(self.inputs[i], ParameterServerCommunicateOp):
                    paramctx = self.optimizer.params[i].ctx
                    self.inputs[i] = super().add_transfer_op(
                        self.inputs[i], paramctx, config.h2d_ops, config.d2h_ops)

    def backward_hook(self, config):
        self.comm_mode = config.comm_mode
        new_inputs = []
        for i, node in enumerate(self.inputs):
            cur_param = self.optimizer.params[i]
            if "expert" in cur_param.name:
                # expert parameter no use allreduce
                continue
            current_strategy = config.node_strategy.get(
                cur_param, self.comm_mode)
            cur_node = node
            if current_strategy == 'AllReduce' or (current_strategy == 'Hybrid' and not node.use_indexed_slices):
                cur_node = ht.allreduceCommunicate_op(
                    node, config.param_allreduce_group.get(cur_param, config.nccl_comm))
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            elif current_strategy == 'PS' or (current_strategy == 'Hybrid' and node.use_indexed_slices):
                cur_node = ht.parameterServerCommunicate_op(
                    node, cur_param, self.optimizer.get_config())
                if config.layer_indices is not None and node in config.layer_indices:
                    config.layer_indices[cur_node] = config.layer_indices[node] + 1
            new_inputs.append(cur_node)
        self.inputs = new_inputs

    def re_minimize(self):
        new_grads = ht.gradients(self.optimizer.loss, self.optimizer.params)
        self.inputs = new_grads


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, l2reg=0):
        super(SGDOptimizer, self).__init__(learning_rate, l2reg)
        self.name = 'SGD'

    def get_config(self):
        return (ctypes.c_int(0), (ctypes.c_float * 1)(self.learning_rate), ctypes.c_int(1))

    def initiate_states(self, config):
        super().initiate_states(config)

    def uninitiate_states(self):
        super().uninitiate_states()

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                sgd_update(
                    self.tensors[i], grads[i], self.learning_rate, self.l2reg, stream_handle)
            else:
                if DNNL_LIB['cpu_SGDOptimizerSparseUpdate'] and DNNL_LIB['cpu_SGDOptimizerUpdate']:
                    cpu_sgd_update(
                        self.tensors[i], grads[i], self.learning_rate, self.l2reg)
                else:
                    # not implement regularization
                    if isinstance(grads[i], IndexedSlices):
                        np_indices = grads[i].indices.asnumpy()
                        np_tensor = self.tensors[i].asnumpy()
                        np_values = grads[i].values.asnumpy().reshape(
                            (-1, np_tensor.shape[-1]))
                        for j, ind in enumerate(np_indices.reshape(-1)):
                            if ind < 0:
                                continue
                            np_tensor[ind] -= self.learning_rate * np_values[j]
                        self.tensors[i][:] = np_tensor
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.tensors[i][:] = prev_param - \
                            self.learning_rate * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, l2reg=0):
        super(MomentumOptimizer, self).__init__(learning_rate, l2reg)
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = "Momentum"

    def get_config(self):
        return (ctypes.c_int(self.nesterov + 1), (ctypes.c_float * 2)(self.learning_rate, self.momentum), ctypes.c_int(2))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.velocity = []
        for t in self.tensors:
            self.velocity.append(None if t is None else array(
                np.zeros(t.shape, dtype=np.float32), t.ctx))

    def uninitiate_states(self):
        super().uninitiate_states()
        self.velocity = []

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                momentum_update(self.tensors[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                self.nesterov, self.l2reg, stream_handle)
            else:
                if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                    cpu_momentum_update(self.tensors[i], grads[i], self.velocity[i],
                                        self.learning_rate, self.momentum, self.nesterov, self.l2reg)
                else:
                    if isinstance(grads[i], IndexedSlices):
                        raise NotImplementedError
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        velo = self.velocity[i].asnumpy()
                        if self.nesterov:
                            lr_grads = -self.learning_rate * grad
                            self.velocity[i][:] = self.momentum * \
                                (velo + lr_grads)
                            self.tensors[i][:] = prev_param + velo + lr_grads
                        else:
                            self.velocity[i][:] = self.momentum * \
                                velo - self.learning_rate * grad
                            self.tensors[i][:] = prev_param + velo


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7, l2reg=0):
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__(learning_rate, l2reg)
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.name = "AdaGrad"

    def get_config(self):
        return (ctypes.c_int(3), (ctypes.c_float * 3)(self.learning_rate, self.initial_accumulator_value, self.eps), ctypes.c_int(3))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.accumulator_value = []
        for t in self.tensors:
            self.accumulator_value.append(None if t is None else array(
                np.full(t.shape, self.initial_accumulator_value), t.ctx))

    def uninitiate_states(self):
        super().uninitiate_states()
        self.accumulator_value = []

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                adagrad_update(self.tensors[i], grads[i], self.accumulator_value[i],
                               self.learning_rate, self.eps, self.l2reg, stream_handle)
            else:
                if DNNL_LIB['cpu_AdaGradOptimizerSparseUpdate'] and DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                    cpu_adagrad_update(
                        self.tensors[i], grads[i], self.accumulator_value[i], self.learning_rate, self.l2reg, self.eps)
                else:
                    if isinstance(grads[i], IndexedSlices):
                        np_indices = grads[i].indices.asnumpy()
                        np_tensor = self.tensors[i].asnumpy()
                        np_values = grads[i].values.asnumpy().reshape(
                            (-1, np_tensor.shape[-1]))
                        np_acc = self.accumulator_value[i].asnumpy()
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
                        self.accumulator_value[i][:] = np_acc
                        self.tensors[i][:] = np_tensor
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.accumulator_value[i][:] = self.accumulator_value[i].asnumpy(
                        ) + np.power(grad, 2)
                        self.tensors[i][:] = \
                            prev_param - self.learning_rate * grad / \
                            (np.sqrt(
                                self.accumulator_value[i].asnumpy()) + self.eps)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, l2reg=0, amsgrad=False):
        super(AdamOptimizer, self).__init__(learning_rate, l2reg)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.maxv = []
        self.name = "Adam"

    def get_config(self):
        return (ctypes.c_int(4), (ctypes.c_float * 4)(self.learning_rate, self.beta1, self.beta2, self.epsilon), ctypes.c_int(4))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))
            if self.amsgrad:
                self.maxv.append(None if t is None else array(
                    np.zeros(t.shape), t.ctx))
            else:
                self.maxv.append(None)

    def uninitiate_states(self):
        super().uninitiate_states()
        self.m = []
        self.v = []
        self.maxv = []

    def uninitiate_states(self):
        super().uninitiate_states()
        self.m = []
        self.v = []

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.maxv[i], self.learning_rate, self.beta1,
                            self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.l2reg, stream_handle)
            else:
                if DNNL_LIB['cpu_AdamOptimizerSparseUpdate'] and DNNL_LIB['cpu_AdamOptimizerUpdate']:
                    cpu_adam_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.maxv[i], self.learning_rate,
                                    self.beta1, self.beta2, self.beta1_t, self.beta2_t, self.l2reg, self.epsilon)
                else:
                    if isinstance(grads[i], IndexedSlices):
                        np_indices = grads[i].indices.asnumpy()
                        np_tensor = self.tensors[i].asnumpy()
                        np_values = grads[i].values.asnumpy().reshape(
                            (-1, np_tensor.shape[-1]))
                        np_m = self.m[i].asnumpy()
                        np_v = self.v[i].asnumpy()
                        if self.amsgrad:
                            np_maxv = self.maxv[i].asnumpy()
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
                            mc = np_m[ind] / (1 - self.beta1_t)
                            vc = np_v[ind] / (1 - self.beta2_t)
                            if self.amsgrad:
                                np_maxv[ind] = np.maximum(vc, np_maxv[ind])
                                np_tensor[ind] -= self.learning_rate * \
                                    mc / (np.sqrt(np_maxv[ind]) + self.epsilon)
                            else:
                                np_tensor[ind] -= self.learning_rate * \
                                    mc / (np.sqrt(np_maxv[ind]) + self.epsilon)
                        self.m[i][:] = np_m
                        self.v[i][:] = np_v
                        self.tensors[i][:] = np_tensor
                        if self.amsgrad:
                            self.maxv[i][:] = np_maxv
                    else:
                        prev_param = self.tensors[i].asnumpy()
                        grad = grads[i].asnumpy(
                        ) + self.l2reg * prev_param if self.l2reg > 0 else grads[i].asnumpy()
                        self.m[i][:] = self.beta1 * \
                            self.m[i].asnumpy() + (1 - self.beta1) * grad
                        self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                            (1 - self.beta2) * grad * grad
                        mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                        vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                        if self.amsgrad:
                            cur_maxv = np.maximum(vc, self.maxv[i].asnumpy())
                            self.tensors[i][:] = prev_param - \
                                self.learning_rate * mc / \
                                (np.sqrt(cur_maxv) + self.epsilon)
                            self.maxv[i][:] = cur_maxv
                        else:
                            self.tensors[i][:] = prev_param - \
                                self.learning_rate * mc / \
                                (np.sqrt(vc) + self.epsilon)


class AMSGradOptimizer(AdamOptimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, l2reg=0):
        super().__init__(learning_rate, beta1, beta2, epsilon, l2reg, amsgrad=True)


class AdamWOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(AdamWOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "AdamW"

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))

    def uninitiate_states(self):
        super().uninitiate_states()
        self.m = []
        self.v = []

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], NDArray)
                assert isinstance(
                    grads[i], (NDArray, IndexedSlices))
                assert isinstance(self.m[i], NDArray)
                assert isinstance(self.v[i], NDArray)
                adamw_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                             self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
            else:
                if isinstance(grads[i], IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * \
                        (update + self.weight_decay * self.tensors[i])


class LambOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0):
        super(LambOptimizer, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = "Lamb"

    def get_config(self):
        return (ctypes.c_int(5), (ctypes.c_float * 5)(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay), ctypes.c_int(5))

    def initiate_states(self, config):
        super().initiate_states(config)
        self.m = []
        self.v = []
        for t in self.tensors:
            self.m.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))
            self.v.append(None if t is None else array(
                np.zeros(t.shape), t.ctx))

    def uninitiate_states(self):
        super().uninitiate_states()
        self.m = []
        self.v = []

    def update(self, grads, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.tensors)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            if grads[i] == None:
                continue
            if self.params[i].on_gpu:
                assert isinstance(self.tensors[i], NDArray)
                assert isinstance(
                    grads[i], (NDArray, IndexedSlices))
                assert isinstance(self.m[i], NDArray)
                assert isinstance(self.v[i], NDArray)
                lamb_update(self.tensors[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                            self.beta2, self.beta1_t, self.beta2_t, self.epsilon, self.weight_decay, stream_handle)
            else:
                if isinstance(grads[i], IndexedSlices):
                    raise NotImplementedError
                else:
                    prev_param = self.tensors[i].asnumpy()
                    grad = grads[i].asnumpy()
                    self.m[i][:] = self.beta1 * \
                        self.m[i].asnumpy() + (1 - self.beta1) * grad
                    self.v[i][:] = self.beta2 * self.v[i].asnumpy() + \
                        (1 - self.beta2) * grad * grad
                    mc = self.m[i].asnumpy() / (1 - self.beta1_t)
                    vc = self.v[i].asnumpy() / (1 - self.beta2_t)
                    update = mc / (np.sqrt(vc) + self.epsilon)
                    norm2_param = np.sqrt(np.sum(np.power(self.tensors[i], 2)))
                    norm2_update = np.sqrt(np.sum(np.power(update, 2)))
                    self.tensors[i][:] = prev_param - \
                        self.learning_rate * norm2_param / norm2_update * \
                        (update + self.weight_decay * self.tensors[i])
