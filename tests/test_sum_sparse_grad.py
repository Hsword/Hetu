import hetu as ht
import numpy as np


def get_var_op(nparray, name, dtype, ctx):
    htarray = ht.array(nparray, dtype=dtype, ctx=ctx)
    return ht.placeholder_op(name=name, dtype=dtype, value=htarray, ctx=ctx)


def compare(indrange=10, shape=(3, 4), dimension=16):
    indices = np.random.randint(indrange, size=shape)
    indices = np.unique(indices)
    real_indices = np.full(shape, -1, dtype=np.int32).reshape(-1)
    real_indices[:len(indices)] = indices
    indices = real_indices.reshape(shape)
    grads = np.random.random(size=shape+(dimension,))
    dense = np.random.random(size=(indrange, dimension))
    cpu_res = get_results(indices, grads, dense,
                          (indrange, dimension), ht.cpu())
    gpu_res = get_results(indices, grads, dense,
                          (indrange, dimension), ht.gpu(0))
    npresult = dense.copy()
    npgrads = grads.reshape((-1, dimension))
    for i, ind in enumerate(indices.reshape(-1)):
        if ind >= 0 and ind < indrange:
            npresult[ind] += npgrads[i]
    np.testing.assert_allclose(gpu_res, npresult)
    np.testing.assert_allclose(cpu_res, npresult)


def get_results(indices, grads, dense, shape, ctx):
    ind_var = get_var_op(indices, 'ind', np.int32, ctx)
    grad_var = get_var_op(grads, 'grad', np.float32, ctx)
    dense_var = get_var_op(dense, 'dense', np.float32, ctx)
    sum_op = ht.sum_sparse_gradient_op(shape, (ind_var, grad_var), dense_var)
    exec = ht.Executor(
        [sum_op], ctx=ctx)
    results = exec.run()
    return results[0].asnumpy()


compare()
compare(20, (3, 4, 5, 7))
compare(1000, (3, 4, 5, 7))
