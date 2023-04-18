import hetu as ht
import numpy as np


def get_var_op(nparray, name, dtype, ctx):
    htarray = ht.array(nparray, dtype=dtype, ctx=ctx)
    return ht.placeholder_op(name=name, dtype=dtype, value=htarray, ctx=ctx)


def compare(indrange=10, shape=(3, 4), dimension=16):
    indices = np.random.randint(indrange, size=shape)
    param = np.random.random(size=(indrange, dimension,))
    lookups = np.random.random(size=shape+(dimension,))
    grads = np.random.random(size=shape+(dimension,))
    cpu_res = get_results(param, indices, lookups, grads, ht.cpu())
    gpu_res = get_results(param, indices, lookups, grads, ht.gpu(0))
    cpu_dedup, cpu_nunique, cpu_offset, cpu_length, cpu_deduplookup, cpu_dedupgrad, cpu_sgd_val, cpu_new_param = cpu_res
    gpu_dedup, gpu_nunique, gpu_offset, gpu_length, gpu_deduplookup, gpu_dedupgrad, gpu_sgd_val, gpu_new_param = gpu_res
    assert np.all(cpu_dedup == gpu_dedup)
    assert cpu_nunique == gpu_nunique
    assert np.all(cpu_offset == gpu_offset)
    assert np.all(cpu_length[:cpu_nunique+1] == gpu_length[:gpu_nunique+1])
    assert np.all(cpu_deduplookup == gpu_deduplookup)
    assert np.all(cpu_dedupgrad == gpu_dedupgrad)
    np.testing.assert_allclose(cpu_sgd_val, gpu_sgd_val, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(
        cpu_new_param, gpu_new_param, rtol=1e-6, atol=2e-6)


def get_results(param, indices, lookups, grads, ctx):
    nele = np.prod(indices.shape).item()
    variable_op = get_var_op(indices, 'ind', np.int32, ctx)
    lookup_var = get_var_op(lookups, 'lookup', np.float32, ctx)
    grad_var = get_var_op(grads, 'grad', np.float32, ctx)
    param_var = get_var_op(param, 'param', np.float32, ctx)
    indices_op = ht.unique_indices_op(variable_op, ctx=ctx)
    offsets_op = ht.unique_indices_offsets_op(indices_op, ctx=ctx)
    deduplookup_op = ht.deduplicate_lookup_op(
        lookup_var, offsets_op, ctx=ctx)
    dedupgrad_op = ht.deduplicate_grad_op(
        grad_var, offsets_op, ctx=ctx)
    opt = ht.optim.SGDOptimizer(learning_rate=10)
    opt.backward2forward = {}
    opt.loss = None
    sgd_op = ht.optimizer.SGDSparseUpdateOp(
        opt, param_var, indices_op, deduplookup_op, dedupgrad_op)
    update_op = ht.assign_with_indexedslices_op(
        param_var, indices_op, sgd_op, ctx=ctx)
    exec = ht.Executor(
        [indices_op, offsets_op, deduplookup_op, dedupgrad_op, sgd_op, update_op, param_var], ctx=ctx)
    results = exec.run()
    ht_dedup = results[0].asnumpy()
    ht_offsets = results[1].asnumpy()
    rets = (ht_dedup, ht_offsets[0],
            ht_offsets[1:nele+1], ht_offsets[nele+1:])
    rets += (results[2].asnumpy(), results[3].asnumpy())
    rets += (results[4].asnumpy(), results[-1].asnumpy())
    return rets


compare()
compare(20, (3, 4, 5, 7))
compare(1000, (3, 4, 5, 7))
