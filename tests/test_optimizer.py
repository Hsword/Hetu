import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op

def test_sgd():
    ctx = ht.gpu(0)
    shape = (500,400)
    param = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    grad = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    lr = 1e-2
    print("Prev param:")
    print(param)
    arr_param = ht.array(param, ctx)
    arr_grad = ht.array(grad, ctx)

    gpu_op.sgd_update(arr_param, arr_grad, lr, 0.0)
    re_param = arr_param.asnumpy()
    param = param - grad * lr

    print("Cur param:")
    print(re_param)
    print(param)

    np.testing.assert_allclose(re_param, param, atol=1e-5)


def test_adamw():
    ctx = ht.gpu(0)
    shape = (500,400)
    param = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    grad = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    m = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    v = np.random.uniform(0, 10, size=shape).astype(np.float32)
    lr = 1e-2
    beta1 = 0.9
    beta2 = 0.99
    beta1t = beta1**10
    beta2t = beta2**10
    eps = 1e-7
    weight_decay = 0.1

    print("Prev param:")
    print(param)
    print("Prev m:")
    print(m)
    print("Prev v:")
    print(v)

    arr_param = ht.array(param, ctx)
    arr_grad = ht.array(grad, ctx)
    arr_m = ht.array(m, ctx)
    arr_v = ht.array(v, ctx)
    gpu_op.adamw_update(arr_param, arr_grad, arr_m, arr_v, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay)
    re_param = arr_param.asnumpy()
    re_m = arr_m.asnumpy()
    re_v = arr_v.asnumpy()

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    mc = m / (1 - beta1t)
    vc = v / (1 - beta2t)
    update = mc / (np.sqrt(vc) + eps)
    param = param - lr * (update + weight_decay * param)

    print("Cur param:")
    print(re_param)
    print(param)

    print("Cur m:")
    print(re_m)
    print(m)

    print("Cur v:")
    print(re_v)
    print(v)

    np.testing.assert_allclose(re_param, param, atol=1e-5)
    np.testing.assert_allclose(re_m, m, atol=1e-5)
    np.testing.assert_allclose(re_v, v, atol=1e-5)

def test_lamb():
    ctx = ht.gpu(0)
    shape = (4,5)
    param = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    grad = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    m = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    v = np.random.uniform(0, 10, size=shape).astype(np.float32)
    lr = 1e-2
    beta1 = 0.9
    beta2 = 0.99
    beta1t = beta1**10
    beta2t = beta2**10
    eps = 1e-7
    weight_decay = 0.1

    print("Prev param:")
    print(param)
    print("Prev m:")
    print(m)
    print("Prev v:")
    print(v)

    arr_param = ht.array(param, ctx)
    arr_grad = ht.array(grad, ctx)
    arr_m = ht.array(m, ctx)
    arr_v = ht.array(v, ctx)
    gpu_op.lamb_update(arr_param, arr_grad, arr_m, arr_v, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay)
    re_param = arr_param.asnumpy()
    re_m = arr_m.asnumpy()
    re_v = arr_v.asnumpy()

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    mc = m / (1 - beta1t)
    vc = v / (1 - beta2t)
    update = mc / (np.sqrt(vc) + eps)
    norm2_param = np.sqrt(np.sum(np.power(param, 2)))
    norm2_update = np.sqrt(np.sum(np.power(update, 2)))
    param = param - lr * norm2_param / norm2_update * (update + weight_decay * param)

    print("Cur param:")
    print(re_param)
    print(param)

    print("Cur m:")
    print(re_m)
    print(m)

    print("Cur v:")
    print(re_v)
    print(v)

    np.testing.assert_allclose(re_param, param, atol=1e-5)
    np.testing.assert_allclose(re_m, m, atol=1e-5)
    np.testing.assert_allclose(re_v, v, atol=1e-5)


def test_adamw_sparse():
    ctx = ht.gpu(0)
    shape = (500, 400)
    l = np.random.randint(0,500,size=(100))
    indices = np.array(l)
    param = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    grad = np.random.uniform(-10, 10, size=(indices.shape[0], shape[1])).astype(np.float32)
    m = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    v = np.random.uniform(0, 10, size=shape).astype(np.float32)
    lr = 1e-2
    beta1 = 0.9
    beta2 = 0.99
    beta1t = beta1**10
    beta2t = beta2**10
    eps = 1e-7
    weight_decay = 0.1

    print("Prev param:")
    print(param)
    print("Prev m:")
    print(m)
    print("Prev v:")
    print(v)
    print("Indices:")
    print(indices)
    print("Grad:")
    print(grad)

    arr_param = ht.array(param, ctx)
    arr_indices = ht.array(indices, ctx)
    arr_value = ht.array(grad, ctx)
    arr_grad = ht.IndexedSlices(indices = arr_indices, values = arr_value, dense_shape = shape)
    arr_m = ht.array(m, ctx)
    arr_v = ht.array(v, ctx)
    gpu_op.adamw_update(arr_param, arr_grad, arr_m, arr_v, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay)
    re_param = arr_param.asnumpy()
    re_m = arr_m.asnumpy()
    re_v = arr_v.asnumpy()

    # numpy deduplicate
    d = dict()
    for i in l:
        d[i]=[]

    for i, g in zip(l, grad):
        d[i].append(g)
    for key in d.keys():
        g0 = d[key][0]
        for i in range(1, len(d[key])):
            g0 += d[key][i]
        d[key] = g0
    grad_new = []
    l_new = []
    for key in d.keys():
        l_new.append(key)
        grad_new.append(d[key])
    grad_new = np.array(grad_new)

    for idx, g in zip(l_new, grad_new):
        m[idx] = beta1 * m[idx] + (1 - beta1) * g
        v[idx] = beta2 * v[idx] + (1 - beta2) * g * g
        mc_idx = m[idx] / (1 - beta1t)
        vc_idx = v[idx] / (1 - beta2t)
        update = mc_idx / (np.sqrt(vc_idx) + eps)
        param[idx] = param[idx] - lr * (update + weight_decay * param[idx])

    print("Cur param:")
    print(re_param)
    print(param)

    print("Cur m:")
    print(re_m)
    print(m)

    print("Cur v:")
    print(re_v)
    print(v)

    np.testing.assert_allclose(re_param, param, atol=1e-5)
    np.testing.assert_allclose(re_m, m, atol=1e-5)
    np.testing.assert_allclose(re_v, v, atol=1e-5)


def test_lamb_sparse():
    ctx = ht.gpu(0)
    shape = (500, 400)
    l = np.random.randint(0,500,size=(100))
    # shape = (5,4)
    # l = [0,2,3]
    indices = np.array(l)
    param = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    grad = np.random.uniform(-10, 10, size=(indices.shape[0], shape[1])).astype(np.float32)
    m = np.random.uniform(-10, 10, size=shape).astype(np.float32)
    v = np.random.uniform(0, 10, size=shape).astype(np.float32)
    lr = 1e-2
    beta1 = 0.9
    beta2 = 0.99
    beta1t = beta1**10
    beta2t = beta2**10
    eps = 1e-7
    weight_decay = 0.1

    print("Prev param:")
    print(param)
    print("Prev m:")
    print(m)
    print("Prev v:")
    print(v)
    print("Indices:")
    print(indices)
    print("Grad:")
    print(grad)

    arr_param = ht.array(param, ctx)
    arr_indices = ht.array(indices, ctx)
    arr_value = ht.array(grad, ctx)
    arr_grad = ht.IndexedSlices(indices = arr_indices, values = arr_value, dense_shape = shape)
    arr_m = ht.array(m, ctx)
    arr_v = ht.array(v, ctx)
    gpu_op.lamb_update(arr_param, arr_grad, arr_m, arr_v, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay)
    re_param = arr_param.asnumpy()
    re_m = arr_m.asnumpy()
    re_v = arr_v.asnumpy()

    # numpy deduplicate
    d = dict()
    for i in l:
        d[i]=[]

    for i, g in zip(l, grad):
        d[i].append(g)
    for key in d.keys():
        g0 = d[key][0]
        for i in range(1, len(d[key])):
            g0 += d[key][i]
        d[key] = g0
    grad_new = []
    l_new = []
    for key in d.keys():
        l_new.append(key)
        grad_new.append(d[key])
    grad_new = np.array(grad_new)

    updates = []

    for idx, g in zip(l_new, grad_new):
        m[idx] = beta1 * m[idx] + (1 - beta1) * g
        v[idx] = beta2 * v[idx] + (1 - beta2) * g * g
        mc_idx = m[idx] / (1 - beta1t)
        vc_idx = v[idx] / (1 - beta2t)
        update = mc_idx / (np.sqrt(vc_idx) + eps)
        updates.append(update)
    updates = np.array(updates)

    param_indexed = []
    for idx in l_new:
        param_indexed.append(param[idx])
    param_indexed = np.array(param_indexed)

    norm2_param = np.sqrt(np.sum(np.power(param_indexed, 2))) # only use indexed params to calculate norm2
    norm2_update = np.sqrt(np.sum(np.power(updates, 2)))

    #print(norm2_param, norm2_update)

    for idx, u in zip(l_new, updates):
        param[idx] = param[idx] - lr * norm2_param / norm2_update * (u + weight_decay * param[idx])

    print("Cur param:")
    print(re_param)
    print(param)

    print("Cur m:")
    print(re_m)
    print(m)

    print("Cur v:")
    print(re_v)
    print(v)

    np.testing.assert_allclose(re_param, param, atol=1e-5)
    np.testing.assert_allclose(re_m, m, atol=1e-5)
    np.testing.assert_allclose(re_v, v, atol=1e-5)

#test_adamw()
#test_lamb()
#test_adamw_sparse()
#test_lamb_sparse()
#test_sgd()
