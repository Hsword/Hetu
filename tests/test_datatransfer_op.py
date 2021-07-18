import numpy as np
import hetu as ht


def test_dense():
    npw = np.random.random((5, 10)).astype(np.float32)
    npx = np.random.random((7, 5)).astype(np.float32)
    cpuctx = ht.cpu(0)
    gpuctx = ht.gpu(0)

    X = ht.Variable(name="x")
    mid = X + 3
    W = ht.Variable(name='w', value=npw, ctx=cpuctx)
    y = ht.matmul_op(mid, W)
    opt = ht.optim.SGDOptimizer(learning_rate=0.1)
    train_op = opt.minimize(y)
    executor = ht.Executor([y, train_op], ctx=gpuctx)
    pred_y, _ = executor.run(
        feed_dict={X: npx}, convert_to_numpy_ret_vals=True)

    nppred_y = np.matmul((npx + 3), npw)
    np.testing.assert_allclose(pred_y, nppred_y, rtol=1e-6)
    new_npw = npw - 0.1 * \
        np.matmul((npx + 3).T, np.ones(nppred_y.shape).astype(np.float32))
    np.testing.assert_allclose(
        executor.config.placeholder_to_arr_map[W].asnumpy(), new_npw, rtol=1e-10)


test_dense()


def test_sparse():
    npemb = np.random.random((100, 20)).astype(np.float32)
    npind = np.array(np.random.randint(100, size=(10,)))
    npw = np.random.random((20, 30)).astype(np.float32)
    cpuctx = ht.cpu(0)
    gpuctx = ht.gpu(0)

    embedding = ht.Variable('embeddingtable', value=npemb, ctx=cpuctx)
    index = ht.Variable(name="index", ctx=cpuctx)
    W = ht.Variable(name="w", value=npw)
    y = ht.embedding_lookup_op(embedding, index)  # (10, 20)
    y = ht.matmul_op(y, W)
    opt = ht.optim.SGDOptimizer(0.1)
    train_op = opt.minimize(y)
    executor = ht.Executor([y, train_op], ctx=gpuctx)

    out, _ = executor.run(feed_dict={index: npind.astype(
        np.float32)}, convert_to_numpy_ret_vals=True)

    np_out = np.matmul(npemb[npind], npw)
    np.testing.assert_allclose(out, np_out, rtol=1e-6)
    tmp_grad = np.matmul(np.ones(np_out.shape).astype(np.float32), npw.T)
    for i, localid in enumerate(npind):
        npemb[localid] -= 0.1 * tmp_grad[i]
    np.testing.assert_allclose(
        executor.config.placeholder_to_arr_map[embedding].asnumpy(), npemb, rtol=1e-6)


test_sparse()
