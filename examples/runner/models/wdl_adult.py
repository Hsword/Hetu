import hetu as ht
from hetu import init


def wdl_adult(X_deep, X_wide, y_, dense_param_ctx):
    lr = 5 / 128
    dim_wide = 809
    dim_deep = 68

    with ht.context(dense_param_ctx):
        W = init.random_normal([dim_wide+20, 2], stddev=0.1, name="W")
        W1 = init.random_normal([dim_deep, 50], stddev=0.1, name="W1")
        b1 = init.random_normal([50], stddev=0.1, name="b1")
        W2 = init.random_normal([50, 20], stddev=0.1, name="W2")
        b2 = init.random_normal([20], stddev=0.1, name="b2")

    # deep
    Embedding = []
    X_deep_input = None

    for i in range(8):
        Embedding_name = "Embedding_deep_" + str(i)
        Embedding.append(init.random_normal(
            [50, 8], stddev=0.1, name=Embedding_name))
        now = ht.embedding_lookup_op(Embedding[i], X_deep[i])
        now = ht.array_reshape_op(now, (-1, 8))
        if X_deep_input is None:
            X_deep_input = now
        else:
            X_deep_input = ht.concat_op(X_deep_input, now, 1)

    for i in range(4):
        now = ht.array_reshape_op(X_deep[i + 8], (-1, 1))
        X_deep_input = ht.concat_op(X_deep_input, now, 1)

    mat1 = ht.matmul_op(X_deep_input, W1)
    add1 = mat1 + ht.broadcastto_op(b1, mat1)
    relu1 = ht.relu_op(add1)
    dropout1 = relu1
    mat2 = ht.matmul_op(dropout1, W2)
    add2 = mat2 + ht.broadcastto_op(b2, mat2)
    relu2 = ht.relu_op(add2)
    dropout2 = relu2
    dmodel = dropout2

    # wide
    wmodel = ht.concat_op(X_wide, dmodel, 1)
    wmodel = ht.matmul_op(wmodel, W)

    prediction = wmodel
    loss = ht.softmaxcrossentropy_op(prediction, y_)
    loss = ht.reduce_mean_op(loss, [0])

    opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)

    return loss, prediction, y_, train_op
