import hetu as ht


def Reshape(shape):
    return lambda x: ht.array_reshape_op(x, shape)
