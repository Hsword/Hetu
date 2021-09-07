import hetu as ht


def DropOut(p=0.5):
    return lambda x: ht.dropout_op(x, 1-p)
