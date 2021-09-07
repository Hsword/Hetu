import hetu as ht


def Concat(axis):
    def loop_concat(*args):
        if len(args) == 1:
            return args[0]
        return ht.concat_op(args[0], loop_concat(*(args[1:])), axis=axis)
    return loop_concat


def ConcatLayers(layers, axis=0):
    def concat_layers(x):
        cur_node = layers[0](x)
        for layer in layers[1:]:
            cur_node = ht.concat_op(cur_node, layer(x), axis=axis)
        return cur_node
    return concat_layers
