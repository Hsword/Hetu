import hetu as ht


def Concatenate(axis):
    def concatenate_helper(*args):
        if len(args) == 1:
            return args[0]
        else:
            return ht.concatenate_op(args, axis=axis)
    return concatenate_helper


def ConcatenateLayers(layers, axis=0):
    def concatenate_layers(x):
        if len(layers) == 1:
            return layers[0](x)
        else:
            return ht.concatenate_op([layer(x) for layer in layers], axis=axis)
    return concatenate_layers
