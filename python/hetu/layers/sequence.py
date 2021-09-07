
def Sequence(*args):
    assert len(args) > 0
    if len(args) == 1:
        return lambda x: args[0](x)
    else:
        return lambda x: Sequence(*(args[1:]))(args[0](x))


"""
an alternative implementation
if using class, please change all layers implementations to class
now all implementations are functions
"""
# class Sequence(object):
#     def __init__(self, *args):
#         self.layers = args

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
