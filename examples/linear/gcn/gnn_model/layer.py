import hetu as ht
from hetu import init


class GCN(object):
    def __init__(self, in_features, out_features, norm_adj, activation=None, dropout=0,
                 name="GCN", custom_init=None):
        if custom_init is not None:
            self.weight = ht.Variable(
                value=custom_init[0], name=name+"_Weight")
            self.bias = ht.Variable(value=custom_init[1], name=name+"_Bias")
        else:
            self.weight = init.xavier_uniform(
                shape=(in_features, out_features), name=name+"_Weight")
            self.bias = init.zeros(shape=(out_features,), name=name+"_Bias")
        # self.mp is a sparse matrix and should appear in feed_dict later
        self.mp = norm_adj
        self.activation = activation
        self.dropout = dropout
        self.output_width = out_features

    def __call__(self, x):
        """
            Build the computation graph, return the output node
        """
        if self.dropout > 0:
            x = ht.dropout_op(x, 1 - self.dropout)
        x = ht.matmul_op(x, self.weight)
        msg = x + ht.broadcastto_op(self.bias, x)
        x = ht.csrmm_op(self.mp, msg)
        if self.activation == "relu":
            x = ht.relu_op(x)
        elif self.activation is not None:
            raise NotImplementedError
        return x


class SageConv(object):
    def __init__(self, in_features, out_features, norm_adj, activation=None, dropout=0,
                 name="GCN", custom_init=None, mp_val=None):

        self.weight = init.xavier_uniform(
            shape=(in_features, out_features), name=name+"_Weight")
        self.bias = init.zeros(shape=(out_features,), name=name+"_Bias")
        self.weight2 = init.xavier_uniform(
            shape=(in_features, out_features), name=name+"_Weight")
        # self.mp is a sparse matrix and should appear in feed_dict later
        self.mp = norm_adj
        self.activation = activation
        self.dropout = dropout
        self.output_width = 2 * out_features

    def __call__(self, x):
        """
            Build the computation graph, return the output node
        """
        feat = x
        if self.dropout > 0:
            x = ht.dropout_op(x, 1 - self.dropout)

        x = ht.csrmm_op(self.mp, x)
        x = ht.matmul_op(x, self.weight)
        x = x + ht.broadcastto_op(self.bias, x)
        if self.activation == "relu":
            x = ht.relu_op(x)
        elif self.activation is not None:
            raise NotImplementedError
        return ht.concat_op(x, ht.matmul_op(feat, self.weight2), axis=1)
