import hetu as ht
import hetu.layers as htl
from .base import CTRModel_Head


class DCN_Head(CTRModel_Head):
    # DCN model without embedding layer
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
        num_cross_layers=3,
    ):
        super().__init__(embed_dim, sparse_slot, dense_slot)
        self.num_cross_layers = num_cross_layers

        input_dim = embed_dim * sparse_slot + dense_slot
        self.cross_weights = [ht.init.xavier_normal(
            (input_dim, 1), name=f'cross_weight_{i}') for i in range(num_cross_layers)]
        self.cross_biases = [ht.init.zeros(
            (input_dim,), name=f'cross_bias_{i}') for i in range(num_cross_layers)]
        self.dnn_layers = self.create_mlp(
            [input_dim, 256, 256, 256], sigmoid_layer=2, name='deep')
        self.last_dnn_layer = htl.Linear(
            256 + input_dim, 1, initializer=ht.init.GenXavierNormal(), name='last')

    def cross_layer(self, x0, x1, i):
        # x0: input embedding feature (batch_size, num_sparse * embedding_size + num_dense)
        # x1: the output of last layer (batch_size, num_sparse * embedding_size + num_dense)
        x1w = ht.matmul_op(x1, self.cross_weights[i])  # (batch_size, 1)
        y = ht.mul_op(x0, ht.broadcastto_op(x1w, x0))
        y = ht.sum_op([y, x1, ht.broadcastto_op(self.cross_biases[i], y)])
        return y

    def build_cross_layers(self, x0):
        x1 = x0
        for i in range(self.num_cross_layers):
            x1 = self.cross_layer(x0, x1, i)
        return x1

    def __call__(self, sparse_input, dense_input, label):
        # here the sparse_input is the output of embedding layer
        sparse_input = ht.array_reshape_op(
            sparse_input, (-1, self.sparse_slot*self.embed_dim))
        if dense_input is None:
            x = sparse_input
        else:
            x = ht.concat_op(sparse_input, dense_input, axis=1)
        cross_output = self.build_cross_layers(x)
        dnn_output = self.dnn_layers(x)
        x = ht.concat_op(cross_output, dnn_output, axis=1)
        x = self.last_dnn_layer(x)
        return self.output(x, label)
