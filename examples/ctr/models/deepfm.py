import hetu as ht
import hetu.layers as htl
from .base import CTRModel_Head


class DeepFM_Head(CTRModel_Head):
    # DeepFM model without embedding layer
    # TODO: implement the 1st order embedding in the model, rather than in the embedding
    def __init__(
        self,
        embed_dim=16,
        sparse_slot=26,
        dense_slot=13,
    ):
        super().__init__(embed_dim, sparse_slot, dense_slot)

        self.fm_weight = htl.Linear(
            dense_slot, 1, bias=False, initializer=ht.init.GenXavierNormal(), name='fm_weight')
        self.dnn_layers = self.create_mlp(
            [(embed_dim-1) * sparse_slot, 256, 256, 1], sigmoid_layer=2, name='deep')

    def __call__(self, sparse_input, dense_input, label):
        # here the sparse_input is the output of embedding layer
        sparse_input = ht.array_reshape_op(
            sparse_input, (-1, self.sparse_slot, self.embed_dim))

        # 1st order output
        sparse_1dim_input = ht.slice_op(sparse_input, [0, 0, 0], [-1, -1, 1])
        fm_sparse_part = ht.reduce_sum_op(sparse_1dim_input, axes=1)
        if dense_input is not None:
            fm_dense_part = self.fm_weight(dense_input)
        else:
            fm_dense_part = None

        # 2nd order output
        sparse_2dim_input = ht.slice_op(sparse_input, [0, 0, 1], [-1, -1, -1])
        sparse_2dim_sum = ht.reduce_sum_op(sparse_2dim_input, axes=1)
        sparse_2dim_sum_square = ht.power_op(sparse_2dim_sum, 2)
        sparse_2dim_square = ht.power_op(sparse_2dim_input, 2)
        sparse_2dim_square_sum = ht.reduce_sum_op(sparse_2dim_square, axes=1)
        sparse_2dim = ht.minus_op(
            sparse_2dim_sum_square, sparse_2dim_square_sum)
        sparse_2dim_half = ht.mul_byconst_op(sparse_2dim, 0.5)
        sparse_2dim_part = ht.reduce_sum_op(
            sparse_2dim_half, axes=1, keepdims=True)

        # dnn
        flatten = ht.array_reshape_op(
            sparse_2dim_input, (-1, self.sparse_slot*(self.embed_dim-1)))
        dnn_output = self.dnn_layers(flatten)

        if fm_dense_part is not None:
            y = ht.sum_op([fm_dense_part, fm_sparse_part,
                           sparse_2dim_part, dnn_output])
        else:
            y = ht.sum_op([fm_sparse_part, sparse_2dim_part, dnn_output])

        return self.output(y, label)
