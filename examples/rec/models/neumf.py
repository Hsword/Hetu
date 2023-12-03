import hetu as ht
import hetu.layers as htl
from .base import RatingModel_Head


class NeuMF_Head(RatingModel_Head):
    def __init__(self, embed_dim, nsparse=2, ndense=0):
        # fixed 2 layers
        assert embed_dim % 5 == 0
        super().__init__(embed_dim)
        self.factor_num = embed_dim // 5
        self.mlp_layers = self.create_mlp(
            [8 * self.factor_num, 4 * self.factor_num, 2 * self.factor_num, self.factor_num])
        self.predict_layer = htl.Linear(
            2 * self.factor_num, 1, initializer=ht.init.GenXavierNormal(), activation=None, name=f'predict')

    def __call__(self, embeddings, dense, label):
        embeddings = ht.array_reshape_op(embeddings, [-1, 2, self.embed_dim])
        gmf_embs = ht.slice_op(
            embeddings, [0, 0, 0], [-1, -1, self.factor_num])
        mlp_embs = ht.slice_op(
            embeddings, [0, 0, self.factor_num], [-1, -1, -1])
        output_gmf = ht.reduce_mul_op(gmf_embs, [1])
        input_mlp = ht.array_reshape_op(
            mlp_embs, [-1, 2 * (self.embed_dim - self.factor_num)])
        output_mlp = self.mlp_layers(input_mlp)
        concat = ht.concatenate_op((output_gmf, output_mlp), -1)
        prediction = self.predict_layer(concat)
        prediction = ht.array_reshape_op(prediction, (-1,))
        return self.output(prediction, label)
