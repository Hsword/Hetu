import hetu as ht
import hetu.layers as htl
from .base import RatingModel_Head


class MLP_Head(RatingModel_Head):
    def __init__(self, embed_dim, nsparse=2, ndense=0):
        # fixed 2 layers
        assert embed_dim % 4 == 0
        super().__init__(embed_dim)
        self.factor_num = embed_dim // 4
        self.mlp_layers = self.create_mlp(
            [8 * self.factor_num, 4 * self.factor_num, 2 * self.factor_num, self.factor_num])
        self.predict_layer = htl.Linear(
            self.factor_num, 1, initializer=ht.init.GenXavierNormal(), activation=None, name=f'predict')

    def __call__(self, embeddings, dense, label):
        input_mlp = ht.array_reshape_op(embeddings, [-1, 2 * self.embed_dim])
        output_mlp = self.mlp_layers(input_mlp)
        prediction = self.predict_layer(output_mlp)
        prediction = ht.array_reshape_op(prediction, (-1,))
        return self.output(prediction, label)
