import hetu as ht
import hetu.layers as htl
from .base import RatingModel_Head


class GMF_Head(RatingModel_Head):
    def __init__(self, embed_dim, nsparse=2, ndense=0):
        # fixed 2 layers
        super().__init__(embed_dim)
        self.predict_layer = htl.Linear(
            self.embed_dim, 1, initializer=ht.init.GenXavierNormal(), activation=None, name=f'predict')

    def __call__(self, embeddings, dense, label):
        gmf_embs = ht.array_reshape_op(embeddings, [-1, 2, self.embed_dim])
        output_gmf = ht.reduce_mul_op(gmf_embs, [1])
        prediction = self.predict_layer(output_gmf)
        prediction = ht.array_reshape_op(prediction, (-1,))
        return self.output(prediction, label)
