import hetu as ht
from .base import RatingModel_Head


class MF_Head(RatingModel_Head):
    def __call__(self, embeddings, dense, label):
        embeddings = ht.array_reshape_op(embeddings, [-1, 2, self.embed_dim])
        output = ht.reduce_mul_op(embeddings, [1])
        prediction = ht.reduce_sum_op(output, [-1])
        return self.output(prediction, label)
