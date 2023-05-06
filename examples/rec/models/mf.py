import hetu as ht
from .base import RatingModel_Head


class MF_Head(RatingModel_Head):
    def __call__(self, embeddings, dense, label):
        user_input, item_input = embeddings
        output = ht.mul_op(user_input, item_input)
        prediction = ht.reduce_sum_op(output, [-1])
        return self.output(prediction, label)
