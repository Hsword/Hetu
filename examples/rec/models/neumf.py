import hetu as ht
import hetu.layers as htl
from .base import RatingModel_Head


class NeuMF_Head(RatingModel_Head):
    def __init__(self, embed_dim):
        # fixed 2 layers
        assert embed_dim % 5 == 0
        super().__init__(embed_dim)
        self.factor_num = embed_dim // 5
        self.mlp_layers = self.create_mlp(
            [8 * self.factor_num, 4 * self.factor_num, 2 * self.factor_num, self.factor_num])
        self.predict_layer = htl.Linear(
            2 * self.factor_num, 1, initializer=ht.init.GenXavierNormal(), activation=None, name=f'predict')

    def __call__(self, embeddings, dense, label):
        user_input, item_input = embeddings
        # here the user_input and the item_input are the output of embedding layer
        user_gmf_emb = ht.slice_op(user_input, [0, 0], [-1, self.factor_num])
        item_gmf_emb = ht.slice_op(item_input, [0, 0], [-1, self.factor_num])
        user_mlp_emb = ht.slice_op(user_input, [0, self.factor_num], [-1, -1])
        item_mlp_emb = ht.slice_op(item_input, [0, self.factor_num], [-1, -1])
        output_gmf = ht.mul_op(user_gmf_emb, item_gmf_emb)
        input_mlp = ht.concatenate_op((user_mlp_emb, item_mlp_emb), -1)
        output_mlp = self.mlp_layers(input_mlp)
        concat = ht.concatenate_op((output_gmf, output_mlp), -1)
        prediction = self.predict_layer(concat)
        prediction = ht.array_reshape_op(prediction, (-1,))
        return self.output(prediction, label)
