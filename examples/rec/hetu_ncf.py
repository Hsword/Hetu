import hetu as ht
import hetu.layers as htl


class NCF(object):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout, model):
        self.model = model
        self.dropout = dropout
        emb_init = ht.init.GenNormal(stddev=0.01)
        self.embed_user_GMF = htl.Embedding(
            user_num, factor_num, initializer=emb_init, name='user_gmf')
        self.embed_item_GMF = htl.Embedding(
            item_num, factor_num, initializer=emb_init, name='item_gmf')
        self.embed_user_MLP = htl.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)), initializer=emb_init, name='user_mlp')
        self.embed_item_MLP = htl.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)), initializer=emb_init, name='item_mlp')

        linear_init = ht.init.GenXavierUniform()
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(htl.DropOut(p=self.dropout))
            MLP_modules.append(htl.Linear(input_size, input_size//2,
                               activation=ht.relu_op, initializer=linear_init, name=f'dnn_{i}'))
        self.MLP_layers = htl.Sequence(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        pred_init = ht.init.GenGeneralXavierUniform(1, 'fan_in')
        self.predict_layer = htl.Linear(
            predict_size, 1, initializer=pred_init, name='pred')

    def __call__(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = ht.mul_op(embed_user_GMF, embed_item_GMF)
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = ht.concatenate_op(
                (embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = ht.concatenate_op((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return ht.array_reshape_op(prediction, (-1,))
