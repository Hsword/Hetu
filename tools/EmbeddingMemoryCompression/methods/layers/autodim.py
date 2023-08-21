import hetu as ht
from hetu.layers import Embedding


class AutoDimEmbedding(Embedding):
    def __init__(self, num_embeddings, dim_candidates, num_slot, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from hetu.layers.normalization import BatchNorm
        self.num_embeddings = num_embeddings
        self.num_slot = num_slot
        self.batch_size = batch_size
        temperature_decay = 0.00005 / 2000 * batch_size
        self.temperature_updater = lambda t: (
            1 / max(0.01, 1-temperature_decay*t))
        self.dim_candidates = dim_candidates
        self.dim_candidates.sort()
        self.num_cands = len(dim_candidates)
        self.max_dim = self.dim_candidates[-1]
        self.name = name
        self.ctx = ctx
        self.initializer = initializer
        self.bn_layers = {dim: BatchNorm(self.max_dim, scale=False, bias=False, name='bn{}'.format(
            dim)) for dim in self.dim_candidates}
        self.embedding_tables = {dim: initializer(shape=(self.num_embeddings, dim), name='{}{}'.format(
            name, dim), ctx=self.ctx) for dim in dim_candidates}
        self.weights = {dim: initializer(shape=(num_slot, dim, self.max_dim), name='weight{}'.format(
            dim), ctx=self.ctx) for dim in dim_candidates}
        self.biases = {dim: ht.init.zeros(shape=(num_slot, 1, self.max_dim,), name='bias{}'.format(
            dim), ctx=self.ctx) for dim in dim_candidates}
        self.alpha = initializer(
            shape=(num_slot, self.num_cands), name='alphas', ctx=self.ctx)

    def __call__(self, x):
        lookups = {}
        for dim in self.dim_candidates:
            cur_x = ht.embedding_lookup_op(self.embedding_tables[dim], x)
            lookups[dim] = cur_x
            # (bs, nslot, cdim)
        self.lookups = lookups
        return self.make_embed(lookups)

    def make_embed(self, lookups):
        middle_results = []
        for dim, lookup in lookups.items():
            # (bs, nslot, cdim)
            cur_x = ht.transpose_op(lookup, (1, 0, 2))
            # (nslot, bs, cdim)
            cur_x = ht.batch_matmul_op(cur_x, self.weights[dim])
            # (nslot, bs, dim)
            cur_bias = ht.broadcastto_op(self.biases[dim], cur_x)
            cur_x = ht.add_op(cur_x, cur_bias)
            # (nslot, bs, dim)
            cur_x = ht.transpose_op(cur_x, (1, 0, 2))
            # (bs, nslot, dim)
            cur_x = ht.array_reshape_op(cur_x, (-1, self.max_dim))
            # (bs * nslot, dim)
            cur_x = self.bn_layers[dim](cur_x)
            cur_x = ht.array_reshape_op(
                cur_x, (-1, self.num_slot, self.max_dim, 1))
            # (bs, nslot, dim, 1)
            middle_results.append(cur_x)
        log_alpha = ht.log_softmax_op(self.alpha)
        w_noise = ht.add_op(log_alpha, ht.gumbel_sample_op(self.alpha.shape))
        w_noise = ht.mul_byconst_op(
            w_noise, 1, const_updater=self.temperature_updater)
        p_weight = ht.softmax_op(w_noise)
        # (nslot, ncands)
        p_weight = ht.array_reshape_op(
            p_weight, (1, self.num_slot, self.num_cands, 1))
        p_weight = ht.broadcast_shape_op(
            p_weight, (self.batch_size, self.num_slot, self.num_cands, 1))
        # (bs, nslot, ncands, 1)
        sparse_inputs = ht.concatenate_op(middle_results, axis=3)
        # (bs, nslot, dim, ncands)
        final_embedding = ht.batch_matmul_op(sparse_inputs, p_weight)
        # (bs, nslot, dim, 1)
        final_embedding = ht.array_reshape_op(
            final_embedding, (self.batch_size, self.num_slot, self.max_dim))
        # (bs, nslot, dim)
        return final_embedding

    def __repr__(self):
        return f'{self.name}({self.num_embeddings};{self.dim_candidates})'


class AutoDimRetrainEmbedding(Embedding):
    def __init__(self, num_embeddings, compressed_dim, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.compressed_dim = compressed_dim
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(num_embeddings, compressed_dim), name=name, ctx=self.ctx)
        self.weight = initializer(
            shape=(compressed_dim, embedding_dim), name=f'{name}_weight', ctx=self.ctx)
        self.bias = ht.init.zeros(
            shape=(embedding_dim,), name=f'{name}_bias', ctx=self.ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            res = ht.embedding_lookup_op(self.embedding_table, x)
            res = ht.linear_op(res, self.weight, self.bias)
        return res

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.compressed_dim},{self.embedding_dim})'
