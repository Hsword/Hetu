from .embedding import Embedding
import hetu as ht
import numpy as np
import os.path as osp


class HashEmbedding(Embedding):
    def __call__(self, x):
        # ref MLSys20, HierPS
        with ht.context(self.ctx):
            sparse_input = ht.mod_hash_op(x, self.num_embeddings)
            return ht.embedding_lookup_op(self.embedding_table, sparse_input)


class CompositionalEmbedding(Embedding):
    def __init__(self, num_quotient, num_remainder, embedding_dim, aggregator='mul', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        # KDD20, CompositionalHash
        # CIKM21, BinaryCodeHash
        # adapted from DLRM QREmbeddingBag
        aggregator = aggregator[:3]
        assert aggregator in ('sum', 'mul')
        self.aggregator = aggregator
        self.num_quotient = num_quotient
        self.num_remainder = num_remainder
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.qemb = initializer(
            shape=(self.num_quotient, self.embedding_dim), name=f'{name}_q', ctx=ctx)
        self.remb = initializer(
            shape=(self.num_remainder, self.embedding_dim), name=f'{name}_r', ctx=ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            qind = ht.div_hash_op(x, self.num_remainder)
            rind = ht.mod_hash_op(x, self.num_remainder)
            q = ht.embedding_lookup_op(self.qemb, qind)
            r = ht.embedding_lookup_op(self.remb, rind)
            if self.aggregator == 'sum':
                result = ht.add_op(q, r)
            elif self.aggregator == 'mul':
                result = ht.mul_op(q, r)
            return result

    def __repr__(self):
        return f'{self.name}({self.num_quotient},{self.num_remainder},{self.embedding_dim})'


class TensorTrainEmbedding(Embedding):
    def __init__(self, decomp_nemb, decomp_ndim, rank, name='embedding', ctx=None):
        self.num_tables = len(decomp_nemb)
        assert len(decomp_ndim) == self.num_tables
        self.decomp_nemb = decomp_nemb
        self.decomp_ndim = decomp_ndim
        self.ranks = [1, rank, rank, 1]
        self.name = name
        self.ctx = ctx
        cur_shapes = []
        for i in range(self.num_tables):
            nrow = decomp_nemb[i]
            ndim = decomp_ndim[i]
            prerank = self.ranks[i]
            postrank = self.ranks[i+1]
            ncol = prerank * ndim * postrank
            cur_shapes.append((nrow, ncol))
        ttcore_initializer = ht.init.GenReversedTruncatedNormal(
            stddev=1 / ((np.sqrt(1 / 3 * np.prod(decomp_nemb))) ** (1/3)))
        self.tt_cores = tuple(ttcore_initializer(
            shape=sh, name=f'{name}_{i}') for i, sh in enumerate(cur_shapes))

    def __call__(self, x):
        indices = x
        accum_embed = None
        accum_dim = 1
        for i in range(self.num_tables):
            if i == self.num_tables - 1:
                cur_ind = indices
            else:
                cur_ind = ht.mod_hash_op(indices, self.decomp_nemb[i])
                indices = ht.div_hash_op(indices, self.decomp_nemb[i])
            partial_embed = ht.embedding_lookup_op(self.tt_cores[i], cur_ind)
            if i == 0:
                accum_embed = partial_embed
            else:
                accum_embed = ht.array_reshape_op(
                    accum_embed, (-1, accum_dim, self.ranks[i]))
                partial_embed = ht.array_reshape_op(
                    partial_embed, (-1, self.ranks[i], self.decomp_ndim[i] * self.ranks[i+1]))
                accum_embed = ht.batch_matmul_op(
                    accum_embed, partial_embed)
            accum_dim *= self.decomp_ndim[i]
        accum_embed = ht.array_reshape_op(
            accum_embed, (-1, accum_dim))
        return accum_embed

    def __repr__(self):
        return f'{self.name}({self.ranks[1]})'


class DeepHashEmbedding(Embedding):
    def __init__(self, embedding_dim, mlp_dim, num_buckets, num_hash, nprs, dist='uniform', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert dist in ('uniform', 'normal')
        self.distribution = dist
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.num_hash = num_hash
        self.name = name
        self.ctx = ctx
        self.mlp_dim = mlp_dim
        prime_path = osp.join(osp.dirname(osp.abspath(
            __file__)), 'primes.npy')
        allprimes = np.load(prime_path)
        for i, p in enumerate(allprimes):
            if p >= num_buckets:
                break
        self.allprimes = allprimes[i:]
        self.slopes = self.make_random(nprs, 'slopes')
        self.biases = self.make_random(nprs, 'biases')
        self.primes = self.make_primes(nprs, 'primes')
        self.layers = self.make_layers(initializer)

    def make_layers(self, initializer):
        from .linear import Linear
        from .normalization import BatchNorm
        from .mish import Mish
        from .sequence import Sequence
        layers = [
            Linear(self.num_hash, self.mlp_dim,
                   initializer=initializer, name='linear1'),
            BatchNorm(self.mlp_dim, name='bn1'),
            Mish(),
        ]
        for i in range(4):
            layers.extend([
                Linear(self.mlp_dim, self.mlp_dim,
                       initializer=initializer, name=f'linear{i+2}'),
                BatchNorm(self.mlp_dim, name=f'bn{i+2}'),
                Mish(),
            ])
        layers.append(Linear(
            self.mlp_dim, self.embedding_dim, initializer=initializer, name='linear6'))
        return Sequence(*layers)

    def make_primes(self, nprs, name):
        primes = ht.placeholder_op(name=name, value=nprs.choice(
            self.allprimes, size=self.num_hash).astype(np.int32), dtype=np.int32, trainable=False)
        return primes

    def make_random(self, nprs, name):
        randoms = ht.placeholder_op(name=name, value=nprs.randint(
            1, self.num_buckets, size=self.num_hash).astype(np.int32), dtype=np.int32, trainable=False)
        return randoms

    def __call__(self, x):
        # KDD21, DHE
        x = ht.learn_hash_op(x, self.slopes, self.biases,
                             self.primes, self.num_buckets, self.distribution)
        x = ht.array_reshape_op(x, (-1, self.num_hash))
        x = self.layers(x)
        return x

    def __repr__(self):
        return f'{self.name}({self.mlp_dim})'


class RobeEmbedding(Embedding):
    def __init__(self, robe_array_size, embedding_dim, Z, nprs, use_slot_coef=True, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.robe_array_size = robe_array_size
        self.embedding_dim = embedding_dim
        assert Z <= embedding_dim
        self.Z = Z
        self.use_slot_coef = use_slot_coef
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.robe_array_size, 1), name=self.name, ctx=ctx)
        random_numbers = np.concatenate(
            [np.array([2038074743]), nprs.randint(1, 2038074743, (9,))]).astype(np.int32)
        self.random_numbers = ht.placeholder_op(
            'random_numbers', value=random_numbers, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            expanded_indices = ht.robe_hash_op(
                x, self.random_numbers, self.robe_array_size, self.embedding_dim, self.Z, self.use_slot_coef)
            signs = ht.robe_sign_op(
                x, self.random_numbers, self.embedding_dim, self.use_slot_coef)
            lookups = ht.embedding_lookup_op(
                self.embedding_table, expanded_indices)
            lookups = ht.reshape_to_op(lookups, signs)
            lookups = ht.mul_op(lookups, signs)
            return lookups

    def __repr__(self):
        return f'{self.name}({self.robe_array_size})'


class DPQEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, num_choices, num_parts, batch_size, share_weights=False, mode='vq', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from ..initializers import nulls
        from .normalization import BatchNorm
        assert mode in ('vq', 'sx')
        assert embedding_dim % num_parts == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_choices = num_choices
        self.num_parts = num_parts
        self.batch_size = batch_size  # contains slot if use multi==0
        self.share_weights = share_weights
        self.mode = mode
        self.part_embedding_dim = embedding_dim // num_parts
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(shape=(
            num_embeddings, self.embedding_dim), name='{}_query'.format(name), ctx=ctx)
        self.key_matrix = self.make_matries(initializer, name+'_key')
        if mode == 'vq':
            self.value_matrix = self.key_matrix
        else:
            self.value_matrix = self.make_matries(initializer, name+'_value')
        self.bn_layer = BatchNorm(
            self.num_choices, scale=False, bias=False, name='{}_bn'.format(name))
        self.codebooks = nulls(shape=(num_embeddings, self.num_parts), name='{}_codebook'.format(
            name), ctx=ctx, dtype=np.int32, trainable=False)
        if not self.share_weights:
            dbase = np.array(
                [self.num_choices * d for d in range(self.num_parts)], dtype=int)
            dbase = np.tile(dbase, [self.batch_size, 1])
            dbase = ht.array(dbase, dtype=np.int32, ctx=self.ctx)
            self.dbase = ht.placeholder_op(
                'dbase', value=dbase, dtype=np.int32, trainable=False)

    def make_matries(self, initializer, name):
        if self.share_weights:
            shape = (self.num_choices, self.part_embedding_dim)
        else:
            shape = (self.num_parts * self.num_choices,
                     self.part_embedding_dim)
        return initializer(shape=shape, name='{}'.format(name), ctx=self.ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            # table: (nembed, dim), x: (bs, slot)
            query_lookups = ht.embedding_lookup_op(
                self.embedding_table, x)
            # (bs, slot, dim)
            inputs = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, self.part_embedding_dim))
            query_lookups = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, 1, self.part_embedding_dim))
            # (bs * slot, npart, 1, pdim)
            query_lookups = ht.tile_op(query_lookups, [self.num_choices, 1])
            # (bs * slot, npart, nkey, pdim)
            key_mat = ht.array_reshape_op(
                self.key_matrix, (-1, self.num_choices, self.part_embedding_dim))
            key_mat = ht.broadcastto_op(key_mat, query_lookups)
            # (bs * slot, npart, nkey, pdim)
            if self.mode == 'vq':
                # query metric: euclidean
                diff = ht.minus_op(query_lookups, key_mat)
                resp = ht.power_op(diff, 2)
                resp = ht.reduce_sum_op(resp, axes=[3])
                resp = ht.opposite_op(resp)
                # (bs * slot, npart, nkey)
            else:
                # query metric: dot
                dot = ht.mul_op(query_lookups, key_mat)
                resp = ht.reduce_sum_op(dot, axes=[3])
                # (bs * slot, npart, nkey)
            resp = self.bn_layer(resp)
            codes = ht.argmax_op(resp, 2)
            self.codebook_update = ht.sparse_set_op(self.codebooks, x, codes)
            # (bs * slot, npart)
            if self.mode == 'vq':
                if not self.share_weights:
                    codes = ht.add_op(codes, self.dbase)
                outputs = ht.embedding_lookup_op(self.value_matrix, codes)
                # (bs * slot, npart, pdim)
                outputs_final = ht.add_op(ht.stop_gradient_op(
                    ht.minus_op(outputs, inputs)), inputs)
                reg = ht.minus_op(outputs, ht.stop_gradient_op(inputs))
                reg = ht.power_op(reg, 2)
                self.reg = ht.reduce_mean_op(reg, axes=(0, 1, 2))
            else:
                resp_prob = ht.softmax_op(resp)
                # (bs * slot, npart, nkey)
                nb_idxs_onehot = ht.one_hot_op(codes, self.num_choices)
                # (bs * slot, npart, nkey)
                nb_idxs_onehot = ht.minus_op(resp_prob, ht.stop_gradient_op(
                    ht.minus_op(resp_prob, nb_idxs_onehot)))
                if self.share_weights:
                    outputs = ht.matmul_op(
                        # (bs * slot * npart, nkey)
                        ht.array_reshape_op(
                            nb_idxs_onehot, (-1, self.num_choices)),
                        self.value_matrix)  # (nkey, pdim)
                    # (bs * slot * npart, pdim)
                else:
                    outputs = ht.batch_matmul_op(
                        # (npart, bs * slot, nkey)
                        ht.transpose_op(nb_idxs_onehot, [1, 0, 2]),
                        ht.array_reshape_op(self.value_matrix, (-1, self.num_choices, self.part_embedding_dim)))  # (npart, nkey, pdim)
                    # (npart, bs * slot, pdim)
                    outputs = ht.transpose_op(outputs, [1, 0, 2])
                    # (bs * slot, npart, pdim)
                outputs_final = ht.array_reshape_op(
                    outputs, (-1, self.embedding_dim))
                # (bs * slot, dim)

            outputs_final = ht.array_reshape_op(
                outputs_final, (-1, self.embedding_dim))
            return outputs_final

    def make_inference(self, embed_input):
        with ht.context(self.ctx):
            codes = ht.embedding_lookup_op(self.codebooks, embed_input)
            # (bs, slot, npart)
            if not self.share_weights:
                codes = ht.add_op(codes, ht.reshape_to_op(self.dbase, codes))
            outputs = ht.embedding_lookup_op(self.value_matrix, codes)
            # (bs, slot, npart, pdim)
            outputs = ht.array_reshape_op(outputs, (-1, self.embedding_dim))
            # (bs * slot, dim)
            return outputs


class MGQEmbedding(DPQEmbedding):
    def __init__(self, num_embeddings, embedding_dim, high_num_choices, low_num_choices, num_parts, frequency, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        super().__init__(num_embeddings, embedding_dim, high_num_choices,
                         num_parts, batch_size, False, 'vq', initializer, name, ctx)
        self.low_num_choices = low_num_choices
        frequency = ht.array(frequency.reshape((-1, 1)),
                             dtype=np.int32, ctx=self.ctx)
        self.frequency = ht.placeholder_op(
            f'{name}_frequency', value=frequency, dtype=np.int32, trainable=False)

    def make_matries(self, initializer, name):
        if self.share_weights:
            shape = (self.num_choices, self.part_embedding_dim)
        else:
            shape = (self.num_parts * self.num_choices,
                     self.part_embedding_dim)
        return initializer(shape=shape, name='{}'.format(name), ctx=self.ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            # table: (nembed, dim), x: (bs, slot)
            query_lookups = ht.embedding_lookup_op(
                self.embedding_table, x)
            # (bs, slot, dim)
            inputs = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, self.part_embedding_dim))
            query_lookups = ht.array_reshape_op(
                query_lookups, (-1, self.num_parts, 1, self.part_embedding_dim))
            # (bs * slot, npart, 1, pdim)
            query_lookups = ht.tile_op(query_lookups, [self.num_choices, 1])
            # (bs * slot, npart, nkey, pdim)
            key_mat = ht.array_reshape_op(
                self.key_matrix, (-1, self.num_choices, self.part_embedding_dim))
            key_mat = ht.broadcastto_op(key_mat, query_lookups)
            # (bs * slot, npart, nkey, pdim)
            # query metric: euclidean
            diff = ht.minus_op(query_lookups, key_mat)
            resp = ht.power_op(diff, 2)
            resp = ht.reduce_sum_op(resp, axes=[3])
            resp = ht.opposite_op(resp)
            # (bs * slot, npart, nkey)
            resp = self.bn_layer(resp)
            # !! only argmax op is changed, compared with DPQ
            mask = ht.embedding_lookup_op(self.frequency, x)
            mask = ht.array_reshape_op(mask, (-1,))
            codes = ht.argmax_partial_op(
                resp, mask, self.low_num_choices, dim=2)
            self.codebook_update = ht.sparse_set_op(self.codebooks, x, codes)
            # (bs * slot, npart)
            codes = ht.add_op(codes, self.dbase)
            outputs = ht.embedding_lookup_op(self.value_matrix, codes)
            # (bs * slot, npart, pdim)
            outputs_final = ht.add_op(ht.stop_gradient_op(
                ht.minus_op(outputs, inputs)), inputs)
            reg = ht.minus_op(outputs, ht.stop_gradient_op(inputs))
            reg = ht.power_op(reg, 2)
            self.reg = ht.reduce_mean_op(reg, axes=(0, 1, 2))

            outputs_final = ht.array_reshape_op(
                outputs_final, (-1, self.embedding_dim))
            return outputs_final


class AdaptiveEmbedding(Embedding):
    def __init__(self, num_freq_emb, num_rare_emb, remap_indices, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        # DeepRec Adaptive Embedding
        self.num_freq_emb = num_freq_emb
        self.num_rare_emb = num_rare_emb
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.freq_emb = initializer(
            shape=(num_freq_emb, embedding_dim), name=f'{name}_freq', ctx=ctx)
        self.rare_emb = initializer(
            shape=(num_rare_emb, embedding_dim), name=f'{name}_rare', ctx=ctx)
        remap_indices = ht.array(remap_indices.reshape(
            (-1, 1)), dtype=np.int32, ctx=self.ctx)
        self.remap_indices = ht.placeholder_op(
            f'{name}_remap', value=remap_indices, dtype=np.int32, trainable=False)

    def __call__(self, x):
        with ht.context(self.ctx):
            remap = ht.embedding_lookup_op(self.remap_indices, x)
            high_freq = ht.embedding_lookup_op(self.freq_emb, remap)
            low_freq_inds = ht.mod_hash_negative_op(
                remap, self.num_rare_emb)
            low_freq = ht.embedding_lookup_op(self.rare_emb, low_freq_inds)
            result = ht.add_op(high_freq, low_freq)
            return ht.array_reshape_op(result, (-1, self.embedding_dim))

    def __repr__(self):
        return f'{self.name}({self.num_freq_emb},{self.num_rare_emb},{self.embedding_dim})'


class MDEmbedding(Embedding):
    def __init__(self, num_embeddings, compressed_dim, embedding_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.compressed_dim = compressed_dim
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(num_embeddings, compressed_dim), name=name, ctx=self.ctx)
        if compressed_dim < embedding_dim:
            self.projection = initializer(
                shape=(compressed_dim, embedding_dim), name=f'{name}_proj', ctx=self.ctx)
        else:
            self.projection = None

    def __call__(self, x):
        with ht.context(self.ctx):
            res = ht.embedding_lookup_op(self.embedding_table, x)
            if self.projection is not None:
                res = ht.matmul_op(res, self.projection)
        return res

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.compressed_dim},{self.embedding_dim})'


class AutoDimEmbedding(Embedding):
    def __init__(self, num_embeddings, dim_candidates, num_slot, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from .normalization import BatchNorm
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
        for dim, lookup in zip(self.dim_candidates, lookups.values()):
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


class OptEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, num_slot, batch_size, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.name = name
        self.ctx = ctx

        self.embedding_table = initializer(
            shape=(num_embeddings, embedding_dim), name=name, ctx=ctx)

        self.threshold = ht.init.zeros(
            shape=(self.num_slot, 1), name=f'{name}_threshold')
        self.potential_field_masks = ht.placeholder_op(
            name=f'{name}_pmask', value=self.pre_potential_field_mask(), trainable=False)

    def pre_potential_field_mask(self):
        masks = []
        for i in range(self.embedding_dim):
            zeros = np.zeros(self.embedding_dim - i - 1)
            ones = np.ones(i + 1)
            mask = np.concatenate(
                (ones, zeros), axis=0)[None, ...]
            masks.append(mask)
        total_masks = np.concatenate(masks, axis=0).astype(np.float32)
        return total_masks

    def get_batch_feature_mask(self, xv, tv):
        xv_norm = ht.reduce_norm1_op(xv, axes=2, keepdims=True)
        mask_f = ht.binary_step_op(ht.minus_op(
            xv_norm, ht.broadcastto_op(tv, xv_norm)))
        return mask_f

    def get_random_field_mask(self):
        indices = ht.randint_sample_op(
            shape=(self.batch_size, self.num_slot), low=0, high=self.embedding_dim)
        field_masks = ht.embedding_lookup_op(
            self.potential_field_masks, indices)
        return field_masks

    def __call__(self, x):
        with ht.context(self.ctx):
            xv = ht.embedding_lookup_op(self.embedding_table, x)
            mask_f = self.get_batch_feature_mask(xv, self.threshold)
            mask_f = ht.broadcastto_op(mask_f, xv)
            mask_e = self.get_random_field_mask()
            xe = ht.mul_op(mask_f, ht.mul_op(mask_e, xv))
        return xe

    def make_inference(self, x):
        with ht.context(self.ctx):
            xv = ht.embedding_lookup_op(self.embedding_table, x)
            mask_f = self.get_batch_feature_mask(xv, self.threshold)
            mask_f = ht.broadcastto_op(mask_f, xv)
            xe = ht.mul_op(mask_f, xv)
        return xe


class OptEmbeddingAfterRowPruning(OptEmbedding):
    def __init__(self, num_embeddings, embedding_dim, num_slot, batch_size, embedding_table, remap_indices, name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_slot = num_slot
        self.batch_size = batch_size
        self.name = name
        self.ctx = ctx

        self.target_shape = (batch_size, num_slot, embedding_dim)
        self.embedding_table = ht.placeholder_op(
            name=name, value=embedding_table, ctx=ctx)
        self.remap_indices = ht.placeholder_op(
            name=f'{name}_remap', value=remap_indices, dtype=np.int32, trainable=False, ctx=ctx)
        self.potential_field_masks = ht.placeholder_op(
            name=f'{name}_pmask', value=self.pre_potential_field_mask(), trainable=False, ctx=ctx)
        self.candidate = ht.init.GenEmpty()(shape=(self.num_slot,),
                                            name=f'{name}_candidate', trainable=False, dtype=np.int32, ctx=ctx)

    def __call__(self, x):
        with ht.context(self.ctx):
            new_indices = ht.embedding_lookup_op(self.remap_indices, x)
            xe = ht.embedding_lookup_op(self.embedding_table, new_indices)
            mask_e = ht.embedding_lookup_op(
                self.potential_field_masks, self.candidate)
            xe = ht.mul_op(ht.broadcast_shape_op(
                mask_e, self.target_shape), ht.array_reshape_op(xe, self.target_shape))
        return xe


class SparseEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, form, name='embedding', ctx=None):
        # only for inference
        self.form = form
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        from ..ndarray import ND_Sparse_Array
        embeddings = ND_Sparse_Array(
            self.num_embeddings, self.embedding_dim, ctx=self.ctx)
        self.sparse_embedding_table = ht.placeholder_op(
            f'{self.name}_sparse', value=embeddings)

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.sparse_embedding_lookup_op(self.sparse_embedding_table, x)

    def make_inference(self, embed_input):
        # here from train to inference
        with ht.context(self.ctx):
            # not for validate; convert to csr format for inference
            from ..ndarray import dense_to_sparse
            embeddings = dense_to_sparse(
                self.embedding_table.tensor_value, form=self.form)
            self.sparse_embedding_table = ht.placeholder_op(
                f'{self.name}_sparse', value=embeddings)
            return ht.sparse_embedding_lookup_op(self.sparse_embedding_table, embed_input)


class DeepLightEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, prune_rate, form, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert form in ('coo', 'csr')
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.prune_rate = prune_rate
        self.form = form

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.embedding_lookup_op(self.embedding_table, x)

    def make_adaptive_rate(self, batch_num):
        ignore_iter = 0 * batch_num

        def updater(n_iter):
            if n_iter <= ignore_iter:
                adaptive_sparse = 0
            else:
                real_niter = n_iter - ignore_iter
                if real_niter % 10 == 0 or real_niter % batch_num == 0:
                    adaptive_sparse = self.prune_rate * \
                        (1 - 0.99**(real_niter / 100.))
                else:
                    adaptive_sparse = 0
            return adaptive_sparse
        return updater

    def make_prune_op(self, y_):
        batch_num = y_.get_batch_num('train')
        rate_updater = self.make_adaptive_rate(batch_num)
        return ht.prune_low_magnitude_op(self.embedding_table, rate_updater)


class PEPEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, threshold_type, threshold_init, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        assert threshold_type in (
            'dimension', 'feature', 'global', 'feature_dimension')
        self.threshold_type = threshold_type
        if threshold_type == 'feature_dimension':
            th_shape = (self.num_embeddings, self.embedding_dim)
        elif threshold_type == 'dimension':
            th_shape = (self.embedding_dim,)
        elif threshold_type == 'feature':
            th_shape = (self.num_embeddings, 1)
        else:
            th_shape = (1,)
        self.threshold = ht.init.constant(
            th_shape, threshold_init, name=f'{name}_threshold')

    def __call__(self, x):
        with ht.context(self.ctx):
            raw_embeddings = ht.embedding_lookup_op(self.embedding_table, x)
            if self.threshold_type.startswith('feature'):
                cur_threshold = ht.embedding_lookup_op(self.threshold, x)
            else:
                cur_threshold = self.threshold
            cur_threshold = ht.sigmoid_op(cur_threshold)
            if self.threshold_type != 'feature_dimension':
                cur_threshold = ht.broadcastto_op(
                    cur_threshold, raw_embeddings)
            embeddings = ht.mul_op(ht.sign_op(raw_embeddings), ht.relu_op(
                ht.minus_op(ht.abs_op(raw_embeddings), cur_threshold)))
            return embeddings


class PEPRetrainEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, mask, form, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.mask = ht.placeholder_op(
            f'{name}_mask', value=mask, trainable=False, dtype=np.int32, ctx=self.ctx)
        self.form = form

    def __call__(self, x):
        with ht.context(self.ctx):
            lookups = ht.embedding_lookup_op(self.embedding_table, x)
            lookup_masks = ht.embedding_lookup_op(self.mask, x)
            return ht.mask_op(lookups, lookup_masks)


class AutoSrhEmbedding(SparseEmbedding):
    def __init__(self, num_embeddings, embedding_dim, nsplit, group_indices, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.nsplit = nsplit
        self.embedding_table = initializer(
            shape=(num_embeddings, embedding_dim), name=name, ctx=self.ctx)
        self.group_indices = ht.placeholder_op(
            name=f'{name}_groupind', value=group_indices.reshape(-1, 1), trainable=False, dtype=np.int32)
        self.alpha = ht.init.ones(
            shape=(self.nsplit, self.embedding_dim), name=f'{name}_alpha')

    def __call__(self, x):
        with ht.context(self.ctx):
            embeddings = ht.embedding_lookup_op(self.embedding_table, x)
            alpha_indices = ht.embedding_lookup_op(self.group_indices, x)
            alphas = ht.embedding_lookup_op(self.alpha, alpha_indices)
            return ht.mul_op(embeddings, ht.reshape_to_op(alphas, embeddings))


class AutoSrhRetrainEmbedding(AutoSrhEmbedding):
    def __init__(self, num_embeddings, embedding_dim, nsplit, group_indices, form='csr', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        super().__init__(num_embeddings, embedding_dim,
                         nsplit, group_indices, initializer, name, ctx)
        self.form = form
        self.alpha.trainable = False


class QuantizedEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, digit, scale=0.01, middle=0, use_qparam=False, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert digit in (8, 16)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.digit = digit
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        if use_qparam:
            self.qparams = ht.init.GenEmpty()(shape=(self.num_embeddings, 2),
                                              name='qparams', trainable=False, ctx=ctx)
        else:
            self.scale = scale
            self.middle = middle
        self.use_qparam = use_qparam

    def __call__(self, x):
        with ht.context(self.ctx):
            if self.use_qparam:
                lookup = ht.quantized_embedding_lookup_op(
                    self.embedding_table, x, self.qparams, self.digit)
            else:
                lookup = ht.unified_quantized_embedding_lookup_op(
                    self.embedding_table, x, self.scale, self.middle, self.digit)
            return lookup

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.embedding_dim},{self.digit})'


class ALPTEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, digit, init_scale, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert digit in (8, 16)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.digit = digit
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.scale = ht.init.constant(shape=(
            self.num_embeddings, 1), fill_value=init_scale, name=f'{self.name}_scale', trainable=False, ctx=ctx)
        self.middle = 0

    def __call__(self, x):
        with ht.context(self.ctx):
            lookup = ht.alpt_embedding_lookup_op(
                self.embedding_table, x, self.scale, self.middle, self.digit)
            return lookup

    def __repr__(self):
        return f'{self.name}({self.num_embeddings},{self.embedding_dim},{self.digit})'
