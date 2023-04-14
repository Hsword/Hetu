from .embedding import Embedding
import hetu as ht
import math
import numpy as np
import os.path as osp


class RobeEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, compress_rate, Z, random_numbers, use_slot_coef=True, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        robe_array_size = int(
            num_embeddings * embedding_dim * compress_rate)
        self.num_embeddings = num_embeddings
        self.robe_array_size = robe_array_size
        self.embedding_dim = embedding_dim
        assert Z <= embedding_dim
        self.Z = Z
        self.use_slot_coef = use_slot_coef

        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.robe_array_size, 1), name=self.name, ctx=ctx)
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


class HashEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, compress_rate=None, size_limit=None, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        assert compress_rate is None or size_limit is None
        if size_limit is not None:
            real_num_embeds = size_limit // embedding_dim
        else:
            if compress_rate is None:
                compress_rate = 1.0
            real_num_embeds = int(num_embeddings * compress_rate)
        self.num_embeddings = num_embeddings
        self.real_num_embeds = real_num_embeds
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.real_num_embeds, self.embedding_dim), name=self.name, ctx=ctx)

    def __call__(self, x):
        # ref MLSys20, HierPS
        with ht.context(self.ctx):
            sparse_input = ht.mod_hash_op(x, self.real_num_embeds)
            return ht.embedding_lookup_op(self.embedding_table, sparse_input)


class MultipleHashEmbedding(Embedding):
    def __init__(self, num_embed_fields, embedding_dim, compress_rate=None, size_limit=None, initializer=ht.init.GenXavierNormal(), names='embedding', ctx=None):
        assert compress_rate is None or size_limit is None
        if size_limit is not None:
            compress_rate = size_limit / sum(num_embed_fields)
        real_num_embeds = [math.ceil(nemb * compress_rate)
                           for nemb in num_embed_fields]
        self.num_embed_fields = num_embed_fields
        self.real_num_embeds = real_num_embeds
        self.embedding_dim = embedding_dim
        if not isinstance(names, list):
            names = [f'{names}_{i}' for i in range(len(num_embed_fields))]
        self.name = names
        self.ctx = ctx
        self.embedding_table = [
            initializer(
                shape=(nemb, self.embedding_dim),
                name=nam,
                ctx=ctx,
            ) for nemb, nam in zip(self.real_num_embeds, self.name)
        ]

    def __call__(self, xs):
        with ht.context(self.ctx):
            results = []
            for emb, x, rnum in zip(self.embedding_table, xs, self.real_num_embeds):
                x = ht.mod_hash_op(x, rnum)
                results.append(ht.embedding_lookup_op(emb, x))
            result = ht.concatenate_op(results, axis=1)
            return result


class CompositionalEmbedding(Embedding):
    # compositional embedding
    def __init__(self, num_embed_fields, embedding_dim, compress_rate, aggregator='mul', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        # KDD20, CompositionalHash
        # CIKM21, BinaryCodeHash
        # adapted from DLRM QREmbeddingBag
        aggregator = aggregator[:3]
        assert aggregator in ('sum', 'mul')
        self.aggregator = aggregator
        self.num_slot = len(num_embed_fields)
        self.num_embed_fields = num_embed_fields
        self.num_embeddings = sum(num_embed_fields)
        self.embedding_dim = embedding_dim
        self.compress_rate = compress_rate
        self.name = name
        self.ctx = ctx
        self.collision = self.get_collision(
            self.num_embeddings, self.num_slot, compress_rate)
        self.embedding_tables = []
        for i, nemb in enumerate(self.num_embed_fields):
            cur_compo = self.decompo(nemb, self.collision)
            if isinstance(cur_compo, tuple):
                nqemb, nremb = cur_compo
                qemb = initializer(
                    shape=(nqemb, self.embedding_dim),
                    name=f'{name}_{i}_q',
                    ctx=ctx,
                )
                remb = initializer(
                    shape=(nremb, self.embedding_dim),
                    name=f'{name}_{i}_r',
                    ctx=ctx,
                )
                self.embedding_tables.append((qemb, remb))
            else:
                cur_embed = initializer(
                    shape=(nemb, self.embedding_dim),
                    name=f'{name}_{i}',
                    ctx=ctx,
                )
                self.embedding_tables.append(cur_embed)

    def get_collision(self, num_embeddings, num_slot, compress_rate):
        from sympy.solvers import solve
        from sympy import Symbol
        x = Symbol('x')
        results = solve(num_embeddings / x + num_slot *
                        x - compress_rate * num_embeddings)
        results = filter(lambda x: x > 0, results)
        res = int(round(min(results)))
        print(f'Collision {res} given compression rate {compress_rate}.')
        return res

    def decompo(self, num, collision):
        if num <= collision:
            return num
        another = math.ceil(num / collision)
        if num <= another + collision:
            return num
        return (another, collision)

    def __call__(self, xs):
        with ht.context(self.ctx):
            results = []
            for emb, x in zip(self.embedding_tables, xs):
                if not isinstance(emb, tuple):
                    results.append(ht.embedding_lookup_op(emb, x))
                else:
                    qemb, remb = emb
                    qind = ht.div_hash_op(x, self.collision)
                    rind = ht.mod_hash_op(x, self.collision)
                    q = ht.embedding_lookup_op(qemb, qind)
                    r = ht.embedding_lookup_op(remb, rind)
                    if self.aggregator == 'sum':
                        cur_embedding = ht.add_op(q, r)
                    elif self.aggregator == 'mul':
                        cur_embedding = ht.mul_op(q, r)
                    results.append(cur_embedding)
            result = ht.concatenate_op(results, axis=1)
            return result


class LearningEmbedding(Embedding):
    # deep learning embedding
    def __init__(self, embedding_dim, num_buckets, num_hash, mlp_dim=1024, dist='uniform', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from .linear import Linear
        from .normalization import BatchNorm
        from .relu import Relu
        from .sequence import Sequence
        assert dist in ('uniform', 'normal')
        self.distribution = dist
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.num_hash = num_hash
        self.mlp_dim = mlp_dim
        self.name = name
        self.ctx = ctx
        self.slopes = ht.Variable(name='slopes', value=np.random.randint(
            0, num_buckets, size=num_hash), trainable=False)
        self.biases = ht.Variable(name='biases', value=np.random.randint(
            0, num_buckets, size=num_hash), trainable=False)
        prime_path = osp.join(osp.dirname(osp.abspath(
            __file__)), 'primes.npy')
        allprimes = np.load(prime_path)
        for i, p in enumerate(allprimes):
            if p >= num_buckets:
                break
        allprimes = allprimes[i:]
        self.primes = ht.Variable(name='primes', value=np.random.choice(
            allprimes, size=num_hash), trainable=False)
        self.layers = Sequence(
            Linear(self.num_hash, self.mlp_dim),
            BatchNorm(self.mlp_dim),
            Relu(),
            Linear(self.mlp_dim, self.mlp_dim),
            BatchNorm(self.mlp_dim),
            Relu(),
            Linear(self.mlp_dim, self.mlp_dim),
            BatchNorm(self.mlp_dim),
            Relu(),
            Linear(self.mlp_dim, self.mlp_dim),
            BatchNorm(self.mlp_dim),
            Relu(),
            Linear(self.mlp_dim, self.embedding_dim),
            BatchNorm(self.embedding_dim),
            Relu(),
        )  # TODO: use mish instead

    def __call__(self, x):
        # KDD21, DHE
        x = ht.learn_hash_op(x, self.slopes, self.biases,
                             self.primes, self.num_buckets, self.distribution)
        x = ht.array_reshape_op(x, (-1, self.num_hash))
        x = self.layers(x)
        return x


class DPQEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, num_choices, num_parts, num_slot, batch_size, share_weights=False, mode='vq', initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from ..initializers import nulls
        from .normalization import BatchNorm
        assert mode in ('vq', 'sx')
        assert embedding_dim % num_parts == 0
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_choices = num_choices
        self.num_parts = num_parts
        self.num_slot = num_slot
        self.batch_size = batch_size
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
            name), ctx=ctx, trainable=False)
        if not self.share_weights:
            dbase = np.array(
                [self.num_choices * d for d in range(self.num_parts)], dtype=int)
            dbase = np.tile(dbase, [self.batch_size * self.num_slot, 1])
            dbase = ht.array(dbase, ctx=self.ctx)
            self.dbase = ht.placeholder_op(
                'dbase', value=dbase, trainable=False)

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

            return outputs_final

    def make_inference(self, embed_input):
        with ht.context(self.ctx):
            codes = ht.embedding_lookup_op(self.codebooks, embed_input)
            # (bs, slot, npart)
            if not self.share_weights:
                codes = ht.add_op(codes, ht.array_reshape_op(
                    self.dbase, (-1, self.num_slot, self.num_parts)))
            outputs = ht.embedding_lookup_op(self.value_matrix, codes)
            # (bs, slot, npart, pdim)
            outputs = ht.array_reshape_op(outputs, (-1, self.embedding_dim))
            # (bs * slot, dim)
            return outputs

    def get_eval_nodes(self, data_ops, model, opt):
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(self(embed_input), dense_input, y_)
        if self.mode == 'vq':
            loss = ht.add_op(loss, self.reg)
        train_op = opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op, self.codebook_update],
        }
        test_embed_input = self.make_inference(embed_input)
        test_loss, test_prediction = model(
            test_embed_input, dense_input, y_)
        eval_nodes['validate'] = [test_loss, test_prediction, y_]
        return eval_nodes

    def get_eval_nodes_inference(self, data_ops, model):
        embed_input, dense_input, y_ = data_ops
        test_embed_input = self.make_inference(embed_input)
        test_loss, test_prediction = model(
            test_embed_input, dense_input, y_)
        eval_nodes = {
            'validate': [test_loss, test_prediction, y_],
        }
        return eval_nodes


class AutoDimEmbedding(Embedding):
    def __init__(self, separate_num_embeds, dim_candidates, num_slot, batch_size, alpha_lr, r=1e-2, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        from .normalization import BatchNorm
        self.num_embeddings = sum(separate_num_embeds)
        self.separate_num_embeds = separate_num_embeds
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
        self.alpha_lr = alpha_lr
        self.r = r
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

    def get_eval_nodes(self, data_ops, model, opt):
        from ..gpu_ops.AssignWithIndexedSlices import AssignWithIndexedSlicesOp
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(self(embed_input), dense_input, y_)
        train_op = opt.minimize(loss)
        lookups = []
        dedup_lookups = []
        dembed_ops = []
        dup_dembed_ops = []
        dparam_ops = []
        dalpha_op = None
        param_opts = []
        for op in train_op:
            if op.inputs[0] is self.alpha:
                dalpha_op = op.inputs[1]
            else:
                param_opts.append(op)
                if isinstance(op, AssignWithIndexedSlicesOp):
                    sparse_opt = op.inputs[1]
                    deduplookup = sparse_opt.inputs[1]
                    lookups.append(deduplookup.inputs[2])
                    dedup_lookups.append(deduplookup)
                    dedupgrad = sparse_opt.inputs[2]
                    dembed_ops.append(dedupgrad)
                    dup_dembed_ops.append(deduplookup.inputs[0])
                    assert deduplookup.inputs[0] is dedupgrad.inputs[1]
                else:
                    dparam_ops.append(op.inputs[1])
        assert dalpha_op is not None

        self.var_lookups = {dim: ht.init.GenEmpty()(
            (self.batch_size, self.num_slot, dim), f'lookups{dim}', False, self.ctx) for dim in self.dim_candidates}
        new_loss, new_pred = model(self.make_embed(
            self.var_lookups), dense_input, y_)
        alpha_grad = ht.gradients(new_loss, [self.alpha])

        eval_nodes = {
            'train': [loss, prediction, y_, param_opts],
            'lookups': lookups + dedup_lookups + param_opts,
            'validate': [loss, prediction, y_],
            'allgrads': [dalpha_op] + dup_dembed_ops + dembed_ops + dparam_ops,
            'alpha': [alpha_grad],
        }

        return eval_nodes

    def make_retrain(self, xs, stream):
        separate_num_embeds = self.separate_num_embeds
        from ..gpu_links import argmax
        dim_choice = ht.empty((self.num_slot, ), ctx=self.ctx, dtype=np.int32)
        argmax(self.alpha.tensor_value, dim_choice, 1, stream=stream)
        stream.sync()
        dim_choice = [self.dim_candidates[int(ind)]
                      for ind in dim_choice.asnumpy()]
        print('Dimension choices:', dim_choice)
        new_embedding_tables = []
        new_weights = []
        new_biases = []
        # ## previous code, copy parameters from the first stage; actually no need?
        # cur_offset = 0
        # for i, (nembed, dim) in enumerate(zip(separate_num_embeds, dim_choice)):
        #     cur_embed_table = ht.empty((nembed, dim), ctx=self.ctx)
        #     cur_embed_table._async_copyfrom_offset(
        #         self.embedding_tables[dim].tensor_value, stream, cur_offset * dim, 0, nembed * dim)
        #     new_embedding_tables.append(ht.placeholder_op(
        #         'new_embed_{}'.format(i), value=cur_embed_table, ctx=self.ctx))
        #     cur_weight = ht.empty((dim, self.max_dim), ctx=self.ctx)
        #     cur_weight._async_copyfrom_offset(
        #         self.weights[dim].tensor_value, stream, i * dim * self.max_dim, 0, dim * self.max_dim)
        #     new_weights.append(ht.placeholder_op(
        #         'new_weight_{}'.format(i), value=cur_weight))
        #     cur_bias = ht.empty((self.max_dim, ), ctx=self.ctx)
        #     cur_bias._async_copyfrom_offset(
        #         self.biases[dim].tensor_value, stream, i * self.max_dim, 0, self.max_dim)
        #     new_biases.append(ht.placeholder_op(
        #         'new_bias_{}'.format(i), value=cur_bias))
        #     cur_offset += nembed
        # stream.sync()
        for i, (nembed, dim) in enumerate(zip(separate_num_embeds, dim_choice)):
            cur_embed_table = self.initializer(
                shape=(nembed, dim), name=f'new_embed_{i}', ctx=self.ctx)
            cur_weight = self.initializer(
                shape=(dim, self.max_dim), name=f'new_weight_{i}', ctx=self.ctx)
            cur_bias = ht.init.zeros(
                shape=(self.max_dim, ), name=f'new_bias_{i}', ctx=self.ctx)
            new_embedding_tables.append(cur_embed_table)
            new_weights.append(cur_weight)
            new_biases.append(cur_bias)
        self.embedding_tables = new_embedding_tables
        self.weights = new_weights
        self.biases = new_biases
        all_lookups = []
        for x, table, weight, bias in zip(xs, self.embedding_tables, self.weights, self.biases):
            lookups = ht.embedding_lookup_op(table, x)
            # (bs, cdim)
            lookups = ht.linear_op(lookups, weight, bias)
            # (bs, dim)
            all_lookups.append(lookups)
        all_lookups = ht.concatenate_op(all_lookups, 1)
        return all_lookups

    def get_eval_nodes_retrain(self, data_ops, model, opt, stream):
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(
            self.make_retrain(embed_input, stream), dense_input, y_)
        train_op = opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes


class MDEmbedding(Embedding):
    def __init__(self, num_embed_fields, embedding_dim, alpha, round_dim, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        dims = self.md_solver(num_embed_fields, alpha,
                              num_dim=embedding_dim, round_dim=round_dim)
        self.ctx = ctx
        embeds = []
        projs = []
        base_dim = embedding_dim
        assert base_dim == max(dims)
        for i, (n, d) in enumerate(zip(num_embed_fields, dims)):
            embeds.append(initializer(
                shape=(n, d), name=f'{name}_fields_{i}', ctx=ctx))
            if dims[i] < base_dim:
                projs.append(initializer(
                    shape=(dims[i], base_dim), name=f'{name}_proj_{i}'))
            else:
                projs.append(None)
        self.embeds = embeds
        self.projs = projs

    def md_solver(self, num_embed_fields, alpha, num_dim=None, mem_cap=None, round_dim=True, freq=None):
        # inherited from dlrm repo
        indices, num_embed_fields = zip(
            *sorted(enumerate(num_embed_fields), key=lambda x: x[1]))
        num_embed_fields = np.array(num_embed_fields)
        if freq is not None:
            num_embed_fields /= freq[indices]
        if num_dim is not None:
            # use max dimension
            lamb = num_dim * (num_embed_fields[0] ** alpha)
        elif mem_cap is not None:
            # use memory capacity
            lamb = mem_cap / np.sum(num_embed_fields ** (1 - alpha))
        else:
            raise ValueError("Must specify either num_dim or mem_cap")
        d = lamb * (num_embed_fields ** (-alpha))
        d = np.round(np.maximum(d, 1))
        if round_dim:
            d = 2 ** np.round(np.log2(d))
        d = d.astype(int)
        undo_sort = [0] * len(indices)
        for i, v in enumerate(indices):
            undo_sort[v] = i
        return d[undo_sort]

    def __call__(self, xs):
        results = []
        for x, embed, proj in zip(xs, self.embeds, self.projs):
            with ht.context(self.ctx):
                res = ht.embedding_lookup_op(embed, x)
            if proj is not None:
                res = ht.matmul_op(res, proj)
            results.append(res)
        result = ht.concatenate_op(results, axis=0)
        return result


class DeepLightEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim, target_sparse, warm=2, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ctx = ctx
        self.embedding_table = initializer(
            shape=(self.num_embeddings, self.embedding_dim), name=self.name, ctx=ctx)
        self.target_sparse = target_sparse
        self.warm = warm
        real_dim = self.target_sparse * embedding_dim
        if real_dim >= 3:
            self.form = 'csr'
            self.real_target_sparse = (real_dim - 1) / 2 / embedding_dim
        else:
            self.form = 'coo'
            self.real_target_sparse = self.target_sparse / 3
        self.prune_rate = 1 - self.real_target_sparse
        print(f'Use {self.form} for sparse storage; final prune rate {self.prune_rate}, given target sparse rate {target_sparse}.')

    def __call__(self, x):
        with ht.context(self.ctx):
            return ht.embedding_lookup_op(self.embedding_table, x)

    def make_adaptive_rate(self, batch_num):
        ignore_iter = self.warm * batch_num

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

    def make_inference(self, embed_input, load_value=True):
        with ht.context(self.ctx):
            # not for validate; convert to csr format for inference
            if load_value:
                from ..ndarray import dense_to_sparse
                embeddings = dense_to_sparse(
                    self.embedding_table.tensor_value, form=self.form)
            else:
                from ..ndarray import ND_Sparse_Array
                embeddings = ND_Sparse_Array(
                    self.num_embeddings, self.embedding_dim, ctx=self.ctx)
            self.sparse_embedding_table = ht.Variable(
                'sparse_embedding', value=embeddings)
            return ht.sparse_embedding_lookup_op(self.sparse_embedding_table, embed_input)

    def get_eval_nodes(self, data_ops, model, opt):
        embed_input, dense_input, y_ = data_ops
        loss, prediction = model(self(embed_input), dense_input, y_)
        train_op = opt.minimize(loss)
        eval_nodes = {
            'train': [loss, prediction, y_, train_op, self.make_prune_op(y_)],
            'validate': [loss, prediction, y_],
        }
        return eval_nodes

    def get_eval_nodes_inference(self, data_ops, model, load_value=True):
        # check inference; use sparse embedding
        embed_input, dense_input, y_ = data_ops
        test_embed_input = self.make_inference(embed_input, load_value)
        test_loss, test_prediction = model(
            test_embed_input, dense_input, y_)
        eval_nodes = {'validate': [test_loss, test_prediction, y_]}
        return eval_nodes


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


class TensorTrainEmbedding(Embedding):
    def __init__(self, num_embed_fields, embedding_dim, compress_rate, ttcore_initializer, initializer=ht.init.GenXavierNormal(), name='embedding', ctx=None):
        self.num_tables = 3
        self.num_embeddings = sum(num_embed_fields)
        self.decomp_nembs = [self.get_decomp_emb(
            nemb) for nemb in num_embed_fields]
        self.decomp_ndim = self.get_decomp_dim(embedding_dim)
        rank = self.get_rank(
            self.num_embeddings, self.decomp_nembs, self.decomp_ndim, compress_rate)
        self.ranks = [1, rank, rank, 1]
        self.name = name
        self.ctx = ctx
        self.embedding_tables = []
        for j, (nemb, dn) in enumerate(zip(num_embed_fields, self.decomp_nembs)):
            cur_shapes = []
            size = 0
            for i in range(self.num_tables):
                nrow = dn[i]
                ndim = self.decomp_ndim[i]
                prerank = self.ranks[i]
                postrank = self.ranks[i+1]
                ncol = prerank * ndim * postrank
                cur_shapes.append((nrow, ncol))
                size += nrow * ncol
            if size < nemb * embedding_dim:
                cur_tables = tuple(ttcore_initializer(
                    shape=sh, name=f'{name}_field{j}_{i}') for i, sh in enumerate(cur_shapes))
            else:
                cur_tables = initializer(
                    shape=(nemb, embedding_dim), name=f'{name}_field{j}')
            self.embedding_tables.append(cur_tables)

    def get_decomp_dim(self, embedding_dim):
        assert embedding_dim >= 8 and embedding_dim & (embedding_dim - 1) == 0
        decomp_ndim = [2, 2, 2]
        idx = 2
        embedding_dim //= 8
        while embedding_dim != 1:
            decomp_ndim[idx] *= 2
            embedding_dim //= 2
            idx = (idx - 1) % 3
        return decomp_ndim

    def get_decomp_emb(self, nemb):
        n1 = math.ceil(nemb ** (1/3))
        n2 = math.ceil((nemb / n1) ** (1/2))
        n3 = math.ceil(nemb / n1 / n2)
        return [n3, n2, n1]

    def get_rank(self, num_embeddings, decomp_nembs, decomp_ndim, compress_rate):
        linear_coef = 0
        quadra_coef = 0
        for dn in decomp_nembs:
            linear_coef += (dn[0] * decomp_ndim[0] + dn[2] * decomp_ndim[2])
            quadra_coef += (dn[1] * decomp_ndim[1])
        from sympy.solvers import solve
        from sympy import Symbol
        x = Symbol('x')
        results = solve(quadra_coef * x * x + linear_coef *
                        x - compress_rate * num_embeddings)
        results = filter(lambda x: x > 0, results)
        res = int(round(min(results)))
        print(f'Rank {res} given compression rate {compress_rate}.')
        return res

    def __call__(self, xs):
        results = []
        for j, (emb, x) in enumerate(zip(self.embedding_tables, xs)):
            if isinstance(emb, tuple):
                indices = x
                accum_embed = None
                accum_dim = 1
                nemb = self.decomp_nembs[j]
                for i in range(self.num_tables):
                    if i == self.num_tables - 1:
                        cur_ind = indices
                    else:
                        cur_ind = ht.mod_hash_op(indices, nemb[i])
                        indices = ht.div_hash_op(indices, nemb[i])
                    partial_embed = ht.embedding_lookup_op(emb[i], cur_ind)
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
                results.append(accum_embed)
            else:
                results.append(ht.embedding_lookup_op(emb, x))
        result = ht.concatenate_op(results, axis=1)
        return result
