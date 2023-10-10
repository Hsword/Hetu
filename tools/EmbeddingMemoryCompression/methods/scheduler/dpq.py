from .switchinference import SwitchInferenceTrainer
from .compressor import Compressor
from ..layers import DPQEmbedding
from hetu.gpu_ops import add_op, sum_op, concatenate_op


class DPQTrainer(SwitchInferenceTrainer):
    def _get_codebook_memory(self):
        npart = self.embedding_args['num_parts']
        codebooks = self.embedding_args['num_choices'] * \
            (self.embedding_dim // npart)
        if not self.embedding_args['share_weights']:
            codebooks *= npart
        return codebooks

    def get_single_embed_layer(self, nemb, batch_size, name):
        return DPQEmbedding(
            nemb,
            self.embedding_dim,
            self.embedding_args['num_choices'],
            self.embedding_args['num_parts'],
            batch_size,
            share_weights=self.embedding_args['share_weights'],
            mode=self.embedding_args['mode'],
            initializer=self.initializer,
            name=name,
            ctx=self.ectx,
        )

    def get_embed_layer(self):
        if self.use_multi:
            codebook = self._get_codebook_memory()
            emb = []
            threshold = self.embedding_args['threshold']
            for i, nemb in enumerate(self.num_embed_separate):
                orimem = nemb * self.embedding_dim
                newmem = nemb * self.embedding_args['num_parts'] + codebook
                if nemb > threshold and orimem > newmem:
                    emb.append(self.get_single_embed_layer(
                        nemb, self.batch_size, f'DPQEmb_{i}'))
                else:
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            emb = self.get_single_embed_layer(
                self.num_embed, self.batch_size * self.num_slot, 'DPQEmb')
        return emb

    def _get_inference_embeddings(self, embed_input):
        if self.use_multi:
            embeddings = []
            for emb, x in zip(self.embed_layer, embed_input):
                if isinstance(emb, DPQEmbedding):
                    embeddings.append(emb.make_inference(x))
                else:
                    embeddings.append(emb(x))
            embeddings = concatenate_op(embeddings, axis=-1)
        else:
            embeddings = self.embed_layer.make_inference(embed_input)
        return embeddings

    def get_eval_nodes(self):
        embed_input, dense_input, y_ = self.data_ops
        embeddings = self.get_embeddings(embed_input)
        loss, prediction = self.model(
            embeddings, dense_input, y_)
        if self.embedding_args['mode'] == 'vq':
            if self.use_multi:
                regs = [emblayer.reg for emblayer in self.embed_layer if isinstance(
                    emblayer, DPQEmbedding)]
                reg = sum_op(regs)
            else:
                reg = self.embed_layer.reg
            loss = add_op(loss, reg)
        train_op = self.opt.minimize(loss)
        train_nodes = [loss, prediction, y_, train_op]
        if self.use_multi:
            for emblayer in self.embed_layer:
                if isinstance(emblayer, DPQEmbedding):
                    train_nodes.append(emblayer.codebook_update)
        else:
            train_nodes.append(self.embed_layer.codebook_update)
        eval_nodes = {
            self.train_name: train_nodes,
        }
        test_embed_input = self._get_inference_embeddings(embed_input)
        test_loss, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes[self.validate_name] = [test_loss, test_prediction, y_]
        eval_nodes[self.test_name] = [test_loss, test_prediction, y_]
        return eval_nodes

    def get_eval_nodes_inference(self):
        embed_input, dense_input, y_ = self.data_ops
        test_embed_input = self._get_inference_embeddings(embed_input)
        test_loss, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes = {
            self.test_name: [test_loss, test_prediction, y_],
        }
        return eval_nodes


class ProductQuantizer(Compressor):
    @staticmethod
    def compress(embedding, subvector_num, subvector_bits):
        import faiss
        index = faiss.index_factory(
            embedding.shape[1], f"PQ{subvector_num}x{subvector_bits}")
        index.train(embedding)
        index.add(embedding)
        memory = embedding.shape[0] * index.code_size / 4 + \
            embedding.shape[1] * (2 ** subvector_bits)
        print('Final compression ratio:', memory /
              embedding.shape[0] / embedding.shape[1])
        return index

    @staticmethod
    def decompress(index):
        return index.reconstruct_n(0, index.ntotal)

    @staticmethod
    def decompress_batch(index, batch_ids):
        if 'reconstruct_batch' in dir(index):
            return index.reconstruct_batch(batch_ids)
        else:
            import numpy as np
            batch_ids = batch_ids.reshape(-1)
            results = [index.reconstruct(i) for i in batch_ids]
            return np.stack(results)
