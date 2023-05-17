from .switchinference import SwitchInferenceTrainer
from ..layers import MGQEmbedding
from ..gpu_ops import add_op, sum_op, concatenate_op


class MGQETrainer(SwitchInferenceTrainer):
    def get_data(self):
        top_percent = self.embedding_args['top_percent']
        embed_input, dense_input, y_ = super().get_data()
        if self.use_multi:
            freq_data = self.dataset.get_separate_frequency_split(
                [op.dataloaders[self.train_name].raw_data for op in embed_input], top_percent)
        else:
            freq_data = self.dataset.get_whole_frequency_split(
                embed_input.dataloaders[self.train_name].raw_data, top_percent)
        self.freq_data = freq_data
        return embed_input, dense_input, y_

    def _get_codebook_memory(self):
        codebooks = self.embedding_args['high_num_choices'] * \
            self.embedding_dim
        return codebooks

    def get_single_embed_layer(self, nemb, frequency, batch_size, name):
        return MGQEmbedding(
            nemb,
            self.embedding_dim,
            self.embedding_args['high_num_choices'],
            self.embedding_args['low_num_choices'],
            self.embedding_args['num_parts'],
            frequency,
            batch_size,
            initializer=self.initializer,
            name=name,
            ctx=self.ectx,
        )

    def get_embed_layer(self):
        if self.use_multi:
            codebook = self._get_codebook_memory()
            emb = []
            for i, nemb in enumerate(self.num_embed_separate):
                orimem = nemb * self.embedding_dim
                newmem = nemb * \
                    (self.embedding_args['num_parts'] + 1) + codebook
                if orimem > newmem:
                    emb.append(self.get_single_embed_layer(
                        nemb, self.freq_data[i], self.batch_size, f'MGQEmb_{i}'))
                else:
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            emb = self.get_single_embed_layer(
                self.num_embed, self.freq_data, self.batch_size * self.num_slot, 'MGQEmb')
        return emb

    def _get_inference_embeddings(self, embed_input):
        if self.use_multi:
            embeddings = []
            for emb, x in zip(self.embed_layer, embed_input):
                if isinstance(emb, MGQEmbedding):
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
        loss,loss2, prediction = self.model(
            embeddings, dense_input, y_)
        if self.use_multi:
            regs = [emblayer.reg for emblayer in self.embed_layer if isinstance(
                emblayer, MGQEmbedding)]
            reg = sum_op(regs)
        else:
            reg = self.embed_layer.reg
        loss = add_op(loss, reg)
        loss2 = add_op(loss2,reg)
        train_op = self.opt.minimize(loss)
        train_nodes = [loss,loss2, prediction, y_, train_op]
        if self.use_multi:
            for emblayer in self.embed_layer:
                if isinstance(emblayer, MGQEmbedding):
                    train_nodes.append(emblayer.codebook_update)
        else:
            train_nodes.append(self.embed_layer.codebook_update)
        eval_nodes = {
            self.train_name: train_nodes,
        }
        test_embed_input = self._get_inference_embeddings(embed_input)
        test_loss,test_loss2, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes[self.validate_name] = [test_loss,test_loss2, test_prediction, y_]
        eval_nodes[self.test_name] = [test_loss,test_loss2, test_prediction, y_]
        return eval_nodes

    def get_eval_nodes_inference(self):
        embed_input, dense_input, y_ = self.data_ops
        test_embed_input = self._get_inference_embeddings(embed_input)
        test_loss,test_loss2, test_prediction = self.model(
            test_embed_input, dense_input, y_)
        eval_nodes = {
            self.test_name: [test_loss,test_loss2, test_prediction, y_],
        }
        return eval_nodes
