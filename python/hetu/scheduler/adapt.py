from .base import EmbeddingTrainer
from ..layers import AdaptiveEmbedding
import math


class AdaptEmbTrainer(EmbeddingTrainer):
    def get_data(self):
        top_percent = self.embedding_args['top_percent']
        embed_input, dense_input, y_ = super().get_data()
        exact_split = bool(self.embedding_args['exact_split'])
        if self.use_multi:
            remap_indices = self.dataset.get_separate_remap(
                [op.dataloaders[self.train_name].raw_data for op in embed_input], top_percent, exact_split=exact_split)
        else:
            remap_indices = self.dataset.get_whole_remap(
                embed_input.dataloaders[self.train_name].raw_data, top_percent, exact_split=exact_split)
        self.remap_indices = remap_indices
        return embed_input, dense_input, y_

    def get_single_embed_layer(self, nfreq, nrare, remap_indices, name):
        return AdaptiveEmbedding(
            nfreq,
            nrare,
            remap_indices,
            self.embedding_dim,
            initializer=self.initializer,
            name=name,
            ctx=self.ectx,
        )

    def _split_freq_rare(self, nemb, remap):
        # nfreq_emb = math.ceil(nemb * self.embedding_args['top_percent'])
        # assert nfreq_emb == remap.max() + 1
        nfreq_emb = remap.max() + 1
        nrare_emb = math.ceil(nemb * self.compress_rate) - nfreq_emb
        assert nrare_emb >= 0
        return nfreq_emb, nrare_emb

    def get_embed_layer(self):
        assert self.embedding_args['top_percent'] < self.compress_rate
        if self.use_multi:
            emb = []
            threshold = self.embedding_args['threshold']
            for i, nemb in enumerate(self.num_embed_separate):
                nfreq, nrare = self._split_freq_rare(
                    nemb, self.remap_indices[i])
                if nemb > threshold and nrare > 0 and nfreq + nrare < nemb:
                    emb.append(self.get_single_embed_layer(
                        nfreq, nrare, self.remap_indices[i], f'AdaptEmb_{i}'))
                else:
                    emb.append(super().get_single_embed_layer(
                        nemb, f'Embedding_{i}'))
        else:
            nfreq, nrare = self._split_freq_rare(
                self.num_embed, self.remap_indices)
            assert nrare > 0
            emb = self.get_single_embed_layer(
                nfreq, nrare, self.remap_indices, 'AdaptEmb')
        return emb
