# only for DLRMs now

from .base import BaseTrainer
from .deeplight import DeepLightTrainer


def get_trainer(layer):
    from ..layers.compressed_embedding import \
        AutoDimEmbedding, \
        DeepLightEmbedding

    trainer_mapping = {
        # Embedding: BaseTrainer,
        # MultipleEmbedding: BaseTrainer,
        # RobeEmbedding: BaseTrainer,
        # HashEmbedding: BaseTrainer,
        # MultipleHashEmbedding: BaseTrainer,
        # CompositionalEmbedding: BaseTrainer,
        # LearningEmbedding: BaseTrainer,
        # DPQEmbedding: BaseTrainer,
        # MDEmbedding: BaseTrainer,
        # QuantizedEmbedding: BaseTrainer,
        DeepLightEmbedding: DeepLightTrainer,
        # AutoDimEmbedding: AutoDimTrainer,
    }

    return trainer_mapping.get(type(layer), BaseTrainer)
