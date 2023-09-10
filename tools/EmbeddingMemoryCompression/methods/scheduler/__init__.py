# only for DLRMs now

from .base import EmbeddingTrainer
from .hash import HashEmbTrainer
from .compo import CompoEmbTrainer
from .tensortrain import TTEmbTrainer
from .dhe import DHETrainer
from .robe import ROBETrainer
from .dpq import DPQTrainer
from .mgqe import MGQETrainer
from .adapt import AdaptEmbTrainer
from .deduplication import DedupOverallTrainer
from .md import MDETrainer
from .autodim import AutoDimOverallTrainer
from .optembed import OptEmbedOverallTrainer
from .deeplight import DeepLightOverallTrainer
from .pep import PEPEmbTrainer
from .autosrh import AutoSrhOverallTrainer
from .quantize import QuantizeEmbTrainer
from .alpt import ALPTEmbTrainer


_layer2trainer_mapping = {
    'full': EmbeddingTrainer,
    'hash': HashEmbTrainer,
    'compo': CompoEmbTrainer,
    'tt': TTEmbTrainer,
    'dhe': DHETrainer,
    'robe': ROBETrainer,
    'dpq': DPQTrainer,
    'mgqe': MGQETrainer,
    'adapt': AdaptEmbTrainer,
    'dedup': DedupOverallTrainer,
    'md': MDETrainer,
    'autodim': AutoDimOverallTrainer,
    'optembed': OptEmbedOverallTrainer,
    'deeplight': DeepLightOverallTrainer,
    'pep': PEPEmbTrainer,
    'autosrh': AutoSrhOverallTrainer,
    'quantize': QuantizeEmbTrainer,
    'alpt': ALPTEmbTrainer,
}


def get_trainer(layer_type):
    trainer = _layer2trainer_mapping[layer_type]
    return trainer
