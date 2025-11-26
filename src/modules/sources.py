import torch.nn as nn
from ..config import ExpConfig
from ..utils.data_structs import TensorBatch
from .context_mappers import BaseContextMapper
from .samplers import BaseSampler

class SourcePipeline(nn.Module):
    """
    Universal Source Generator.
    Logic:
    1. Get Assortment S from Sampler (Synthetic or Real).
    2. Map S -> Context X using ContextMapper.
    3. Pack into TensorBatch.
    """
    def __init__(self, 
                 cfg: ExpConfig, 
                 sampler: BaseSampler,         
                 context_mapper: BaseContextMapper): 
        super().__init__()
        self.cfg = cfg
        self.sampler = sampler
        self.context_mapper = context_mapper

    def forward(self) -> TensorBatch:
        # 1. Get Raw Assortment (Items + Mask)
        # Whether synthetic or real, this interface is uniform.
        items, mask = self.sampler.sample()
        
        # 2. Derive Global Context X
        # This logic is shared across synthetic and real data.
        context = self.context_mapper(items, mask)
        
        return TensorBatch(context=context, items=items, mask=mask)