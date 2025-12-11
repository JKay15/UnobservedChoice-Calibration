import torch
import torch.nn as nn
from ..config import ExpConfig
from ..utils.data_structs import TensorBatch
from .context_mappers import BaseContextMapper
from .samplers import BaseSampler

class SourcePipeline(nn.Module):
    """
    Universal Source Generator.
    [Updated] Logic:
    1. Get Data from Sampler.
    2. IF Sampler provides independent Context X (Assumption 2 Fix), use it.
    3. ELSE Map S -> Context X using ContextMapper (Legacy/Real Data Mode).
    4. Pack into TensorBatch.
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
        # 1. Get Data from Sampler
        sample_out = self.sampler.sample()
        
        # Handle dict vs tuple return types for compatibility
        if isinstance(sample_out, dict):
            items = sample_out['items']
            mask = sample_out['mask']
            # Try to get pre-generated context (from SyntheticSampler)
            raw_context = sample_out.get('context', None)
        else:
            # Fallback for legacy samplers returning tuple
            items, mask = sample_out
            raw_context = None
        
        # 2. Determine Context Source
        # Priority: Independent Context (from Sampler) > Derived Context (from Mapper)
        # This branch is CRITICAL for Assumption 2 (Non-degeneracy)
        if raw_context is not None and raw_context.numel() > 0:
            context = raw_context
        else:
            # Fallback: Derive context from items (e.g. mean pooling)
            # Used when dim_context=0 or in Real Data experiments
            context = self.context_mapper(items, mask)
        
        return TensorBatch(context=context, items=items, mask=mask)