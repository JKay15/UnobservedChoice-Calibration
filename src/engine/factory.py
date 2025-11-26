from ..config import ExpConfig

# Modules
from ..modules.samplers import SyntheticSampler
from ..modules.context_mappers import AvgContextMapper, ConcatContextMapper
from ..modules.sources import SourcePipeline
from ..modules.z_mappers import StatsZMapper, NeuralZMapper
from ..modules.u_mappers import LinearUtilityMapper, NeuralUtilityMapper
from ..modules.y_mappers import LinearYMapper, MonotoneYMapper

# Engine
from .synthetic import SyntheticDataEngine

class EngineFactory:
    """
    Factory class to assemble the Data Engine based on strategy names.
    """
    
    @staticmethod
    def build_synthetic_engine(cfg: ExpConfig, 
                               z_type: str = 'stats',
                               u_type: str = 'linear',
                               y_type: str = 'linear',
                               z_model_path: str = None) -> SyntheticDataEngine:
        
        # 1. Build Source Pipeline
        sampler = SyntheticSampler(cfg)
        context_mapper = AvgContextMapper(cfg)
        source_pipeline = SourcePipeline(cfg, sampler, context_mapper)
        
        # 2. Build Z Mapper
        if z_type == 'stats':
            z_mapper = StatsZMapper(cfg)
        elif z_type == 'neural':
            z_mapper = NeuralZMapper(cfg, model_path=z_model_path)
        else:
            raise ValueError(f"Unknown z_type: {z_type}")
            
        # 3. Build U Mapper
        if u_type == 'linear':
            u_mapper = LinearUtilityMapper(cfg)
        elif u_type == 'neural':
            u_mapper = NeuralUtilityMapper(cfg)
        else:
            raise ValueError(f"Unknown u_type: {u_type}")
            
        # 4. Build Y Mapper (Simulator)
        if y_type == 'linear':
            y_mapper = LinearYMapper(cfg)
        elif y_type == 'monotone':
            y_mapper = MonotoneYMapper(cfg)
        else:
            raise ValueError(f"Unknown y_type: {y_type}")
            
        device = cfg.device
        source_pipeline.to(device)
        z_mapper.to(device)
        u_mapper.to(device)
        y_mapper.to(device)

        # 5. Assemble Engine
        engine = SyntheticDataEngine(
            cfg=cfg,
            source_pipeline=source_pipeline,
            z_mapper=z_mapper,
            u_mapper=u_mapper,
            y_mapper=y_mapper
        )
        
        return engine