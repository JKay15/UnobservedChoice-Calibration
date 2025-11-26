import torch
from typing import Dict, Any

from ..config import ExpConfig
from ..utils.data_structs import TensorBatch

# 引入接口用于类型提示
from ..modules.sources import SourcePipeline
from ..modules.z_mappers import BaseZMapper
from ..modules.u_mappers import BaseUtilityMapper
from ..modules.y_mappers import BaseYMapper

class SyntheticDataEngine:
    """
    The Orchestrator (流水线指挥官).
    
    Flow: 
    1. SourcePipeline -> Raw Batch (X, S, Mask)
    2. Mappers -> Features (z) & Utilities (u)
    3. Logic -> Latent Truth (s, eta, p0)
    4. Simulator -> Observed Logit (y)
    5. Noise Injection -> Estimated Inclusive Value (s_hat)
    """
    def __init__(self, 
                 cfg: ExpConfig,
                 source_pipeline: SourcePipeline, 
                 z_mapper: BaseZMapper,           
                 u_mapper: BaseUtilityMapper,     
                 y_mapper: BaseYMapper):          
        
        self.cfg = cfg
        self.source = source_pipeline
        self.z_mapper = z_mapper
        self.u_mapper = u_mapper
        self.y_mapper = y_mapper
        
        # --- Latent Truth (God's Parameters) ---
        # Gamma*: The true coefficients for the Outside Option
        # Generated once and fixed for the experiment instance.
        self.gamma_star = torch.randn(cfg.dim_z, device=cfg.device)

    def generate(self) -> Dict[str, Any]:
        """
        Execute one run of data generation.
        """
        # 1. Source Generation (S -> X)
        # Returns TensorBatch with context, items, mask
        batch: TensorBatch = self.source()
        
        # 2. Feature Mapping (X -> z)
        # (N, dim_z)
        z_val = self.z_mapper(batch)
        
        # 3. Utility Mapping (X -> u)
        # (N, max_len) - Padded positions are -inf
        u_val = self.u_mapper(batch)
        
        # 4. Calculate Latent Truth
        # Inclusive Value s*
        s_true = torch.logsumexp(u_val, dim=1)
        
        # Outside Utility u0*
        u0_true = z_val @ self.gamma_star
        
        # True Logit eta*
        eta_true = u0_true - s_true
        
        # True Probability p0* (for validation)
        p0_true = torch.sigmoid(eta_true)
        
        # 5. Simulator Process (eta* -> y)
        # Applies Link Function and Simulator Noise
        y_obs = self.y_mapper(eta_true)
        
        # 6. Estimation Error Injection (s* -> s_hat)
        # Mode A: Noise Injection
        if self.cfg.utility_mode == 'noise_injection':
            noise = torch.randn_like(s_true) * self.cfg.est_noise_sigma
            s_hat = s_true + noise
        else:
            # Mode B: MLE Estimation
            s_hat = s_true 

        # 7. Return Package
        return {
            # Input for Algorithm
            'inputs': {
                'z': z_val,
                's_hat': s_hat,
                'y': y_obs,
                'batch': batch 
            },
            
            # Ground Truth
            'truth': {
                'p0': p0_true,
                'gamma': self.gamma_star,
                's': s_true,
                'eta': eta_true,
                'u0': u0_true
            }
        }