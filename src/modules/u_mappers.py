import torch
import torch.nn as nn
from typing import Optional
from ..utils.data_structs import TensorBatch
from ..config import ExpConfig

class BaseUtilityMapper(nn.Module):
    """
    Base class for calculating Inside Utilities u_{ik}.
    
    Dual Roles:
    1. Synthetic Mode: Generates Ground Truth u* (Randomly initialized & Fixed).
    2. Real Data Mode: Calculates Estimated u_hat (Loads pre-trained weights).
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        """
        Compute utilities for the assortment.
        Returns:
            u: (Batch, Max_Len)
        CRITICAL: Padded positions (where mask=0) MUST be set to -inf 
                  to effectively remove them from Softmax/LogSumExp.
        """
        raise NotImplementedError
    
    def load_weights(self, path: str):
        """
        Interface to load pre-trained parameters.
        Used in Real Data experiments to load the estimated beta/net.
        """
        print(f"[{self.__class__.__name__}] Loading weights from {path} ...")
        # Load to CPU first to avoid device conflicts, then move to current device
        state_dict = torch.load(path, map_location='cpu')
        
        # Handle cases where the file might be a raw Tensor (for Linear) or a State Dict
        if isinstance(state_dict, torch.Tensor):
            # Legacy support if you saved just the beta tensor
            self._load_tensor(state_dict)
        else:
            # Standard state_dict loading
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                print(f"Warning: Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"Warning: Unexpected keys: {unexpected}")

    def _load_tensor(self, tensor: torch.Tensor):
        raise NotImplementedError

class LinearUtilityMapper(BaseUtilityMapper):
    """
    Linear Utility Model: u = x @ beta
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__(cfg)
        
        # Define the parameter beta
        # In Synthetic Mode: This is initialized randomly -> acts as Truth beta*
        # In Real Mode: This acts as a placeholder -> overwritten by load_weights()
        self.beta = nn.Parameter(torch.randn(cfg.dim_item_feat))
        
        # Freeze it! We are not training it here. We are just using it.
        self.beta.requires_grad = False

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        # batch.items: (B, L, D)
        # self.beta:   (D,)
        
        # 1. Linear Projection
        # u_raw: (B, L)
        u_raw = torch.einsum("bld,d->bl", batch.items, self.beta)
        
        # 2. Masking
        # (1.0 - mask) * -1e9
        # Padded items become -inf
        penalty = (1.0 - batch.mask) * -1e9
        
        return u_raw + penalty

    def _load_tensor(self, tensor: torch.Tensor):
        # Verify shape
        if tensor.shape != self.beta.shape:
            raise ValueError(f"Shape mismatch: Config {self.beta.shape} vs Loaded {tensor.shape}")
        with torch.no_grad():
            self.beta.copy_(tensor)

class NeuralUtilityMapper(BaseUtilityMapper):
    """
    Non-linear Utility Model: u = MLP(x)
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__(cfg)
        
        input_dim = cfg.dim_item_feat
        # You might want to add 'hidden_dim' to ExpConfig if you want to tune this
        hidden_dim = 64 
        
        # Define the network structure
        # IMPORTANT: This structure must match exactly what you used in your Estimator script
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output scalar utility
        )
        
        # Freeze parameters
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        # 1. Forward Pass
        # input: (B, L, D) -> output: (B, L, 1)
        u_raw = self.net(batch.items).squeeze(-1)
        
        # 2. Masking
        penalty = (1.0 - batch.mask) * -1e9
        
        return u_raw + penalty
    
    def _load_tensor(self, tensor: torch.Tensor):
        raise NotImplementedError("NeuralMapper requires a full state_dict, not a single tensor.")