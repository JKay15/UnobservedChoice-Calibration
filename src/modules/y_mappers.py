import torch
import torch.nn as nn
from ..config import ExpConfig

class BaseYMapper(nn.Module):
    """
    Base class for the Simulator Link Function h(eta).
    Responsible for:
    1. Mapping True Logit (eta) -> Observed Logit (y) via a link function.
    2. Injecting Simulator Noise (epsilon).
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, eta_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eta_true: The ground truth logit (u0 - s).
        Returns:
            y_obs: The simulator output with bias and noise.
        """
        # 1. Deterministic Link Function h(.)
        y_clean = self._link_function(eta_true)
        
        # 2. Noise Injection
        # epsilon ~ N(0, sigma^2)
        if self.cfg.sim_noise_sigma > 0:
            noise = torch.randn_like(y_clean) * self.cfg.sim_noise_sigma
            return y_clean + noise
        
        return y_clean

    def _link_function(self, eta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class LinearYMapper(BaseYMapper):
    """
    Scenario A: Linear Bias.
    y = a + b * eta
    
    Target: To verify Algorithm 1 (Linear Calibration).
    """
    def _link_function(self, eta: torch.Tensor) -> torch.Tensor:
        # a (intercept) and b (slope) are controlled by config
        return self.cfg.sim_bias_a + self.cfg.sim_bias_b * eta

class MonotoneYMapper(BaseYMapper):
    """
    Scenario B: Non-linear Monotone Distortion.
    """
    def _link_function(self, eta: torch.Tensor) -> torch.Tensor:
        # [FIX] Use config to control scale instead of hardcoded 10.0
        # sim_bias_b controls the Amplitude/Steepness of the distortion
        # Larger b = More saturation = Harder for Linear algo
        scale = self.cfg.sim_bias_b 
        shift = self.cfg.sim_bias_a
        
        # y = Scale * (Sigmoid(eta) - 0.5) + Shift
        return scale * (torch.sigmoid(eta) - 0.5) + shift