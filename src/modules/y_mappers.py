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
        y_clean = self._link_function(eta_true)
        
        sigma = self.cfg.sim_noise_sigma
        noise_type = getattr(self.cfg, 'sim_noise_type', 'gaussian')
        
        if sigma > 0:
            if noise_type == 'gaussian':
                noise = torch.randn_like(y_clean) * sigma
            elif noise_type == 'cauchy':
                # [CRITICAL] Cauchy Noise: The Linear Killer
                # Generate from Cauchy(0, sigma)
                # Cauchy = StandardNormal / StandardNormal
                dist = torch.distributions.Cauchy(loc=0.0, scale=sigma)
                noise = dist.sample(y_clean.shape).to(y_clean.device)
                
                # Optional: Clip extreme artifacts if numerical stability is an issue
                # But we want to hurt OLS, so let them be large.
                noise.clamp_(-1000, 1000) 
            
            y_clean += noise
            
        # Outlier injection (Binary injection)
        prob = getattr(self.cfg, 'outlier_prob', 0.0)
        scale = getattr(self.cfg, 'outlier_scale', 50.0)
        if prob > 0:
            mask = torch.rand_like(y_clean) < prob
            outliers = torch.randn_like(y_clean) * scale
            y_clean[mask] = outliers[mask]
            
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
        # sim_bias_b controls the Amplitude/Steepness of the distortion
        # Larger b = More saturation = Harder for Linear algo
        scale = self.cfg.sim_bias_b 
        shift = self.cfg.sim_bias_a
        
        # y = Scale * (Sigmoid(eta) - 0.5) + Shift
        # return scale * (torch.sigmoid(eta) - 0.5) + shift
        beta  = 20.0  # 自己加的超参数，例如 1.0~10

        y = (1.0/beta) * torch.log1p(torch.exp(beta * eta))
        return scale * (y - (1.0/beta)*torch.log1p(torch.exp(torch.tensor(0.0)))) + shift
# class MonotoneYMapper(BaseYMapper):
#     """
#     Scenario B: Step Function (The "Linear Killer").
#     Linear regression fails to fit steps well, but rank correlation is perfect.
#     """
#     def _link_function(self, eta: torch.Tensor) -> torch.Tensor:
#         scale = self.cfg.sim_bias_b 
#         shift = self.cfg.sim_bias_a
        
#         # [NEW] Step Function (Soft steps to keep gradient flow for Simulator generation?)
#         # No, for generation we can be hard.
#         # y = floor(eta)
#         # This destroys local gradient information for Linear, but preserves global rank.
        
#         # Using a steep sigmoid sum to simulate steps (differentiable-ish) or just raw steps
#         # Let's use raw steps. 
#         # eta range is approx [-3, 3].
#         # We create steps at integers.
        
#         y_step = torch.floor(eta)
        
#         return shift + scale * y_step