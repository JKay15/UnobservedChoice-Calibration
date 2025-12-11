import torch
import numpy as np
import torch.nn.functional as F

def compute_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Root Mean Squared Error"""
    mse = torch.mean((y_pred - y_true) ** 2)
    return torch.sqrt(mse).item()

def compute_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return torch.mean(torch.abs(y_pred - y_true)).item()

def compute_parameter_error(gamma_hat: torch.Tensor, gamma_star: torch.Tensor) -> float:
    """L2 Distance between estimated and true parameters"""
    # Ensure they are on the same device and shape
    diff = gamma_hat.flatten() - gamma_star.flatten()
    return torch.norm(diff, p=2).item()

def compute_p0_from_logits(z: torch.Tensor, s_hat: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Helper to reconstruct p0 given the estimated parameters.
    p0 = sigmoid( gamma^T z - s_hat )
    """
    # z: (N, d), gamma: (d, ) -> u0: (N, )
    u0 = z @ gamma
    eta = u0 - s_hat
    return torch.sigmoid(eta)

def compute_nll(y_prob: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-7) -> float:
    """
    Computes Negative Log-Likelihood (Binary Cross Entropy) broadly.
    Args:
        y_prob: Predicted probability of class 1 (p0). shape (N,)
        y_true: Binary labels (0 or 1). shape (N,)
    """
    # 1. Ensure inputs are Tensors
    if not isinstance(y_prob, torch.Tensor):
        y_prob = torch.tensor(y_prob)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
        
    # 2. Alignment & Type
    y_prob = y_prob.float().flatten()
    y_true = y_true.float().flatten()
    
    if y_prob.device != y_true.device:
        y_true = y_true.to(y_prob.device)
        
    # 3. Clip for numerical stability (Standard practice for NLL)
    y_prob = torch.clamp(y_prob, min=epsilon, max=1.0-epsilon)
    
    # 4. Compute BCE
    loss = F.binary_cross_entropy(y_prob, y_true)
    return loss.item()

def compute_nll_from_gamma(
    gamma: torch.Tensor, 
    z: torch.Tensor, 
    s: torch.Tensor, 
    y_true: torch.Tensor
) -> float:
    """
    Wrapper: Calculates probability first, then calls generic NLL.
    Useful for optimization loops or synthetic evaluation where p0 isn't materialized yet.
    """
    p0 = compute_p0_from_logits(z, s, gamma)
    return compute_nll(p0, y_true)

def compute_empirical_error_bound(
    y_prob: torch.Tensor, 
    y_true: torch.Tensor, 
    n_seeds: int,
    delta: float = 0.3,
    x_axis: str='n_samples'
) -> float:
    
    
        
    return 