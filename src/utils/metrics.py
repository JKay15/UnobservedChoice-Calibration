import torch
import numpy as np

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