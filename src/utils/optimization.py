import torch
import torch.nn as nn
from typing import Tuple

def solve_optimal_assortment(
    gamma: torch.Tensor,
    z: torch.Tensor,
    item_revenues: torch.Tensor,
    item_utilities: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """
    Solves the unconstrained Assortment Optimization problem under MNL.
    
    Problem:
        max_{S} R(S) = (sum_{i in S} r_i * v_i) / (v_0 + sum_{i in S} v_i)
        where v_i = exp(u_i), v_0 = exp(u_0) = exp(gamma^T z)
    
    Algorithm:
        Static MNL Optimization. The optimal set is always a "Revenue-Ordered" set.
        We sort items by revenue r_i descending, and check all prefixes.
    
    Args:
        gamma: (dim_z,) Estimated or True parameter for outside option.
        z: (dim_z,) Context feature for the current decision instance.
        item_revenues: (n_items,) Prices/Revenues of candidate items.
        item_utilities: (n_items,) Inside utilities u_i (before exp) of candidate items.
        
    Returns:
        best_mask: (n_items,) Binary mask of the optimal assortment.
        max_revenue: Expected revenue of the optimal assortment.
    """
    # 1. Calculate Terms
    # v0 = exp(gamma^T z)
    u0 = torch.dot(z, gamma)
    v0 = torch.exp(u0)
    
    # v_i = exp(u_i)
    v_items = torch.exp(item_utilities)
    
    # 2. Sort items by Revenue (Descending)
    # This is the key property of MNL assortment optimization
    sorted_revs, sorted_indices = torch.sort(item_revenues, descending=True)
    sorted_vs = v_items[sorted_indices]
    
    # 3. Iterate through all candidate Revenue-Ordered sets
    # Set k: includes top-k items
    # R_k = (sum_{1..k} r_i v_i) / (v0 + sum_{1..k} v_i)
    
    # Cumulative Sums
    numerator_cumsum = torch.cumsum(sorted_revs * sorted_vs, dim=0)
    denominator_v_cumsum = torch.cumsum(sorted_vs, dim=0)
    
    # Expected Revenue for each prefix size k=1...N
    # Shape: (n_items,)
    revenues = numerator_cumsum / (v0 + denominator_v_cumsum)
    
    # 4. Find Best k
    best_idx = torch.argmax(revenues)
    max_revenue = revenues[best_idx].item()
    
    # 5. Construct the optimal set mask
    # We select top-(best_idx+1) items from the sorted list
    best_k = best_idx + 1
    selected_indices = sorted_indices[:best_k]
    
    best_mask = torch.zeros_like(item_revenues, dtype=torch.bool)
    best_mask[selected_indices] = True
    
    return best_mask, max_revenue

def calculate_revenue(
    mask: torch.Tensor,
    gamma_true: torch.Tensor,
    z: torch.Tensor,
    item_revenues: torch.Tensor,
    item_utilities_true: torch.Tensor
) -> float:
    """
    Evaluates the TRUE Expected Revenue of a given assortment (mask).
    Used to compute Regret: R(S_opt) - R(S_hat).
    """
    if not mask.any():
        return 0.0
        
    # Select chosen items
    r_s = item_revenues[mask]
    u_s = item_utilities_true[mask]
    v_s = torch.exp(u_s)
    
    # Outside option
    v0 = torch.exp(torch.dot(z, gamma_true))
    
    # Revenue Formula
    revenue = torch.sum(r_s * v_s) / (v0 + torch.sum(v_s))
    return revenue.item()