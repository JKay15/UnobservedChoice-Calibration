import torch
from typing import Tuple

def solve_optimal_assortment(
    gamma: torch.Tensor,
    z: torch.Tensor,
    item_revenues: torch.Tensor,
    item_utilities: torch.Tensor,
    yes:bool=False
) -> Tuple[torch.Tensor, float]:
    """
    Solves unconstrained Assortment Optimization:
    max_S sum(r_i * v_i) / (v0 + sum v_i)
    """
    # 1. Calculate Outside Option 的 "Score" v0
    # gamma: (d,), z: (d,)
    u0 = torch.dot(z, gamma)
    v0 = torch.exp(u0)
    
    # 2. calculate Inside Items 的 "Score" v_i
    # item_utilities 是 log scale 的 u_i
    v_items = torch.exp(item_utilities)
    
    # 3. (Revenue-Ordered Set)
    sorted_revs, sorted_indices = torch.sort(item_revenues, descending=True)
    sorted_vs = v_items[sorted_indices]
    
    # 4. calculate top k revenue
    # Numerator: cumsum(r_i * v_i)
    # Denominator: v0 + cumsum(v_i)
    num = torch.cumsum(sorted_revs * sorted_vs, dim=0)
    den = v0 + torch.cumsum(sorted_vs, dim=0)
    
    expected_revenues = num / den
    
    # 5.find the best k
    best_k_idx = torch.argmax(expected_revenues)
    max_rev = expected_revenues[best_k_idx].item()
    
    # 6. Mask
    best_k = best_k_idx + 1
    selected_indices = sorted_indices[:best_k]
    
    best_mask = torch.zeros_like(item_revenues, dtype=torch.bool)
    best_mask[selected_indices] = True
    if yes:
        print(best_mask)
    
    return best_mask, max_rev

def calculate_revenue(
    mask: torch.Tensor,
    gamma_true: torch.Tensor,
    z: torch.Tensor,
    item_revenues: torch.Tensor,
    item_utilities_true: torch.Tensor
) -> float:
    """
    Evaluate the TRUE revenue of a chosen assortment (mask).
    R(S) = (sum_{i in S} r_i v_i^*) / (v0^* + sum_{i in S} v_i^*)
    """
    if not mask.any():
        return 0.0
        
    r_s = item_revenues[mask]
    u_s_true = item_utilities_true[mask]
    v_s_true = torch.exp(u_s_true)
    
    # True Outside Option
    v0_true = torch.exp(torch.dot(z, gamma_true))
    
    # Revenue Formula
    numerator = torch.sum(r_s * v_s_true)
    denominator = v0_true + torch.sum(v_s_true)
    
    return (numerator / denominator).item()