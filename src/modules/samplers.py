import torch
import torch.nn as nn
import numpy as np
from ..config import ExpConfig

class BaseSampler(nn.Module):
    """
    Interface for obtaining a batch of assortments (S).
    Responsibility:
    1. Manage the Item Universe (Features).
    2. Return a batch of Assortments (Items + Mask) AND Context.
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def sample(self) -> dict: 
        """
        Returns a dictionary containing:
            - 'items': (B, Max_Len, D_item)
            - 'mask': (B, Max_Len)
            - 'context': (B, D_context) [Optional/New]
        """
        raise NotImplementedError

class SyntheticSampler(BaseSampler):
    """
    Sampler for Synthetic Data.
    1. Generates a random Item Universe on init.
    2. Randomly samples subsets (assortments) on each call.
    3. [NEW] Generates independent Context X to ensure Assumption 2 (Non-degeneracy).
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__(cfg)
        
        # --- 1. Construct Item Universe (Fixed for the experiment) ---
        D = cfg.dim_item_feat
        rho = getattr(cfg, 'item_feat_corr', 0.0)
        
        # Build Covariance Matrix for Items
        cov_matrix = torch.full((D, D), rho)
        cov_matrix.fill_diagonal_(1.0)
        mean_vector = torch.zeros(D)
        
        # Sample Universe
        dist = torch.distributions.MultivariateNormal(mean_vector, covariance_matrix=cov_matrix)
        universe = dist.sample((cfg.n_items_pool,))
        
        # Register as buffer to save/load with model state
        self.register_buffer("item_universe", universe)

    def sample(self) -> dict:
        N = self.cfg.n_samples
        device = self.item_universe.device
        
        # --- [NEW] Independent Context Generation ---
        # 目的：构造满足 Assumption 2 的设计矩阵。
        # 独立生成 X 避免了 z(X) 和 s(S) 之间的内生共线性。
        
        dim_ctx = getattr(self.cfg, 'dim_context', 0)
        
        if dim_ctx > 0:
            # 1. Generate Standard Normal
            batch_context = torch.randn(N, dim_ctx, device=device)
            
            # 2. [Assumption 8] Bounded Regressors
            # 虽然 Gaussian 是 unbounded，但在有限样本下极值概率很低。
            # 显式截断是满足定理假设的最严谨做法。常数 3.0 覆盖了 99.7% 的区间。
            batch_context.clamp_(-3.0, 3.0) 
        else:
            batch_context = torch.empty(N, 0, device=device)

        # --- Random Sizes ---
        sizes = np.random.randint(
            self.cfg.min_assortment_size, 
            self.cfg.max_assortment_size + 1, 
            size=N
        )
        max_len = np.max(sizes)
        
        # --- Prepare Containers ---
        batch_items = torch.zeros((N, max_len, self.cfg.dim_item_feat), device=device)
        batch_mask = torch.zeros((N, max_len), device=device)
        
        pool_indices = np.arange(self.cfg.n_items_pool)
        
        # --- Sampling Loop ---
        for i in range(N):
            k = sizes[i]
            chosen_idx_np = np.random.choice(pool_indices, size=k, replace=False)
            chosen_idx = torch.tensor(chosen_idx_np, device=device)
            
            batch_items[i, :k, :] = self.item_universe[chosen_idx]
            batch_mask[i, :k] = 1.0
            
        return {
            'items': batch_items,
            'mask': batch_mask,
            'context': batch_context
        }