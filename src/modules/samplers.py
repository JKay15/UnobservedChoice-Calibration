import torch
import torch.nn as nn
import numpy as np
from ..config import ExpConfig

class BaseSampler(nn.Module):
    """
    Interface for obtaining a batch of assortments (S).
    Responsibility:
    1. Manage the Item Universe (Features).
    2. Return a batch of Assortments (Items + Mask).
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            items: (B, Max_Len, D_item) - Padded
            mask: (B, Max_Len)
        """
        raise NotImplementedError

class SyntheticSampler(BaseSampler):
    """
    Sampler for Synthetic Data.
    1. Generates a random Item Universe on init.
    2. Randomly samples subsets (assortments) on each call.
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__(cfg)
        
        # --- 1. Construct Item Universe (Fixed for the experiment) ---
        D = cfg.dim_item_feat
        rho = getattr(cfg, 'item_feat_corr', 0.0)
        
        # Build Covariance
        cov_matrix = torch.full((D, D), rho)
        cov_matrix.fill_diagonal_(1.0)
        mean_vector = torch.zeros(D)
        
        # Sample Universe
        dist = torch.distributions.MultivariateNormal(mean_vector, covariance_matrix=cov_matrix)
        universe = dist.sample((cfg.n_items_pool,))
        
        self.register_buffer("item_universe", universe)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.cfg.n_samples
        device = self.item_universe.device
        
        # 1. Randomly determine sizes
        sizes = np.random.randint(
            self.cfg.min_assortment_size, 
            self.cfg.max_assortment_size + 1, 
            size=N
        )
        max_len = np.max(sizes)
        
        # 2. Prepare containers
        batch_items = torch.zeros((N, max_len, self.cfg.dim_item_feat), device=device)
        batch_mask = torch.zeros((N, max_len), device=device)
        
        # 3. Sampling Loop (Select indices -> Lookup features)
        pool_indices = np.arange(self.cfg.n_items_pool)
        
        for i in range(N):
            k = sizes[i]
            chosen_idx_np = np.random.choice(pool_indices, size=k, replace=False)
            chosen_idx = torch.tensor(chosen_idx_np, device=device)
            
            # Lookup from Universe
            batch_items[i, :k, :] = self.item_universe[chosen_idx]
            batch_mask[i, :k] = 1.0
            
        return batch_items, batch_mask