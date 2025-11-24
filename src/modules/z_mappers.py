import torch
import torch.nn as nn
from ..utils.data_structs import TensorBatch
from ..config import ExpConfig

class BaseZMapper(nn.Module):
    """
    Base class for mapping raw input (TensorBatch) to feature vector z.
    Output shape: (Batch_Size, Dim_Z)
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        raise NotImplementedError
    
class StatsZMapper(BaseZMapper):
    """
    Baseline Strategy: Statistical Features + Linear Projection.
    
    Features calculated:
    1. Mean: Average level of items.
    2. Max: The best/highest feature (e.g., Max Quality).
    3. Min: The lowest threshold (e.g., Min Price).
    4. Std: Diversity/Dispersion of the assortment.
    
    Logic:
    Concat[Context, Mean, Max, Min, Std] -> Project(dim_z)
    """
    def __init__(self, cfg: ExpConfig):
        super().__init__(cfg)
        
        # LazyLinear allows us to handle any number of input statistics 
        # without manually calculating input dimensions.
        self.projector = nn.LazyLinear(cfg.dim_z)
        
        # Freeze the projector to make it a deterministic Random Projection
        for param in self.projector.parameters():
            param.requires_grad = False

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        # items: (B, L, D)
        # mask: (B, L)
        mask_expanded = batch.mask.unsqueeze(-1) # (B, L, 1)
        
        # Pre-calculate counts to avoid division by zero
        # (B, 1)
        counts = torch.sum(batch.mask, dim=1, keepdim=True).clamp(min=1.0)

        # --- 1. Mean ---
        sum_items = torch.sum(batch.items * mask_expanded, dim=1)
        mean_items = sum_items / counts

        # --- 2. Max ---
        # Fill padded values with -inf so they don't affect Max
        items_for_max = batch.items.clone()
        items_for_max[batch.mask == 0] = -1e9
        max_items = torch.max(items_for_max, dim=1)[0] # (B, D)

        # --- 3. Min ---
        # Fill padded values with +inf so they don't affect Min
        items_for_min = batch.items.clone()
        items_for_min[batch.mask == 0] = 1e9
        min_items = torch.min(items_for_min, dim=1)[0] # (B, D)

        # --- 4. Standard Deviation (Std) ---
        # Formula: sqrt( sum((x - mean)^2) / count )
        # Expand mean to (B, 1, D) for broadcasting
        mean_expanded = mean_items.unsqueeze(1) 
        
        # Calculate squared differences
        diff_sq = (batch.items - mean_expanded) ** 2
        
        # Mask out the differences calculated on padded zeros
        diff_sq_masked = diff_sq * mask_expanded
        
        # Variance
        var_items = torch.sum(diff_sq_masked, dim=1) / counts
        
        # Std (add epsilon for numerical stability)
        std_items = torch.sqrt(var_items + 1e-8)

        # --- 5. Concatenate All Info ---
        # Input vector = [Context, Mean, Max, Min, Std]
        raw_features = torch.cat([
            batch.context, 
            mean_items, 
            max_items, 
            min_items, 
            std_items
        ], dim=1)
        
        # --- 6. Project to Fixed Dimension ---
        z = self.projector(raw_features)
        
        return z
    
# class NeuralZMapper(BaseZMapper):
#     """
#     Advanced Strategy: Neural Network based mapping.
    
#     This class acts as a WRAPPER/ADAPTER. 
#     It does not define the network structure itself.
#     Instead, it accepts an injected `backbone` (nn.Module).
    
#     This allows using:
#     1. Random MLPs (for synthetic ablation)
#     2. Pre-trained Deep Sets/Transformers (for real data)
#     """
#     def __init__(self, cfg: ExpConfig, backbone: nn.Module = None):
#         super().__init__(cfg)
        
#         if backbone is None:
#             # Optional: Provide a sensible default for simple synthetic tests
#             # ONLY if the user didn't provide one.
#             # We create a simple Permutation Invariant Network (Deep Sets style)
#             input_dim = cfg.dim_item_feat
#             # Note: We rely on cfg.dim_z here because we must output the correct dimension
#             self.backbone = self._build_default_backbone(input_dim, cfg.dim_z)
#         else:
#             # Use the injected pre-trained model
#             self.backbone = backbone

#     def _build_default_backbone(self, input_dim: int, output_dim: int) -> nn.Module:
#         """
#         Helper to build a simple Deep Sets network for synthetic experiments.
#         Phi(x) -> Sum -> Rho(x)
#         """
#         class SimpleDeepSets(nn.Module):
#             def __init__(self, in_d, out_d):
#                 super().__init__()
#                 self.phi = nn.Sequential(nn.Linear(in_d, 32), nn.ReLU())
#                 self.rho = nn.Sequential(nn.Linear(32, out_d))
            
#             def forward(self, items, mask, context):
#                 # items: (B, L, D)
#                 h = self.phi(items)
#                 # Masking
#                 h = h * mask.unsqueeze(-1)
#                 # Sum Pooling
#                 h_sum = torch.sum(h, dim=1)
#                 # If context exists, concat it (Simple fusion)
#                 if context.numel() > 0:
#                     # Assuming context mapping needs to be handled by rho, 
#                     # but for this default, we ignore context fusion complexity.
#                     pass 
#                 return self.rho(h_sum)
                
#         return SimpleDeepSets(input_dim, output_dim)

#     def forward(self, batch: TensorBatch) -> torch.Tensor:
#         """
#         Passes the raw batch data to the backbone network.
#         The backbone is expected to handle padding/masking internally.
#         """
#         # We pass individual tensors to be flexible with different model signatures
#         z = self.backbone(batch.items, batch.mask, batch.context)
#         return z

class NeuralZMapper(BaseZMapper):
    """
    Neural Network based mapping.
    For synthetic experiments, we initialize it randomly and freeze it.
    This simulates a "Pre-trained Feature Extractor" of dimension dim_z.
    """
    def __init__(self, cfg: ExpConfig, model_path: str = None):
        super().__init__(cfg)
        
        input_dim = cfg.dim_item_feat
        output_dim = cfg.dim_z  # <--- 关键：这里接收 Config 里的动态维度
        
        # 1. Build the Network (Your logic)
        self.backbone = self._build_default_backbone(input_dim, output_dim)
            
        # 2. (Optional) Load weights if provided (for specific reproducibility)
        if model_path:
            self.load_weights(model_path)
            
        # 3. Freeze parameters! 
        # In synthetic methodology, a frozen random net = a generic pre-trained net.
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _build_default_backbone(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Your SimpleDeepSets implementation.
        """
        class SimpleDeepSets(nn.Module):
            def __init__(self, in_d, out_d):
                super().__init__()
                # A slightly deeper Phi to make features more "non-linear"
                self.phi = nn.Sequential(
                    nn.Linear(in_d, 32), 
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU()
                )
                # Rho maps to the target dimension d
                self.rho = nn.Sequential(
                    nn.Linear(32, out_d) 
                    # No activation at the end to allow full range for Z
                )
            
            def forward(self, items, mask, context):
                # items: (B, L, D)
                h = self.phi(items)
                # Masking
                h = h * mask.unsqueeze(-1)
                # Sum Pooling
                h_sum = torch.sum(h, dim=1)
                # Context fusion (if needed, simple concat logic can go here)
                return self.rho(h_sum)
                
        return SimpleDeepSets(input_dim, output_dim)

    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location=self.cfg.device)
        self.backbone.load_state_dict(state_dict)

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        z = self.backbone(batch.items, batch.mask, batch.context)
        return z