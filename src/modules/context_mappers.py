import torch
import torch.nn as nn
from ..utils.data_structs import TensorBatch
from ..config import ExpConfig


class BaseContextMapper(nn.Module):
    """
    Mapping: Assortment Items (S) -> Global Context (X)
    """

    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            items: (B, Max_Len, D_item)
            mask: (B, Max_Len)
        Returns:
            context: (B, D_context)
        """
        raise NotImplementedError


class AvgContextMapper(BaseContextMapper):
    """
    Strategy A: X is the average feature vector of the assortment.
    Example: "Market Context" is defined by the average quality of items.
    """

    def forward(self, items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)
        sum_items = torch.sum(items * mask_expanded, dim=1)
        count = torch.sum(mask, dim=1, keepdim=True).clamp(min=1.0)
        return sum_items / count


class ConcatContextMapper(BaseContextMapper):
    """
    Strategy B: X is the concatenation of all item features (Flatten).
    Note: This requires fixed assortment size or careful padding handling.
    For variable size, usually used with RNNs/Transformers later.
    """

    def forward(self, items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Flatten: (B, Max_Len * D_item)
        # Padded zeros remain zeros, acting as "empty slots"
        return items.view(items.size(0), -1)
