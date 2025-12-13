import torch
from dataclasses import dataclass


@dataclass
class TensorBatch:
    """
    A data container for variable-sized assortments, compatible with PyTorch.

    This structure handles the 'Padding & Masking' logic required when different
    samples have different assortment sizes |S_k|.
    """

    # Global context features X (e.g., user features).
    # Shape: (batch_size, dim_global_feat)
    # If dim_global_feat is 0, this might be an empty tensor or unused.
    context: torch.Tensor

    # Item features f_i for all items in the assortment.
    # Shape: (batch_size, max_assortment_size, dim_item_feat)
    # Note: Samples with fewer items than max_assortment_size are padded with zeros.
    items: torch.Tensor

    # Binary mask indicating valid items.
    # Shape: (batch_size, max_assortment_size)
    # Value: 1.0 for real items, 0.0 for padded items.
    mask: torch.Tensor

    def to(self, device: str) -> "TensorBatch":
        """Moves all tensors to the specified device (CPU/GPU/MPS)."""
        return TensorBatch(
            context=self.context.to(device),
            items=self.items.to(device),
            mask=self.mask.to(device),
        )

    @property
    def batch_size(self) -> int:
        """Returns the number of samples in the batch."""
        return self.items.shape[0]

    @property
    def max_size(self) -> int:
        """Returns the maximum assortment size in this batch."""
        return self.items.shape[1]
