"""Loss functions."""

import torch
from torch import Tensor


def binary_loss(logits: Tensor, labels: Tensor, positive_weight: float | None = None) -> Tensor:
    weight = torch.tensor(positive_weight, device=logits.device) if positive_weight else None
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
