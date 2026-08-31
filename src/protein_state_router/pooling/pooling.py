"""Masked single and pair representation pooling utilities."""

import torch
from torch import Tensor


def _validate_mask(mask: Tensor) -> None:
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not mask.any(dim=-1).all():
        raise ValueError("fully masked examples are not supported")


def masked_mean_std_max(values: Tensor, mask: Tensor) -> Tensor:
    """Pool values across all masked axes into mean, standard deviation, and maximum."""
    if values.shape[:-1] != mask.shape:
        raise ValueError("mask shape must match all non-feature value dimensions")
    flat_values = values.flatten(1, -2)
    flat_mask = mask.flatten(1)
    _validate_mask(flat_mask)
    expanded = flat_mask.unsqueeze(-1)
    count = expanded.sum(dim=1).clamp_min(1)
    mean = (flat_values * expanded).sum(dim=1) / count
    centered = (flat_values - mean.unsqueeze(1)) * expanded
    std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0))
    maximum = flat_values.masked_fill(~expanded, -torch.inf).max(dim=1).values
    return torch.cat([mean, std, maximum], dim=-1)


def pool_single(single: Tensor, residue_mask: Tensor) -> Tensor:
    """Pool a single-residue representation."""
    return masked_mean_std_max(single, residue_mask)


def separation_band_masks(pair_mask: Tensor, exclude_diagonal: bool = True) -> dict[str, Tensor]:
    """Return global, local, medium, and long masks for a pair representation."""
    if pair_mask.ndim != 3:
        raise ValueError("pair_mask must be [B,L,L]")
    length = pair_mask.shape[-1]
    offsets = torch.arange(length, device=pair_mask.device)
    separation = (offsets[:, None] - offsets[None, :]).abs()
    global_mask = pair_mask.clone()
    if exclude_diagonal:
        global_mask &= separation.ne(0)
    return {
        "global": global_mask,
        "local": global_mask & separation.ge(1) & separation.le(8),
        "medium": global_mask & separation.ge(9) & separation.le(32),
        "long": global_mask & separation.gt(32),
    }


def pool_pair(pair: Tensor, pair_mask: Tensor, exclude_diagonal: bool = True) -> Tensor:
    """Pool global and separation-band pair features."""
    features = []
    for mask in separation_band_masks(pair_mask, exclude_diagonal).values():
        if not mask.flatten(1).any(dim=1).all():
            per_example = []
            for values, example_mask in zip(pair, mask, strict=True):
                if example_mask.any():
                    per_example.append(
                        masked_mean_std_max(values.unsqueeze(0), example_mask.unsqueeze(0))[0]
                    )
                else:
                    per_example.append(
                        torch.zeros(pair.shape[-1] * 3, device=pair.device, dtype=pair.dtype)
                    )
            features.append(torch.stack(per_example))
        else:
            features.append(masked_mean_std_max(pair, mask))
    return torch.cat(features, dim=-1)


def pair_per_residue_mean(pair: Tensor, pair_mask: Tensor) -> Tensor:
    """Return a masked per-residue pair summary."""
    counts = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return (pair * pair_mask.unsqueeze(-1)).sum(dim=-2) / counts
