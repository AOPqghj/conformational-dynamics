"""Backbone-neutral representation contract."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class ProteinRepresentation:
    protein_id: str
    single: Tensor | None
    pair: Tensor | None
    residue_mask: Tensor
    pair_mask: Tensor | None
    confidence_features: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
