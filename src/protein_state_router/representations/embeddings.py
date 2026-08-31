"""Typed, backbone-neutral embedding records used outside backend runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(frozen=True, slots=True)
class EmbeddingSource:
    """Immutable provenance shared by single and pair tensors."""

    backend: str
    backend_version: str
    model_name: str
    extraction_config_hash: str
    sequence_sha256: str
    representation_layer: str | None = None
    recycle_index: int | None = None


@dataclass(frozen=True, slots=True)
class SingleEmbedding:
    values: Tensor  # [L, d_single]
    residue_mask: Tensor  # [L]
    source: EmbeddingSource
    token_indices: Tensor | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.residue_mask.ndim != 1:
            raise ValueError("single embedding must be [L,d] with [L] residue mask")
        if self.values.shape[0] != self.residue_mask.numel():
            raise ValueError("single embedding and residue mask length differ")


@dataclass(frozen=True, slots=True)
class PairEmbedding:
    values: Tensor  # [L, L, d_pair]
    pair_mask: Tensor  # [L, L]
    source: EmbeddingSource
    diagonal_included: bool = True

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.pair_mask.ndim != 2:
            raise ValueError("pair embedding must be [L,L,d] with [L,L] pair mask")
        if (
            self.values.shape[:2] != self.pair_mask.shape
            or self.values.shape[0] != self.values.shape[1]
        ):
            raise ValueError("pair embedding and pair mask must have matching square dimensions")


@dataclass(frozen=True, slots=True)
class ProteinEmbeddings:
    protein_id: str
    sequence: str
    sequence_sha256: str
    source: EmbeddingSource
    single: SingleEmbedding | None
    pair: PairEmbedding | None
    confidence_features: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.single is None and self.pair is None:
            raise ValueError("at least one embedding modality is required")
        length = len(self.sequence)
        if self.single is not None and self.single.values.shape[0] != length:
            raise ValueError("single embedding length differs from canonical sequence")
        if self.pair is not None and self.pair.values.shape[0] != length:
            raise ValueError("pair embedding length differs from canonical sequence")

    @property
    def residue_mask(self) -> Tensor:
        if self.single is not None:
            return self.single.residue_mask
        assert self.pair is not None
        return self.pair.pair_mask.any(dim=-1)
