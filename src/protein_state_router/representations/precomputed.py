"""Loader for a cache generated outside this project."""

from pathlib import Path

from protein_state_router.representations.cache import load_cached_representation
from protein_state_router.representations.types import ProteinRepresentation


class PrecomputedRepresentationExtractor:
    backbone_name = "precomputed"
    backbone_version = "unknown"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def extract(self, protein_id: str, sequence: str) -> ProteinRepresentation:
        representation = load_cached_representation(self.root / protein_id)
        if representation.residue_mask.numel() != len(sequence):
            raise ValueError(f"Cached representation length differs for {protein_id}")
        return representation
