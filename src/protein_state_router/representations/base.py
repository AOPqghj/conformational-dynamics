"""Extractor protocol kept independent of backbone implementation details."""

from typing import Protocol

from protein_state_router.representations.types import ProteinRepresentation


class RepresentationExtractor(Protocol):
    backbone_name: str
    backbone_version: str

    def extract(self, protein_id: str, sequence: str) -> ProteinRepresentation: ...
