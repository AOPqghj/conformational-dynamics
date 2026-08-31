"""Documented AlphaFold adapter boundary; implement after selecting a distribution."""

from protein_state_router.representations.types import ProteinRepresentation


class AlphaFoldRepresentationExtractor:
    backbone_name = "alphafold"
    backbone_version = "unconfigured"

    def extract(self, protein_id: str, sequence: str) -> ProteinRepresentation:
        raise NotImplementedError(
            "Map final-trunk, final-recycle single [L,d] and pair [L,L,d] tensors here; record "
            "model version, layer, recycle, MSA settings, dtype, and preprocessing in metadata."
        )
