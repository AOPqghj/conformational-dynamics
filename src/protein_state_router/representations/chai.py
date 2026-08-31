"""Chai adapter boundary."""

from protein_state_router.representations.alphafold import AlphaFoldRepresentationExtractor


class ChaiRepresentationExtractor(AlphaFoldRepresentationExtractor):
    backbone_name = "chai"
