"""Canonical residue-representation contracts used across experiment CLIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """Immutable shape and provenance contract for one residue representation."""

    name: str
    width: int
    representation_layer: str
    model_id: str
    model_revision: str
    max_sequence_length: int | None = None


REPRESENTATIONS: dict[str, RepresentationSpec] = {
    "esmfold": RepresentationSpec(
        "esmfold", 1024, "folding_trunk_s_s", "facebook/esmfold_v1",
        "75a3841ee059df2bf4d56688166c8fb459ddd97a", 1022,
    ),
    "bioemu": RepresentationSpec(
        "bioemu", 384, "alphafold2_evoformer_single",
        "bioemu.colabfold_inline.alphafold2_model_3", "1.4.1",
    ),
    "random_esmfold_trunk": RepresentationSpec(
        "random_esmfold_trunk", 1024, "folding_trunk_s_s",
        "random_esmfold_folding_trunk", "seed42_v1", 1022,
    ),
    "bioemu_no_msa": RepresentationSpec(
        "bioemu_no_msa", 384, "alphafold2_evoformer_single",
        "bioemu.colabfold_inline.alphafold2_model_3", "1.4.1",
    ),
    "esm2_3b": RepresentationSpec(
        "esm2_3b", 2560, "esm2_final_hidden_state",
        "facebook/esm2_t36_3B_UR50D", "7bfbb6ae874b2d2948a5ecb2a62fbad7e9083c32", 1022,
    ),
}


def representation_choices() -> tuple[str, ...]:
    return tuple(REPRESENTATIONS)


def representation_spec(name: str, width: int | None = None) -> RepresentationSpec:
    """Return a registered representation and optionally enforce its width."""
    try:
        spec = REPRESENTATIONS[name]
    except KeyError as error:
        raise ValueError(f"unsupported representation: {name}") from error
    if width is not None and width != spec.width:
        raise ValueError(f"{name} embeddings must have width {spec.width}, got {width}")
    return spec
