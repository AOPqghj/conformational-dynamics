"""Validated catalog schema and parquet conversion."""

from __future__ import annotations

from typing import Any

import pandas as pd
from protein_state_router.constants import AMINO_ACIDS, CONFIDENCE_TIERS
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CatalogRecord(BaseModel):
    """Evidence-backed record used by every training and evaluation step."""

    model_config = ConfigDict(extra="allow")

    protein_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    sequence_length: int | None = None
    single_structure_insufficient: bool
    label_confidence_tier: str
    source_dataset: str
    source_reference: str = ""
    evidence_type: str = ""
    evidence_summary: str = ""
    family_id: str | None = None
    sequence_cluster_id: str
    structure_ids: list[str] = Field(default_factory=list)
    experimental_conditions: list[str] = Field(default_factory=list)
    ligands_or_partners: list[str] = Field(default_factory=list)
    alternate_structured_states: bool | None = None
    disorder_or_heterogeneity: bool | None = None
    condition_aware_required: bool | None = None
    fold_switching: bool | None = None
    representation_backbone: str | None = None
    representation_version: str | None = None
    split: str | None = None

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        sequence = value.upper().strip()
        invalid = set(sequence) - AMINO_ACIDS
        if invalid:
            raise ValueError(f"Illegal amino-acid characters: {''.join(sorted(invalid))}")
        return sequence

    @field_validator("label_confidence_tier")
    @classmethod
    def validate_tier(cls, value: str) -> str:
        tier = value.upper()
        if tier not in CONFIDENCE_TIERS:
            raise ValueError(f"confidence tier must be one of {sorted(CONFIDENCE_TIERS)}")
        return tier

    @model_validator(mode="after")
    def add_length(self) -> CatalogRecord:
        if self.sequence_length is None:
            self.sequence_length = len(self.sequence)
        if self.sequence_length != len(self.sequence):
            raise ValueError("sequence_length must equal sequence length")
        return self


def validate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate records, canonicalize fields, and reject duplicate IDs."""
    if frame.empty:
        raise ValueError("Catalog cannot be empty")
    records = [CatalogRecord.model_validate(row).model_dump() for row in frame.to_dict("records")]
    validated = pd.DataFrame(records)
    if validated.protein_id.duplicated().any():
        raise ValueError("Duplicate protein_id values are not allowed")
    return validated


def read_catalog(path: str) -> pd.DataFrame:
    return validate_catalog(pd.read_parquet(path))


def write_catalog(frame: pd.DataFrame, path: str) -> None:
    validate_catalog(frame).to_parquet(path, index=False)


def records_from_dicts(records: list[dict[str, Any]]) -> pd.DataFrame:
    return validate_catalog(pd.DataFrame(records))
