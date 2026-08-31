"""Leakage-aware foundations for protein-classifier interpretability experiments."""

from interpretability.contracts import (
    AuditSummary,
    EmbeddingAudit,
    FrozenModelAudit,
    apply_reference_split,
    audit_inputs,
    load_residue_matrix,
    pool_residue_matrix,
    read_table,
    validate_catalog,
    validate_embedding_manifest,
    validate_feature_columns,
    validate_frozen_models,
)
from interpretability.model import FrozenResidueCNNModel, FrozenResidueStackedModel

__all__ = [
    "AuditSummary",
    "EmbeddingAudit",
    "FrozenModelAudit",
    "apply_reference_split",
    "audit_inputs",
    "load_residue_matrix",
    "pool_residue_matrix",
    "read_table",
    "validate_catalog",
    "validate_embedding_manifest",
    "validate_feature_columns",
    "validate_frozen_models",
    "FrozenResidueCNNModel",
    "FrozenResidueStackedModel",
]
