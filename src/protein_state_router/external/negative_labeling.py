"""Conservative, positive-exclusion labeling for structured negative examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def label_negative_candidates(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    dynamicmpnn_positives: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Label candidates using precomputed structural evidence.

    This function is deliberately offline: coordinate/evidence generation belongs to
    an upstream importer. Incomplete/weak evidence remains a bronze audit candidate,
    never a training-ready negative.
    """
    frame = candidates.copy()
    root = config.get("negative_labeling", config)
    structure = root.get("structure", {})
    labeling = root.get("labeling", root.get("labeling", {}))
    minimum_structures = int(
        structure.get("min_num_experimental_structures", labeling.get("minimum_structures", 3))
    )
    min_coverage = float(
        structure.get("min_aligned_coverage", labeling.get("minimum_aligned_coverage", 0.8))
    )
    max_rmsd = float(structure.get("max_negative_ca_rmsd", labeling.get("maximum_ca_rmsd", 2.0)))
    ambiguous_rmsd = float(
        structure.get("ambiguous_rmsd_upper", labeling.get("ambiguous_ca_rmsd", 4.0))
    )
    min_tm = float(structure.get("min_negative_tm_score", labeling.get("minimum_tm_score", 0.95)))
    positive_ids = _positive_ids(dynamicmpnn_positives)
    labels: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        identity = _first(row, "uniprot_id", "primary_accession", "protein_id")
        evidence = _first(row, "condition_signal", "condition_aware", "has_condition_signal")
        n_structures = _number(row, "n_structures", "structure_count", "n_available_conformations")
        coverage = _number(row, "aligned_coverage", "min_aligned_coverage", "coverage")
        rmsd = _number(row, "max_ca_rmsd", "ca_rmsd", "rmsd")
        tm = _number(row, "min_tm_score", "min_pair_tm", "tm_score")
        reason = ""
        confidence = "bronze"
        label_class = "single_dominant_structured_state"
        audit = True
        if identity is not None and str(identity) in positive_ids:
            confidence = "excluded"
            label_class = "excluded"
            audit = False
            reason = "dynamicmpnn_positive_overlap"
        elif _truth(evidence):
            reason = "condition_signal"
        elif n_structures is None or n_structures < minimum_structures:
            reason = "insufficient_structures"
        elif coverage is None or coverage < min_coverage:
            reason = "insufficient_aligned_coverage"
        elif rmsd is None:
            reason = "missing_rmsd"
        elif rmsd > ambiguous_rmsd:
            confidence = "excluded"
            label_class = "excluded"
            audit = False
            reason = "high_structural_diversity"
        elif rmsd > max_rmsd:
            reason, audit = "ambiguous_structural_diversity", True
        elif tm is not None and tm < min_tm:
            reason, audit = "low_tm_score", True
        else:
            label_class = "single_dominant_structured_state"
            confidence = "gold" if tm is not None and tm >= min_tm else "silver"
            audit = False
            reason = "convergent_experimental_structures"
        labels.append(
            {
                "label_class": label_class,
                "label_confidence": confidence,
                "single_structure_insufficient_derived": False,
                "exclusion_reason": reason,
                "requires_manual_audit": audit,
                "is_training_ready_negative": label_class == "single_dominant_structured_state"
                and not audit,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(labels)], axis=1)


def _positive_ids(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or frame.empty:
        return set()
    columns = [
        c
        for c in (
            "uniprot_id",
            "primary_accession",
            "protein_id",
            "sequence_cluster_id",
            "dynamicmpnn_cluster_id",
        )
        if c in frame
    ]
    return {str(value) for column in columns for value in frame[column].dropna()}


def _first(row: pd.Series, *columns: str) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return None


def _number(row: pd.Series, *columns: str) -> float | None:
    value = _first(row, *columns)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _truth(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return value is True or str(value).lower() in {"1", "true", "yes", "y"}
