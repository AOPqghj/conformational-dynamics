"""Normalize ATLAS' official parsable metadata into conservative SS candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from protein_state_router.representations.query import sequence_sha256

ATLAS_REFERENCE = "ATLAS parsable metadata API (2023_03_09_ATLAS_info.tsv)"
REQUIRED_COLUMNS = {
    "PDB",
    "length",
    "UniProt",
    "sequence",
    "div_SE",
    "div_MM",
    "avg_RMSF",
    "avg_gyration",
}


def read_parsable_table(root: str | Path) -> pd.DataFrame:
    """Read the current ATLAS full-protein table, regardless of release date prefix."""
    matches = sorted(Path(root).rglob("*ATLAS_info.tsv"))
    if not matches:
        raise FileNotFoundError(f"no *ATLAS_info.tsv found below {root}")
    frame = pd.read_csv(matches[-1], sep="\t", dtype=str, keep_default_na=False)
    normalized = frame.rename(columns={"Len.": "length"}).copy()
    missing = REQUIRED_COLUMNS - set(normalized)
    if missing:
        raise ValueError(f"ATLAS table missing columns: {sorted(missing)}")
    for column in ("length", "div_SE", "div_MM", "avg_RMSF", "avg_gyration"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def select_low_flexibility_candidates(
    frame: pd.DataFrame,
    *,
    known_positive_hashes: set[str] | None = None,
    min_length: int = 50,
    min_tm_score: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose low-quartile ATLAS records within length bins for external SS use.

    ATLAS describes MD trajectories rather than a single-state ground truth. The
    resulting records are therefore bronze, explicitly non-training-ready, and
    preserve all selection signals for later review.
    """
    required = REQUIRED_COLUMNS - {"UniProt"}
    if missing := required - set(frame):
        raise ValueError(f"ATLAS frame missing columns: {sorted(missing)}")
    data = frame.copy()
    data["sequence"] = data.sequence.astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    data["sequence_hash"] = data.sequence.map(sequence_sha256)
    data["sequence_length"] = data.sequence.str.len()
    data["length_bin"] = pd.cut(
        data.sequence_length,
        [min_length - 1, 100, 200, 400, 800, float("inf")],
        labels=["50-100", "101-200", "201-400", "401-800", "801+"],
    ).astype(str)
    signals = ["div_SE", "div_MM", "avg_RMSF", "avg_gyration"]
    # ``div_SE`` and ``div_MM`` are ATLAS diversity scores, not the requested
    # trajectory RMSD/RMSF variation measures. Keep them for audit, but only
    # apply the conservative within-bin low-quartile rule to actual mobility
    # signals; requiring all four produces no biologically useful cohort.
    mobility = ["avg_RMSF", "avg_gyration"]
    data["has_required_metrics"] = data[signals].notna().all(axis=1)
    low = pd.Series(True, index=data.index)
    for column in mobility:
        cutoff = data.groupby("length_bin", observed=False)[column].transform("quantile", q=0.25)
        data[f"{column}_low_quartile"] = data[column] <= cutoff
        low &= data[f"{column}_low_quartile"]
    # The table's refinement TM score is a model-vs-experimental quality flag,
    # not an MD trajectory TM score, but is retained as a conservative source
    # quality filter when it is available.
    tm = pd.to_numeric(data.get("refinement_TMscore", pd.Series(index=data.index)), errors="coerce")
    data["min_tm_score_source"] = tm
    data["passes_tm_source_filter"] = tm.isna() | (tm >= min_tm_score)
    positive_hashes = known_positive_hashes or set()
    data["overlaps_known_positive"] = data.sequence_hash.isin(positive_hashes)
    accepted_mask = (
        (data.sequence_length >= min_length)
        & data.has_required_metrics
        & low
        & data.passes_tm_source_filter
        & ~data.overlaps_known_positive
    )
    accepted = data.loc[accepted_mask].copy()
    accepted = accepted.drop_duplicates("sequence_hash", keep="first")
    candidates = pd.DataFrame(
        {
            "protein_id": "atlas:" + accepted.PDB.astype(str),
            "uniprot_id": accepted.UniProt.replace("", None),
            "sequence": accepted.sequence,
            "sequence_hash": accepted.sequence_hash,
            "sequence_length": accepted.sequence_length,
            "source_dataset": "ATLAS",
            "source_record_id": accepted.PDB.astype(str),
            "source_reference": ATLAS_REFERENCE,
            "atlas_pdb_chain": accepted.PDB.astype(str),
            "atlas_signals_json": accepted.apply(
                lambda row: json.dumps(
                    {key: _json_number(row[key]) for key in signals}, sort_keys=True
                ),
                axis=1,
            ),
            "length_bin": accepted.length_bin,
            "dataset_label": 0,
            "label_class": "single_dominant_structured_state_candidate",
            "label_confidence": "bronze",
            "is_training_ready": False,
            "requires_manual_audit": True,
            "label_notes": "Low-flexibility ATLAS MD candidate; not a validated single-state label.",
        }
    )
    exclusions = data.loc[~accepted_mask].copy()
    exclusions["exclusion_reason"] = exclusions.apply(_reason, axis=1)
    return candidates.reset_index(drop=True), exclusions.reset_index(drop=True)


def source_manifest(path: str | Path) -> dict[str, object]:
    """Return stable metadata for the exact ATLAS table used."""
    location = Path(path)
    return {
        "source_dataset": "ATLAS",
        "source_reference": ATLAS_REFERENCE,
        "path": str(location),
        "sha256": hashlib.sha256(location.read_bytes()).hexdigest(),
    }


def _reason(row: pd.Series) -> str:
    if row.sequence_length < 50:
        return "sequence_too_short"
    if not row.has_required_metrics:
        return "missing_md_metrics"
    if row.overlaps_known_positive:
        return "overlaps_known_positive"
    if not row.passes_tm_source_filter:
        return "low_source_refinement_tm"
    return "not_low_quartile_in_length_bin"


def _json_number(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    number = float(str(value))
    return number if pd.notna(number) else None
