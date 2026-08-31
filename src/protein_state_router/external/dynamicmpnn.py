"""Parse trusted DynamicMPNN processed examples as auditable weak-evidence candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from protein_state_router.external.zenodo import DYNAMICMPNN_ZENODO_DOI, DYNAMICMPNN_ZENODO_RECORD
from protein_state_router.representations.query import sequence_sha256

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
CANDIDATE_COLUMNS = (
    "source_dataset",
    "source_record",
    "source_doi",
    "dynamicmpnn_mode",
    "dynamicmpnn_original_split",
    "dynamicmpnn_archive_name",
    "dynamicmpnn_file",
    "dynamicmpnn_cluster_id",
    "dynamicmpnn_example_id",
    "dynamicmpnn_group_id",
    "cluster_members_json",
    "n_available_conformations",
    "all_member_ids_json",
    "pdb_ids_json",
    "chain_ids_json",
    "sequence",
    "sequence_hash",
    "sequence_length",
    "n_standard_residues",
    "fraction_standard_residues",
    "has_tm_scores",
    "tm_scores_shape_json",
    "min_pair_tm",
    "max_pair_tm",
    "mean_pair_tm",
    "max_ca_rmsd",
    "mean_ca_rmsd",
    "aligned_coverage",
    "min_aligned_coverage",
    "min_pairwise_sequence_identity",
    "malformed_object",
    "target_chain_ambiguous",
    "single_structure_insufficient",
    "single_structure_insufficient_candidate",
    "alternate_structured_states",
    "alternate_structured_states_candidate",
    "disorder_or_heterogeneity_candidate",
    "condition_aware_required_candidate",
    "label_confidence_tier",
    "requires_manual_audit",
    "audit_priority",
    "exclusion_reason",
    "notes",
    "sequence_extraction_method",
    "n_sequence_conflicts",
    "sequence_conflict_fraction",
    "load_failed",
    "load_error",
)


@dataclass(frozen=True, slots=True)
class DynamicMPNNInspection:
    file: str
    object_type: str
    attributes: list[str]
    cluster_members: list[str]
    n_cluster_members: int
    pyg_dict_keys: list[str]
    has_tm_scores: bool
    tm_scores_shape: list[int] | None
    load_failed: bool = False
    load_error: str | None = None


def inspect_pt(path: str | Path) -> DynamicMPNNInspection:
    """Inspect a trusted DynamicMPNN file; individual failures are represented, not raised."""
    return _load_inspection(Path(path))[0]


def _load_inspection(path: Path) -> tuple[DynamicMPNNInspection, Any | None]:
    """Load one trusted file and return both its inspection summary and value."""
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        members = [str(item) for item in _field(value, "cluster_members", [])]
        pyg = _field(value, "pyg_dict", {}) or {}
        tm_scores = _field(value, "tm_scores", None)
        shape = list(np.asarray(tm_scores).shape) if tm_scores is not None else None
        return (
            DynamicMPNNInspection(
                path.name,
                type(value).__name__,
                _attributes(value),
                members,
                len(members),
                [str(key) for key in pyg.keys()] if hasattr(pyg, "keys") else [],
                tm_scores is not None,
                shape,
            ),
            value,
        )
    except (
        Exception
    ) as error:  # trusted file still may require unavailable optional classes or be malformed
        return (
            DynamicMPNNInspection(path.name, "", [], [], 0, [], False, None, True, repr(error)),
            None,
        )


def inspect_directory(input_dir: str | Path, max_files: int | None = None) -> list[dict[str, Any]]:
    paths = sorted(Path(input_dir).rglob("*.pt"))
    if max_files is not None:
        paths = paths[:max_files]
    return [asdict(inspect_pt(path)) for path in paths]


def build_candidate_catalog(
    dynamicmpnn_root: str | Path,
    modes: Iterable[str] = ("single_chain",),
    splits: Iterable[str] = ("val", "test"),
    max_sequence_conflict_fraction: float = 0.05,
) -> pd.DataFrame:
    """Build one weak-evidence row per processed DynamicMPNN example file."""
    root = Path(dynamicmpnn_root)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for split in splits:
            folder = root / f"{split}_pt_{mode}"
            for path in sorted(folder.rglob("*.pt")) if folder.is_dir() else []:
                rows.append(_candidate_from_pt(path, mode, split, max_sequence_conflict_fraction))
    return validate_candidate_catalog(pd.DataFrame(rows, columns=CANDIDATE_COLUMNS))


def router_positive_candidates(frame: pd.DataFrame, criteria: Mapping[str, Any]) -> pd.DataFrame:
    """Filter auditable candidate rows without converting them into final router labels."""
    valid = frame.copy()
    mask = (
        valid.sequence_length.between(
            criteria["min_sequence_length"], criteria["max_sequence_length"]
        )
        & valid.fraction_standard_residues.ge(criteria["min_standard_fraction"])
        & valid.n_available_conformations.ge(criteria["min_conformations"])
        & ~valid.load_failed
        & valid.exclusion_reason.isna()
    )
    coverage = valid.min_aligned_coverage
    mask &= coverage.isna() | coverage.ge(criteria["min_aligned_coverage"])
    if criteria["strict_structural_filter"]:
        mask &= valid.max_ca_rmsd.ge(criteria["min_ca_rmsd"]) | valid.min_pair_tm.le(
            criteria["max_pair_tm"]
        )
    return valid.loc[mask].copy()


def validate_candidate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CANDIDATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"DynamicMPNN candidate catalog is missing columns: {sorted(missing)}")
    if frame.single_structure_insufficient.notna().any():
        raise ValueError(
            "DynamicMPNN candidates must not be assigned final router labels by default"
        )
    if frame.dynamicmpnn_example_id.duplicated().any():
        raise ValueError("DynamicMPNN example IDs must be unique")
    return pd.DataFrame(frame.loc[:, list(CANDIDATE_COLUMNS)].copy())


def _candidate_from_pt(path: Path, mode: str, split: str, conflict_limit: float) -> dict[str, Any]:
    inspected, value = _load_inspection(path)
    row = _empty_candidate(path, mode, split, inspected)
    if inspected.load_failed:
        row.update(
            {
                "load_failed": True,
                "load_error": inspected.load_error,
                "exclusion_reason": "load_failed",
            }
        )
        return row
    members = inspected.cluster_members
    pdb_ids, chain_ids = _member_ids(members)
    target_chain_ambiguous = any(parse_member_identifier(member) is None for member in members)
    pyg_dict = _field(value, "pyg_dict", {})
    malformed = not hasattr(pyg_dict, "values") or not members
    row.update(
        {
            "cluster_members_json": json.dumps(members),
            "all_member_ids_json": json.dumps(members),
            "pdb_ids_json": json.dumps(pdb_ids),
            "chain_ids_json": json.dumps(chain_ids),
            "malformed_object": malformed,
            "target_chain_ambiguous": target_chain_ambiguous,
        }
    )
    if malformed:
        row["exclusion_reason"] = "malformed_object"
    elif target_chain_ambiguous:
        row["exclusion_reason"] = "ambiguous_target_chain"
    sequences = _member_sequences(pyg_dict)
    if sequences:
        sequence, conflicts, fraction = _consensus_sequence(sequences)
        row.update(
            {
                "sequence": sequence,
                "sequence_hash": sequence_sha256(sequence),
                "sequence_length": len(sequence),
                "n_standard_residues": sum(letter in STANDARD_AA for letter in sequence),
                "fraction_standard_residues": sum(letter in STANDARD_AA for letter in sequence)
                / max(1, len(sequence)),
                "sequence_extraction_method": "target_chain_consensus",
                "n_sequence_conflicts": conflicts,
                "sequence_conflict_fraction": fraction,
                "min_pairwise_sequence_identity": _minimum_pairwise_identity(sequences),
            }
        )
        if fraction > conflict_limit:
            row["exclusion_reason"] = row["exclusion_reason"] or "high_sequence_conflict"
    else:
        row["exclusion_reason"] = row["exclusion_reason"] or "sequence_extraction_failed"
    row.update(_tm_summary(_field(value, "tm_scores", None)))
    row.update(_coordinate_summary(pyg_dict))
    row["audit_priority"] = _audit_priority(row["max_ca_rmsd"], row["min_pair_tm"])
    return row


def _empty_candidate(
    path: Path, mode: str, split: str, inspected: DynamicMPNNInspection
) -> dict[str, Any]:
    archive = f"{split}_pt_{mode}.tar.gz" if split == "train" else f"{split}_pt_{mode}.tar"
    return {
        "source_dataset": "DynamicMPNN",
        "source_record": f"Zenodo {DYNAMICMPNN_ZENODO_RECORD}",
        "source_doi": DYNAMICMPNN_ZENODO_DOI,
        "dynamicmpnn_mode": mode,
        "dynamicmpnn_original_split": split,
        "dynamicmpnn_archive_name": archive,
        "dynamicmpnn_file": str(path),
        "dynamicmpnn_cluster_id": path.stem,
        "dynamicmpnn_example_id": f"{split}:{mode}:{path.stem}",
        "dynamicmpnn_group_id": f"{split}:{path.stem}",
        "cluster_members_json": "[]",
        "n_available_conformations": inspected.n_cluster_members,
        "all_member_ids_json": "[]",
        "pdb_ids_json": "[]",
        "chain_ids_json": "[]",
        "sequence": None,
        "sequence_hash": None,
        "sequence_length": None,
        "n_standard_residues": None,
        "fraction_standard_residues": None,
        "has_tm_scores": inspected.has_tm_scores,
        "tm_scores_shape_json": json.dumps(inspected.tm_scores_shape),
        "min_pair_tm": None,
        "max_pair_tm": None,
        "mean_pair_tm": None,
        "max_ca_rmsd": None,
        "mean_ca_rmsd": None,
        "aligned_coverage": None,
        "min_aligned_coverage": None,
        "min_pairwise_sequence_identity": None,
        "malformed_object": False,
        "target_chain_ambiguous": False,
        "single_structure_insufficient": None,
        "single_structure_insufficient_candidate": True,
        "alternate_structured_states": None,
        "alternate_structured_states_candidate": True,
        "disorder_or_heterogeneity_candidate": None,
        "condition_aware_required_candidate": None,
        "label_confidence_tier": "C_dynamicmpnn_unreviewed",
        "requires_manual_audit": True,
        "audit_priority": "low",
        "exclusion_reason": None,
        "notes": "DynamicMPNN candidate evidence; not a final router label.",
        "sequence_extraction_method": None,
        "n_sequence_conflicts": None,
        "sequence_conflict_fraction": None,
        "load_failed": False,
        "load_error": None,
    }


def _field(value: object, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _attributes(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    return sorted(vars(value)) if hasattr(value, "__dict__") else []


def _member_ids(members: list[str]) -> tuple[list[str], list[str]]:
    pdb_ids, chain_ids = set(), set()
    for member in members:
        parsed = parse_member_identifier(member)
        if parsed is not None:
            pdb_id, chain_id = parsed
            pdb_ids.add(pdb_id)
            chain_ids.add(chain_id)
    return sorted(pdb_ids), sorted(chain_ids)


def parse_member_identifier(member: str) -> tuple[str, str] | None:
    """Parse a DynamicMPNN member into its PDB and author-chain identifiers."""
    match = re.fullmatch(r"([0-9A-Za-z]{4})(?:-\d+)?_([^_]+)", member.strip())
    if not match:
        return None
    return match.group(1).upper(), match.group(2)


def _member_sequences(pyg_dict: Any) -> list[str]:
    values = pyg_dict.values() if hasattr(pyg_dict, "values") else []
    sequences = [_sequence_from_data(value) for value in values]
    return [sequence for sequence in sequences if sequence]


def _sequence_from_data(data: Any) -> str | None:
    for name in ("sequence", "seq"):
        value = _field(data, name, None)
        if isinstance(value, str) and value:
            return value.upper()
    for name in ("residue_type", "aatype", "x"):
        value = _field(data, name, None)
        if value is None:
            continue
        array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
        if array.ndim == 1 and np.issubdtype(array.dtype, np.integer):
            return "".join(
                STANDARD_AA[int(index)] if 0 <= int(index) < 20 else "X" for index in array
            )
        if array.ndim == 2 and array.shape[1] >= 20:
            return "".join(STANDARD_AA[int(index)] for index in array[:, :20].argmax(axis=1))
    return None


def _consensus_sequence(sequences: list[str]) -> tuple[str, int, float]:
    width = max(map(len, sequences))
    consensus, conflicts = [], 0
    for index in range(width):
        letters = [
            sequence[index]
            for sequence in sequences
            if index < len(sequence) and sequence[index] != "-"
        ]
        if not letters:
            continue
        unique = sorted(set(letters))
        if len(unique) > 1:
            conflicts += 1
        consensus.append(max(unique, key=letters.count))
    result = "".join(consensus)
    return result, conflicts, conflicts / max(1, len(result))


def _minimum_pairwise_identity(sequences: list[str]) -> float | None:
    identities = [
        _sequence_identity(first, second)
        for index, first in enumerate(sequences)
        for second in sequences[index + 1 :]
    ]
    return min(identities) if identities else None


def _sequence_identity(first: str, second: str) -> float:
    width = max(len(first), len(second))
    if not width:
        return 0.0
    return sum(left == right for left, right in zip(first, second, strict=False)) / width


def _tm_summary(tm_scores: Any) -> dict[str, Any]:
    if tm_scores is None:
        return {
            "has_tm_scores": False,
            "tm_scores_shape_json": "null",
            "min_pair_tm": None,
            "max_pair_tm": None,
            "mean_pair_tm": None,
        }
    values = np.asarray(tm_scores, dtype=float)
    pairs = values[~np.eye(values.shape[0], dtype=bool)] if values.ndim == 2 else values.reshape(-1)
    pairs = pairs[np.isfinite(pairs)]
    return {
        "has_tm_scores": True,
        "tm_scores_shape_json": json.dumps(list(values.shape)),
        "min_pair_tm": float(pairs.min()) if pairs.size else None,
        "max_pair_tm": float(pairs.max()) if pairs.size else None,
        "mean_pair_tm": float(pairs.mean()) if pairs.size else None,
    }


def _coordinate_summary(pyg_dict: Any) -> dict[str, float | None]:
    coordinates: list[tuple[np.ndarray, np.ndarray]] = []
    for data in _values(pyg_dict):
        coordinate = _coordinate_residues(data)
        if coordinate is not None:
            coordinates.append(coordinate)
    rmsds, coverages = [], []
    for index, (first_indices, first_coordinates) in enumerate(coordinates):
        for second_indices, second_coordinates in coordinates[index + 1 :]:
            _, first_positions, second_positions = np.intersect1d(
                first_indices, second_indices, return_indices=True
            )
            first_values, second_values = (
                first_coordinates[first_positions],
                second_coordinates[second_positions],
            )
            valid = (
                np.isfinite(first_values).all(axis=1)
                & np.isfinite(second_values).all(axis=1)
                & (np.abs(first_values) < 1e4).all(axis=1)
                & (np.abs(second_values) < 1e4).all(axis=1)
            )
            if valid.sum() < 3:
                continue
            aligned = _kabsch_rmsd(first_values[valid], second_values[valid])
            rmsds.append(aligned)
            coverages.append(valid.sum() / max(len(first_indices), len(second_indices)))
    return {
        "max_ca_rmsd": max(rmsds) if rmsds else None,
        "mean_ca_rmsd": float(np.mean(rmsds)) if rmsds else None,
        "aligned_coverage": min(coverages) if coverages else None,
        "min_aligned_coverage": min(coverages) if coverages else None,
    }


def _coordinate_residues(data: Any) -> tuple[np.ndarray, np.ndarray] | None:
    residue_indices = _field(data, "residue_index", None)
    if residue_indices is None:
        return None
    indices = np.asarray(
        residue_indices.detach().cpu() if hasattr(residue_indices, "detach") else residue_indices
    )
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        return None
    for name in ("pos", "coords", "ca_coords"):
        value = _field(data, name, None)
        if value is None:
            continue
        array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=float)
        if array.ndim == 2 and array.shape[1] == 3:
            return (indices, array) if len(indices) == len(array) else None
        if array.ndim == 3 and array.shape[1:] == (3, 3):
            # DynamicMPNN stores backbone N/CA/C coordinates in this order.
            return (indices, array[:, 1, :]) if len(indices) == len(array) else None
    return None


def _values(value: Any) -> Iterable[Any]:
    return value.values() if hasattr(value, "values") else []


def _kabsch_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    first, second = first - first.mean(0), second - second.mean(0)
    left, _, right = np.linalg.svd(first.T @ second)
    rotation = right.T @ np.diag([1, 1, np.linalg.det(right.T @ left.T)]) @ left.T
    return float(np.sqrt(np.mean(np.sum((first @ rotation - second) ** 2, axis=1))))


def _audit_priority(max_rmsd: float | None, min_tm: float | None) -> str:
    if (max_rmsd is not None and max_rmsd >= 5.0) or (min_tm is not None and min_tm <= 0.65):
        return "high"
    if (max_rmsd is not None and max_rmsd >= 2.0) or (min_tm is not None and min_tm <= 0.80):
        return "medium"
    return "low"
