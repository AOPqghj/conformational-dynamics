"""Normalize the lightweight public PATHpre SS/MS release into protein records.

PATHpre publishes compact PDB-chain lists as well as multi-gigabyte coordinate
archives.  This module deliberately starts with the lists: source labels and
chain provenance remain useful for an external benchmark even when coordinate
geometry has not yet been independently recomputed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

import pandas as pd

from protein_state_router.representations.query import sequence_sha256

PATHPRE_REFERENCE = "PATHpre Zenodo release 10.5281/zenodo.13337019"
_IDENTIFIER = re.compile(r"^(?P<pdb>[0-9A-Za-z]{4})(?P<chain>[A-Za-z0-9]+)$")


@dataclass(frozen=True, slots=True)
class PathpreChain:
    """Resolved canonical sequence for one PATHpre PDB chain."""

    pdb_id: str
    chain_id: str
    sequence: str
    uniprot_id: str | None = None


def parse_identifier(value: str) -> tuple[str, str]:
    """Split a PATHpre identifier such as ``1a2wA`` into PDB and author chain."""
    match = _IDENTIFIER.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"invalid PATHpre PDB-chain identifier: {value!r}")
    return match.group("pdb").upper(), match.group("chain")


def inspect_release(source_dir: str | Path) -> pd.DataFrame:
    """Classify compact PATHpre text tables without relying on their file names."""
    records: list[dict[str, object]] = []
    for path in sorted(Path(source_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() not in {"", ".txt", ".tsv", ".csv"}:
            continue
        lines = [
            line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()
        ]
        tokens = [re.split(r"[\s,;]+", line) for line in lines]
        one = sum(len(parts) == 1 and _is_identifier(parts[0]) for parts in tokens)
        two = sum(
            len(parts) == 2 and _is_identifier(parts[0]) and _is_identifier(parts[1])
            for parts in tokens
        )
        records.append(
            {
                "path": str(path),
                "line_count": len(lines),
                "single_chain_rows": one,
                "chain_pair_rows": two,
                "suggested_class": "MS" if two and two >= one else "SS" if one else "unrecognized",
            }
        )
    return pd.DataFrame(records)


def read_release(source_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read discovered MS chain-pair and SS chain-list tables from a release directory."""
    inspection = inspect_release(source_dir)
    if inspection.empty:
        raise FileNotFoundError(f"no PATHpre text files found in {source_dir}")
    ms_rows: list[dict[str, str]] = []
    ss_rows: list[dict[str, str]] = []
    for item in inspection.to_dict("records"):
        path = Path(str(item["path"]))
        kind = str(item["suggested_class"])
        if kind == "unrecognized":
            continue
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            pieces = re.split(r"[\s,;]+", line.strip())
            if kind == "MS" and len(pieces) == 2 and all(_is_identifier(value) for value in pieces):
                ms_rows.append(
                    {
                        "source_file": path.name,
                        "source_line": str(line_number),
                        "state_a": pieces[0],
                        "state_b": pieces[1],
                    }
                )
            elif kind == "SS" and len(pieces) == 1 and _is_identifier(pieces[0]):
                ss_rows.append(
                    {
                        "source_file": path.name,
                        "source_line": str(line_number),
                        "structure": pieces[0],
                    }
                )
    if not ms_rows or not ss_rows:
        raise ValueError("PATHpre release must contain both recognizable MS pairs and SS chains")
    return pd.DataFrame(ms_rows), pd.DataFrame(ss_rows)


def source_manifest(source_dir: str | Path) -> dict[str, object]:
    """Record checksums for the compact official files used by an ingestion."""
    root = Path(source_dir)
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in {"", ".txt", ".tsv", ".csv"}
    }
    return {"source_dataset": "PATHpre", "source_reference": PATHPRE_REFERENCE, "files": files}


def build_catalogs(
    ms_rows: pd.DataFrame,
    ss_rows: pd.DataFrame,
    *,
    resolver: Callable[[str, str], PathpreChain | None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve source rows, retaining unresolvable or conflicted rows in the audit table."""
    ms_catalog, ms_audit = _resolve_ms(ms_rows, resolver)
    ss_catalog, ss_audit = _resolve_ss(ss_rows, resolver)
    audit = pd.concat([ms_audit, ss_audit], ignore_index=True, sort=False)

    # A sequence assigned to both source classes is never silently used as a
    # benchmark label.  Keep the source records in audit for human review.
    shared = set(ms_catalog.sequence_hash) & set(ss_catalog.sequence_hash)
    if shared:
        for _label, catalog in (("MS", ms_catalog), ("SS", ss_catalog)):
            collided = catalog.loc[catalog.sequence_hash.isin(shared)].copy()
            if not collided.empty:
                collided["audit_reason"] = "sequence_occurs_in_both_pathpre_classes"
                audit = pd.concat([audit, collided], ignore_index=True, sort=False)
            catalog.drop(catalog.index[catalog.sequence_hash.isin(shared)], inplace=True)
    return (
        ms_catalog.reset_index(drop=True),
        ss_catalog.reset_index(drop=True),
        audit.reset_index(drop=True),
    )


def _resolve_ms(
    rows: pd.DataFrame, resolver: Callable[[str, str], PathpreChain | None]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for row in cast(list[dict[str, object]], rows.to_dict("records")):
        source_id = f"{row['source_file']}:{row['source_line']}"
        try:
            a_pdb, a_chain = parse_identifier(str(row["state_a"]))
            b_pdb, b_chain = parse_identifier(str(row["state_b"]))
            state_a, state_b = resolver(a_pdb, a_chain), resolver(b_pdb, b_chain)
        except (ValueError, KeyError) as error:
            state_a = state_b = None
            audit.append({**row, "source_record_id": source_id, "audit_reason": str(error)})
        if state_a is None or state_b is None:
            if not any(item.get("source_record_id") == source_id for item in audit):
                audit.append(
                    {**row, "source_record_id": source_id, "audit_reason": "sequence_lookup_failed"}
                )
            continue
        identity = SequenceMatcher(None, state_a.sequence, state_b.sequence, autojunk=False).ratio()
        if identity < 0.95:
            audit.append(
                {
                    **row,
                    "source_record_id": source_id,
                    "audit_reason": "state_sequence_identity_below_0.95",
                    "sequence_identity_between_states": identity,
                }
            )
            continue
        accepted.append(
            _record(
                protein_id=f"pathpre:MS:{source_id}",
                source_record_id=source_id,
                sequence=max((state_a.sequence, state_b.sequence), key=len),
                uniprot_id=state_a.uniprot_id or state_b.uniprot_id,
                pathpre_class="MS",
                structures=[f"{a_pdb}_{a_chain}", f"{b_pdb}_{b_chain}"],
                state_a=[f"{a_pdb}_{a_chain}"],
                state_b=[f"{b_pdb}_{b_chain}"],
                state_identity=identity,
            )
        )
    return _deduplicate(accepted, audit, "duplicate_ms_sequence")


def _resolve_ss(
    rows: pd.DataFrame, resolver: Callable[[str, str], PathpreChain | None]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for row in cast(list[dict[str, object]], rows.to_dict("records")):
        source_id = f"{row['source_file']}:{row['source_line']}"
        try:
            pdb_id, chain_id = parse_identifier(str(row["structure"]))
            resolved = resolver(pdb_id, chain_id)
        except (ValueError, KeyError) as error:
            resolved = None
            audit.append({**row, "source_record_id": source_id, "audit_reason": str(error)})
        if resolved is None:
            if not any(item.get("source_record_id") == source_id for item in audit):
                audit.append(
                    {**row, "source_record_id": source_id, "audit_reason": "sequence_lookup_failed"}
                )
            continue
        accepted.append(
            _record(
                protein_id=f"pathpre:SS:{source_id}",
                source_record_id=source_id,
                sequence=resolved.sequence,
                uniprot_id=resolved.uniprot_id,
                pathpre_class="SS",
                structures=[f"{pdb_id}_{chain_id}"],
                state_a=[],
                state_b=[],
            )
        )
    return _deduplicate(accepted, audit, "duplicate_ss_sequence")


def _record(
    *,
    protein_id: str,
    source_record_id: str,
    sequence: str,
    uniprot_id: str | None,
    pathpre_class: str,
    structures: Iterable[str],
    state_a: Iterable[str],
    state_b: Iterable[str],
    state_identity: float | None = None,
) -> dict[str, object]:
    label = int(pathpre_class == "MS")
    return {
        "protein_id": protein_id,
        "uniprot_id": uniprot_id,
        "sequence": sequence,
        "sequence_hash": sequence_sha256(sequence),
        "sequence_length": len(sequence),
        "source_dataset": "PATHpre",
        "source_record_id": source_record_id,
        "source_reference": PATHPRE_REFERENCE,
        "pathpre_class": pathpre_class,
        "structure_ids_json": json.dumps(list(structures)),
        "state_a_structure_ids_json": json.dumps(list(state_a)),
        "state_b_structure_ids_json": json.dumps(list(state_b)),
        "n_experimental_structures": len(list(structures)),
        "max_ca_rmsd": None,
        "min_tm_score": None,
        "aligned_coverage": None,
        "sequence_identity_between_states": state_identity,
        "label": label,
        "dataset_label": label,
        "label_class": "alternate_structured_state"
        if label
        else "single_dominant_structured_state",
        "label_confidence": "silver",
        "single_structure_insufficient_derived": bool(label),
        "is_training_ready": False,
        "requires_manual_audit": True,
        "label_notes": "Source label preserved; independent coordinate geometry not recomputed.",
    }


def _deduplicate(
    records: list[dict[str, object]], audit: list[dict[str, object]], reason: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, pd.DataFrame(audit)
    duplicate = frame.duplicated("sequence_hash", keep="first")
    if duplicate.any():
        duplicates = frame.loc[duplicate].copy()
        duplicates["audit_reason"] = reason
        audit.extend(cast(list[dict[str, object]], duplicates.to_dict("records")))
        frame = frame.loc[~duplicate].copy()
    return frame, pd.DataFrame(audit)


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(str(value).strip()))
