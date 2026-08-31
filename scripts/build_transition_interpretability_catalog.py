"""Tabulate verified transition candidates from frozen 8,598-protein provenance."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

CATALOG = Path("data/lifecycle/final/initial_8598_dataset/catalog.parquet")
OUTPUT = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/transition_important_proteins.csv"
)
PDB_CODE = re.compile(r"^[0-9A-Z]{4}$")


def json_list(value: object) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def structure_pdb_codes(structure_ids: list[Any]) -> list[str]:
    codes = {str(item).split("_", 1)[0].upper() for item in structure_ids}
    return sorted(code for code in codes if PDB_CODE.fullmatch(code))


def number(record: dict[str, Any], name: str) -> float | None:
    value = record.get(name)
    if value is None and isinstance(record.get("evidence_summary"), str):
        try:
            value = json.loads(record["evidence_summary"]).get(name)
        except json.JSONDecodeError:
            value = None
    return float(value) if value is not None and pd.notna(value) else None


def first_nonempty(records: list[dict[str, Any]], field: str) -> list[Any]:
    for record in records:
        values = json_list(record.get(field))
        if values:
            return values
    return []


def transition_row(row: Any) -> dict[str, object] | None:
    records = [item for item in json_list(row.source_metadata_json) if isinstance(item, dict)]
    state_a = first_nonempty(records, "state_a_structure_ids_json")
    state_b = first_nonempty(records, "state_b_structure_ids_json")
    structures = first_nonempty(records, "structure_ids_json")
    if state_a and state_b:
        evidence = "explicit_state_partition"
        priority = "direct_pair"
        selected_a, selected_b = str(state_a[0]), str(state_b[0])
    elif len(structures) >= 2:
        evidence = "curated_multi_conformer"
        priority = "coordinate_pair_scan"
        selected_a = selected_b = ""
    else:
        return None
    rmsd_values = [number(record, "max_ca_rmsd") for record in records]
    tm_values = [number(record, "min_pair_tm") for record in records]
    coverage_values = [number(record, "aligned_coverage") for record in records]
    all_structures = sorted({str(item) for item in [*state_a, *state_b, *structures]})
    return {
        "protein_id": row.protein_id,
        "sequence_sha256": row.sequence_sha256,
        "sequence_length": int(row.sequence_length),
        "source_dataset": row.source_dataset,
        "split": row.split,
        "uniprot_accession": row.uniprot_accession,
        "transition_evidence": evidence,
        "transition_priority": priority,
        "state_a_structure_ids_json": json.dumps(state_a, separators=(",", ":")),
        "state_b_structure_ids_json": json.dumps(state_b, separators=(",", ":")),
        "structure_ids_json": json.dumps(all_structures, separators=(",", ":")),
        "state_a_pdb_codes": json.dumps(structure_pdb_codes(state_a), separators=(",", ":")),
        "state_b_pdb_codes": json.dumps(structure_pdb_codes(state_b), separators=(",", ":")),
        "pdb_codes": json.dumps(structure_pdb_codes(all_structures), separators=(",", ":")),
        "selected_state_a_structure_id": selected_a,
        "selected_state_b_structure_id": selected_b,
        "precomputed_max_ca_rmsd": max(
            (value for value in rmsd_values if value is not None), default=None
        ),
        "precomputed_min_pair_tm": min(
            (value for value in tm_values if value is not None), default=None
        ),
        "precomputed_aligned_coverage": min(
            (value for value in coverage_values if value is not None), default=None
        ),
        "coordinate_status": "pdb_mmcif_download_required",
    }


def build(catalog: pd.DataFrame) -> pd.DataFrame:
    positive = catalog.loc[catalog.dataset_label.eq(1)].copy()
    candidates = [transition_row(row) for row in positive.itertuples(index=False)]
    result = pd.DataFrame(candidate for candidate in candidates if candidate is not None)
    if len(result) != len(positive):
        raise ValueError("every positive row must retain multi-conformer transition evidence")
    result["_rank_group"] = result.transition_priority.map(
        {"direct_pair": 0, "coordinate_pair_scan": 1}
    )
    return (
        result.sort_values(
            ["_rank_group", "precomputed_max_ca_rmsd", "protein_id"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop(columns="_rank_group")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(pd.read_parquet(args.catalog))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"wrote={args.output} rows={len(result)} "
        f"direct_pairs={(result.transition_priority == 'direct_pair').sum()}"
    )


if __name__ == "__main__":
    main()
