# ruff: noqa: E402 - executable script needs repository root before local imports.

"""Build the available-coverage BioEmu catalog while retaining frozen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interpretability.contracts import load_residue_matrix

EMBEDDING_WIDTH = 384


def build_catalog(
    source_catalog: Path,
    embedding_root: Path,
    output_catalog: Path,
    report_path: Path,
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Filter to valid flat embeddings without changing source split assignments."""
    catalog = pd.read_parquet(source_catalog).copy()
    required = {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
        "dataset_label",
        "split",
        "homology_group_id",
    }
    missing = required - set(catalog)
    if (
        missing
        or catalog.protein_id.duplicated().any()
        or catalog.sequence_sha256.duplicated().any()
    ):
        raise ValueError(f"source catalog lacks unique required columns: {sorted(missing)}")
    paths = catalog.sequence_sha256.map(lambda digest: embedding_root / f"{digest}.npz")
    selected = catalog.loc[paths.map(Path.is_file)].copy()
    selected["embedding_path"] = selected.sequence_sha256.map(
        lambda digest: str(embedding_root / f"{digest}.npz")
    )
    if len(selected) != expected_rows:
        raise ValueError(f"BioEmu coverage is {len(selected)}, expected {expected_rows}")
    if set(selected.split) != {"train", "val", "test"}:
        raise ValueError("selected catalog lacks a frozen split")
    if (selected.groupby("homology_group_id").split.nunique() > 1).any():
        raise ValueError("selected catalog leaks a homology group across splits")
    for split in ("train", "val", "test"):
        if selected.loc[selected.split.eq(split), "dataset_label"].nunique() != 2:
            raise ValueError(f"{split} does not contain both classes")
    for row in selected.itertuples(index=False):
        load_residue_matrix(
            row.embedding_path,
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=EMBEDDING_WIDTH,
        )
    selected = selected.sort_values("protein_id").reset_index(drop=True)
    assignments = selected[["protein_id", "split"]].to_csv(index=False).encode()
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_catalog, index=False)
    report = {
        "source_catalog": str(source_catalog),
        "source_catalog_sha256": hashlib.sha256(source_catalog.read_bytes()).hexdigest(),
        "output_catalog": str(output_catalog),
        "rows": len(selected),
        "embedding_width": EMBEDDING_WIDTH,
        "split_counts": selected.split.value_counts().sort_index().to_dict(),
        "split_class_counts": {
            f"{split}:{label}": int(count)
            for (split, label), count in selected.groupby(["split", "dataset_label"]).size().items()
        },
        "split_sha256": hashlib.sha256(assignments).hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=7875)
    args = parser.parse_args()
    print(
        json.dumps(
            build_catalog(
                args.source_catalog,
                args.embedding_root,
                args.output_catalog,
                args.report_path,
                expected_rows=args.expected_rows,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
