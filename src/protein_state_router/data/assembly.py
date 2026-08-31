"""Assemble auditable positive/negative router datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from protein_state_router.data.schema import validate_catalog


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Unsupported catalog format: {path}")


def prepare_catalog(path: str | Path, label: int) -> pd.DataFrame:
    """Read and validate one catalog, assigning its binary dataset label."""
    frame = _read(path).copy()
    if frame.empty:
        raise ValueError(f"Catalog is empty: {path}")
    frame["dataset_label"] = int(label)
    frame["dataset_label_name"] = "positive" if label else "negative"
    frame["dataset_partition"] = "positive_examples" if label else "negative_examples"
    if "protein_id" not in frame:
        raise ValueError(f"Catalog missing protein_id: {path}")
    return validate_catalog(frame)


def assemble_dataset(
    positive_path: str | Path,
    negative_path: str | Path,
    *,
    require_both_classes: bool = True,
) -> pd.DataFrame:
    """Combine canonical catalogs and enforce training-readiness invariants."""
    positive = prepare_catalog(positive_path, 1)
    negative = prepare_catalog(negative_path, 0)
    overlap = set(positive.protein_id) & set(negative.protein_id)
    if overlap:
        raise ValueError(f"Protein IDs occur in both classes: {sorted(overlap)[:5]}")
    frame = pd.concat([positive, negative], ignore_index=True, sort=False)
    frame["dataset_row_id"] = (
        frame["dataset_partition"].astype(str) + ":" + frame["protein_id"].astype(str)
    )
    if frame["dataset_row_id"].duplicated().any():
        raise ValueError("Duplicate dataset_row_id values")
    if require_both_classes and frame["dataset_label"].nunique() < 2:
        raise ValueError("Dataset must contain both positive and negative examples")
    return frame


def write_dataset(
    frame: pd.DataFrame, output_dir: str | Path, stem: str = "router_dataset_v0"
) -> None:
    """Write parquet plus human-readable CSV/XLSX review exports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / f"{stem}.parquet", index=False)
    frame.to_csv(output_dir / f"{stem}_review.csv", index=False)
    try:
        frame.to_excel(output_dir / f"{stem}_review.xlsx", index=False)
    except ImportError as error:
        raise RuntimeError("Excel output requires the dataset-export extra (openpyxl)") from error
