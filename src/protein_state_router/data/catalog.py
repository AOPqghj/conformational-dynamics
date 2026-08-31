"""Catalog construction helpers."""

from pathlib import Path

import pandas as pd
from protein_state_router.data.schema import validate_catalog


def build_catalog(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Validate CSV or Parquet source records and persist a canonical Parquet catalog."""
    input_path = Path(input_path)
    frame = (
        pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    )
    catalog = validate_catalog(frame)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output_path, index=False)
    return catalog
