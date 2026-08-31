"""Persist predictions, metrics, and a lightweight run manifest."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_report(
    run_dir: str | Path,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path / "predictions.parquet", index=False)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
