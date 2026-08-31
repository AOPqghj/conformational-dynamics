"""Evaluate immutable initial-2k classifier artifacts on an external catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.experiments.benchmark import (
    _predict,
    embedding_cnn_tensor,
    sequence_cnn_tensor,
    sequence_feature_matrix,
    single_embedding_feature_matrix,
)


def freeze_registry(models_root: str | Path, output: str | Path) -> dict[str, str]:
    """Checksum each model artifact once; evaluation aborts if any later changes."""
    root = Path(models_root)
    registry = {
        str(path.relative_to(root)): _sha256(path) for path in sorted(root.glob("*/model.joblib"))
    }
    if not registry:
        raise FileNotFoundError(f"no model.joblib files below {root}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def evaluate_catalog(
    catalog: pd.DataFrame,
    *,
    models_root: str | Path,
    registry_path: str | Path,
    embedding_manifest: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Score every compatible frozen model without fitting, tuning, or calibration."""
    _validate_catalog(catalog)
    root = Path(models_root)
    registry = json.loads(Path(registry_path).read_text())
    embedding = _read_embedding_manifest(embedding_manifest)
    cache: dict[str, np.ndarray] = {}
    rows: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float | str]] = {}
    for relative, expected in sorted(registry.items()):
        path = root / relative
        name = path.parent.name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen artifact checksum mismatch: {relative}")
        try:
            features = _features(name, catalog, embedding, cache)
        except ValueError as error:
            metrics[name] = {"status": "incompatible", "reason": str(error)}
            continue
        probabilities = _predict(joblib.load(path), features)
        output = catalog.loc[
            :, ["protein_id", "dataset_label", "source_dataset", "source_record_id"]
        ].copy()
        output["model_name"] = name
        output["probability"] = probabilities
        output["prediction"] = (probabilities >= 0.5).astype(int)
        rows.append(output)
        metrics[name] = _metrics(output.dataset_label.to_numpy(), probabilities)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), metrics


def _features(
    name: str,
    catalog: pd.DataFrame,
    embedding_manifest: pd.DataFrame | None,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    if name == "sequence_cnn":
        return sequence_cnn_tensor(catalog)[0]
    if name.startswith("sequence_") and not name.startswith("sequence_plus_esmfold_"):
        return sequence_feature_matrix(catalog)[0]
    if embedding_manifest is None:
        raise ValueError("embedding manifest is required")
    if "pooled" not in cache:
        root = embedding_manifest.attrs["bundle_root"]
        cache["pooled"] = single_embedding_feature_matrix(catalog, embedding_manifest, root)[0]
    pooled = cache["pooled"]
    if name == "esmfold_single_embedding_cnn":
        return embedding_cnn_tensor(pooled)[0]
    if name.startswith("esmfold_single_"):
        return pooled
    if name.startswith("sequence_plus_esmfold_"):
        return np.concatenate((sequence_feature_matrix(catalog)[0], pooled), axis=1)
    raise ValueError(f"unrecognized frozen model feature view: {name}")


def _read_embedding_manifest(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    location = Path(path)
    if not location.is_file():
        raise ValueError(f"embedding manifest does not exist: {location}")
    frame = pd.read_parquet(location) if location.suffix == ".parquet" else pd.read_csv(location)
    frame.attrs["bundle_root"] = location.parent
    return frame


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | str]:
    labels = np.asarray(labels, dtype=int)
    predicted = probabilities >= 0.5
    if len(np.unique(labels)) == 2:
        values: dict[str, float | str] = {
            "status": "complete",
            **classification_metrics(labels, probabilities),
        }
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        values["sensitivity"] = float(tp / (tp + fn)) if tp + fn else float("nan")
        values["specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")
        return values
    negative = labels == 0
    return {
        "status": "single_class_external_cohort",
        "sample_count": float(len(labels)),
        "specificity": float((~predicted[negative]).mean()) if negative.any() else float("nan"),
        "false_positive_rate": float(predicted[negative].mean())
        if negative.any()
        else float("nan"),
        "fraction_predicted_positive": float(predicted.mean()),
        "mean_probability": float(probabilities.mean()),
    }


def _validate_catalog(catalog: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "sequence",
        "sequence_length",
        "dataset_label",
        "source_dataset",
        "source_record_id",
    }
    if missing := required - set(catalog):
        raise ValueError(f"external catalog missing columns: {sorted(missing)}")
    if catalog.protein_id.duplicated().any():
        raise ValueError("external catalog protein IDs must be unique")
    if not set(catalog.dataset_label).issubset({0, 1}):
        raise ValueError("external catalog dataset_label must be binary")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
