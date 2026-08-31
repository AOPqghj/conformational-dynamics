"""Train the seed-42 pooled-ESMFold frozen linear and tree references."""
# ruff: noqa: E402 - script execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interpretability.contracts import load_residue_matrix, pool_residue_matrix
from protein_state_router.experiments.benchmark import BenchmarkConfig, run_benchmark
from protein_state_router.representations.registry import (
    representation_choices,
    representation_spec,
)
from scripts.datasets.make_router_dataset_splits import make_splits

MODEL_NAMES = ("esmfold_single_linear", "esmfold_single_tree")
FEATURE_NAMES = tuple(
    f"esmfold_single_{stat}_{index}" for stat in ("mean", "std", "max") for index in range(1024)
)


def representation_contract(name: str, width: int) -> tuple[tuple[str, str], tuple[str, ...]]:
    """Return isolated model names and pooled feature names for one representation."""
    representation_spec(name, width)
    models = (f"{name}_single_linear", f"{name}_single_tree")
    features = tuple(
        f"{name}_single_{stat}_{index}" for stat in ("mean", "std", "max") for index in range(width)
    )
    return models, features


def sha256(path: Path) -> str:
    """Return a stable checksum without loading a model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status(event: str, **details: object) -> None:
    """Emit one compact, line-buffered terminal event for long local runs."""
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def prepare_seed_split(catalog: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a deterministic external split without altering the frozen catalog."""
    assignments, report = make_splits(catalog, seed, group_column="homology_group_id")
    dataset = catalog.drop(columns="split", errors="ignore").merge(
        assignments[["protein_id", "split"]], on="protein_id", validate="one_to_one"
    )
    if len(dataset) != len(catalog) or dataset.split.isna().any():
        raise ValueError("seed split does not cover the canonical catalog")
    split_bytes = assignments.sort_values("protein_id").to_csv(index=False).encode()
    return dataset, {
        **report,
        "seed": seed,
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }


def prepare_catalog_split(
    catalog: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Use an already-frozen split after validating its leakage-safety contract."""
    required = {"protein_id", "split", "dataset_label", "homology_group_id"}
    missing = required - set(catalog)
    if missing:
        raise ValueError(f"catalog split preservation requires columns: {sorted(missing)}")
    dataset = catalog.copy()
    if set(dataset.split) != {"train", "val", "test"}:
        raise ValueError("preserved catalog must contain train, val, and test splits")
    if dataset.split.isna().any() or not set(dataset.dataset_label).issubset({0, 1}):
        raise ValueError("preserved catalog has invalid split or binary labels")
    if (dataset.groupby("homology_group_id").split.nunique() > 1).any():
        raise ValueError("preserved catalog leaks homology groups across splits")
    for split in ("train", "val", "test"):
        if dataset.loc[dataset.split.eq(split), "dataset_label"].nunique() != 2:
            raise ValueError(f"preserved catalog {split} split lacks a class")
    assignments = dataset[["protein_id", "split"]].sort_values("protein_id")
    split_bytes = assignments.to_csv(index=False).encode()
    return dataset, {
        "seed": seed,
        "split_source": "preserved_catalog",
        "split_counts": dataset.split.value_counts().sort_index().to_dict(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }


def pooled_features(catalog: pd.DataFrame, width: int = 1024) -> np.ndarray:
    """Load canonical matrices and reproduce the frozen feature order."""
    rows: list[np.ndarray] = []
    total = len(catalog)
    for index, row in enumerate(catalog.itertuples(index=False), start=1):
        values = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=width,
        )
        rows.append(pool_residue_matrix(values))
        if index == 1 or index % 100 == 0 or index == total:
            status("pooling_embeddings", completed=index, total=total)
    features = np.stack(rows).astype(np.float32)
    if features.shape != (len(catalog), width * 3):
        raise ValueError(f"unexpected pooled feature shape: {features.shape}")
    return features


def export_linear_weights(model_path: Path, feature_names: tuple[str, ...] = FEATURE_NAMES) -> None:
    """Save auditable linear parameters alongside the trusted joblib pipeline."""
    model = joblib.load(model_path)
    scaler, classifier = model.named_steps["scale"], model.named_steps["model"]
    if classifier.coef_.shape != (1, len(feature_names)):
        raise ValueError("frozen linear model does not have the pooled feature contract")
    standardized = classifier.coef_[0].astype(np.float64)
    raw = standardized / scaler.scale_
    intercept = float(classifier.intercept_[0] - np.dot(raw, scaler.mean_))
    np.savez_compressed(
        model_path.with_name("linear_weights.npz"),
        feature_names=np.asarray(feature_names),
        standardized_coefficients=standardized,
        raw_coefficients=raw,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        raw_intercept=np.asarray(intercept),
    )


def export_tree_importances(
    model_path: Path, feature_names: tuple[str, ...] = FEATURE_NAMES
) -> None:
    """Write the tree's descriptive importances without claiming they are weights."""
    model = joblib.load(model_path)
    classifier = model.named_steps["model"] if hasattr(model, "named_steps") else model
    values = getattr(classifier, "feature_importances_", None)
    if values is None:
        return
    pd.DataFrame({"feature_name": feature_names, "importance": values}).to_parquet(
        model_path.with_name("tree_feature_importances.parquet"), index=False
    )


def train(
    catalog_path: Path,
    output_root: Path,
    *,
    seed: int = 42,
    cpu_threads: int = 2,
    replace: bool = False,
    representation_name: str = "esmfold",
    embedding_width: int = 1024,
    preserve_catalog_split: bool = False,
) -> dict[str, object]:
    """Train into a staging directory, then replace the frozen reference root."""
    catalog = pd.read_parquet(catalog_path)
    if catalog.empty or catalog.protein_id.duplicated().any():
        raise ValueError("expected a non-empty unique protein catalog")
    status("loaded_catalog", rows=len(catalog), seed=seed)
    dataset, split_report = (
        prepare_catalog_split(catalog, seed)
        if preserve_catalog_split
        else prepare_seed_split(catalog, seed)
    )
    status("created_split", **split_report)
    model_names, feature_names = representation_contract(representation_name, embedding_width)
    features = pooled_features(dataset, embedding_width)
    status("pooled_features_ready", shape=list(features.shape))
    if output_root.exists() and any(output_root.iterdir()) and not replace:
        raise FileExistsError("frozen model root exists; pass --replace after reviewing it")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frozen-8598-", dir=output_root.parent) as temporary:
        staging = Path(temporary) / "frozen_models"
        staging.mkdir()
        dataset_path = staging / f"seed_{seed}_catalog.parquet"
        split_path = staging / f"seed_{seed}_split.parquet"
        dataset.to_parquet(dataset_path, index=False)
        dataset[["protein_id", "split"]].to_parquet(split_path, index=False)
        (staging / f"seed_{seed}_split.json").write_text(
            json.dumps(split_report, indent=2, sort_keys=True) + "\n"
        )
        metrics: dict[str, object] = {}
        for family, name in (("linear", model_names[0]), ("tree", model_names[1])):
            status("training_started", family=family, model=name, search="standard")
            model_dir = staging / name
            metrics[name] = run_benchmark(
                dataset_path,
                model_dir,
                BenchmarkConfig(
                    family=family, random_seed=seed, search="standard", cpu_threads=cpu_threads
                ),
                features=features,
                feature_names=feature_names,
                dataset_reference=f"../seed_{seed}_catalog.parquet",
            )
            status("training_completed", family=family, model=name, metrics=metrics[name])
        status("exporting_artifacts")
        export_linear_weights(staging / model_names[0] / "model.joblib", feature_names)
        export_tree_importances(staging / model_names[1] / "model.joblib", feature_names)
        registry_paths = [
            dataset_path,
            split_path,
            staging / f"seed_{seed}_split.json",
            *[
                staging / name / artifact
                for name in model_names
                for artifact in (
                    "model.joblib",
                    "manifest.json",
                    "metrics.json",
                    "validation_selection.json",
                    "test_predictions.parquet",
                )
            ],
            staging / model_names[0] / "linear_weights.npz",
            staging / model_names[1] / "tree_feature_importances.parquet",
        ]
        registry = {
            str(path.relative_to(staging)): sha256(path)
            for path in registry_paths
            if path.is_file()
        }
        (staging / "frozen_model_registry.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n"
        )
        (staging / "README.md").write_text(
            f"# Homology-grouped pooled {representation_name} models\n\n"
            "Seed-42 MMseqs2 homology-grouped split. Models were selected on validation AUROC, refit on train plus validation, and evaluated once on the held-out test split.\n"
        )
        replacement = output_root.with_name(f".{output_root.name}.replacement-backup")
        if replacement.exists():
            raise FileExistsError(f"stale replacement backup requires review: {replacement}")
        had_previous = output_root.exists()
        if had_previous:
            output_root.replace(replacement)
        try:
            staging.replace(output_root)
        except Exception:
            if had_previous and not output_root.exists():
                replacement.replace(output_root)
            raise
        if had_previous:
            shutil.rmtree(replacement)
        status("atomic_replacement_completed", output_root=str(output_root))
    return {
        "seed": seed,
        "split": split_report,
        "metrics": metrics,
        "output_root": str(output_root),
        "representation_name": representation_name,
        "embedding_width": embedding_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("ml/results/homology35_frozen_models")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    parser.add_argument("--embedding-width", type=int, default=1024)
    parser.add_argument(
        "--preserve-catalog-split",
        action="store_true",
        help="validate and retain the catalog's existing frozen split instead of recalculating it",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.catalog,
                args.output_root,
                seed=args.seed,
                cpu_threads=args.cpu_threads,
                replace=args.replace,
                representation_name=args.representation_name,
                embedding_width=args.embedding_width,
                preserve_catalog_split=args.preserve_catalog_split,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
