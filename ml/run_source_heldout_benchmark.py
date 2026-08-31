"""Run homology-purged source-held-out tests for the frozen router dataset.

The outer test blocks are either one provenance source (classic leave-one-source-
out) or one dynamic and one static source (paired source stress tests). Model
selection uses only an internal, homology-grouped split of the remaining data.
One-class source blocks are reported with class-conditional threshold metrics;
AUROC and AUPRC are deliberately undefined for those blocks.
"""
# ruff: noqa: E402 - script execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.train_frozen_8598_models import pooled_features, representation_contract
from protein_state_router.experiments.benchmark import (
    BenchmarkConfig,
    _candidates,
    _fit_candidate,
    _predict,
    _select_candidate,
    sequence_feature_matrix,
)

DEFAULT_CATALOGS = {
    "esmfold": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
    ),
    "bioemu": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/"
        "bioemu_8572_catalog.parquet"
    ),
}
WIDTHS = {"esmfold": 1024, "bioemu": 384}
FIXED_CANDIDATES = {
    ("esmfold", "linear"): "logistic_l1_C0.1",
    ("esmfold", "tree"): "hist_gradient_leaf31_lr0.03",
    ("bioemu", "linear"): "logistic_l1_C0.1",
    ("bioemu", "tree"): "hist_gradient_leaf31_lr0.08",
}
SOURCE_LABELS = {
    "atlas": "atlas",
    "dynamicmpnn": "dynamicmpnn",
    "pathpre": "pathpre",
    "promise": "promise",
    "rcsb": "rcsb",
}


def origin_source(protein_id: str) -> str:
    """Derive immutable provenance from the canonical protein identifier."""
    prefix = str(protein_id).split(":", 1)[0].strip().lower()
    if prefix not in SOURCE_LABELS:
        raise ValueError(f"unknown protein_id provenance prefix: {protein_id!r}")
    return SOURCE_LABELS[prefix]


def source_blocks(frame: pd.DataFrame, suite: str = "all") -> dict[str, np.ndarray]:
    """Return the prespecified classic and paired outer-test masks."""
    source = frame["origin_source"] if "origin_source" in frame else frame.protein_id.map(origin_source)
    label = frame.dataset_label.to_numpy(dtype=int)
    classic = {
        "classic_dynamicmpnn": source.eq("dynamicmpnn").to_numpy(),
        "classic_promise": source.eq("promise").to_numpy(),
        "classic_rcsb": source.eq("rcsb").to_numpy(),
        "classic_atlas": source.eq("atlas").to_numpy(),
        "classic_pathpre_dynamic": (source.eq("pathpre") & (label == 1)).to_numpy(),
        "classic_pathpre_static": (source.eq("pathpre") & (label == 0)).to_numpy(),
        "classic_pathpre_all": source.eq("pathpre").to_numpy(),
    }
    paired = {
        f"paired_{positive}_{negative}": (
            source.eq(positive) | source.eq(negative)
        ).to_numpy()
        for positive in ("dynamicmpnn", "promise")
        for negative in ("rcsb", "atlas")
    }
    selected = classic if suite == "classic" else paired if suite == "paired" else {**classic, **paired}
    if any(not mask.any() for mask in selected.values()):
        raise ValueError("one or more source-held-out blocks are empty")
    return selected


def assign_outer_split(
    frame: pd.DataFrame, test_mask: np.ndarray, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Purge test homologs, then make a deterministic grouped 80/20 train/val split."""
    if len(test_mask) != len(frame) or not test_mask.any():
        raise ValueError("test mask must be aligned and non-empty")
    test_groups = set(frame.loc[test_mask, "homology_group_id"].astype(str))
    homolog_mask = frame.homology_group_id.astype(str).isin(test_groups).to_numpy()
    purged_mask = homolog_mask & ~test_mask
    development_mask = ~homolog_mask
    development = frame.loc[development_mask]
    if development.dataset_label.nunique() != 2:
        raise ValueError("homology purge leaves fewer than two development classes")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    labels = development.dataset_label.to_numpy(dtype=int)
    groups = development.homology_group_id.astype(str).to_numpy()
    train_index, val_index = next(splitter.split(np.zeros(len(development)), labels, groups))
    if np.unique(labels[train_index]).size != 2 or np.unique(labels[val_index]).size != 2:
        raise ValueError("internal train/validation split must contain both classes")
    assigned = frame.copy()
    assigned["split"] = "excluded"
    assigned.loc[frame.index[test_mask], "split"] = "test"
    assigned.loc[development.index[train_index], "split"] = "train"
    assigned.loc[development.index[val_index], "split"] = "val"
    if (
        assigned.loc[assigned.split.ne("excluded")]
        .groupby("homology_group_id")
        .split.nunique()
        .gt(1)
        .any()
    ):
        raise AssertionError("homology group crosses an assigned split")
    report = {
        "test_rows": int(test_mask.sum()),
        "test_groups": len(test_groups),
        "purged_homolog_rows": int(purged_mask.sum()),
        "purged_homolog_groups": int(frame.loc[purged_mask, "homology_group_id"].nunique()),
        "train_rows": int((assigned.split == "train").sum()),
        "validation_rows": int((assigned.split == "val").sum()),
        "development_class_counts": {
            str(key): int(value) for key, value in development.dataset_label.value_counts().items()
        },
    }
    return assigned, report


def evaluation_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    """Compute binary or honest one-class metrics at the fixed 0.5 threshold."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or len(labels) == 0 or probabilities.shape != labels.shape:
        raise ValueError("labels and probabilities must be aligned, non-empty vectors")
    predicted = probabilities >= 0.5
    positives, negatives = labels == 1, labels == 0
    metrics: dict[str, float | None] = {
        "sample_count": float(len(labels)),
        "positive_prevalence": float(labels.mean()),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(np.median(probabilities)),
        "predicted_positive_fraction": float(predicted.mean()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(predicted[positives].mean()) if positives.any() else None,
        "specificity": float((~predicted[negatives]).mean()) if negatives.any() else None,
        "false_negative_rate": float((~predicted[positives]).mean()) if positives.any() else None,
        "false_positive_rate": float(predicted[negatives].mean()) if negatives.any() else None,
    }
    if positives.any() and negatives.any():
        metrics.update(
            {
                "auroc": float(roc_auc_score(labels, probabilities)),
                "auprc": float(average_precision_score(labels, probabilities)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            }
        )
    else:
        metrics.update({"auroc": None, "auprc": None, "balanced_accuracy": None})
    return metrics


def bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, dict[str, float]]:
    """Return deterministic protein-level percentile intervals, stratified when binary."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    values: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(index, len(index), replace=True) for index in strata])
        metrics = evaluation_metrics(labels[sampled], probabilities[sampled])
        for key, value in metrics.items():
            if value is not None:
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
        }
        for key, samples in values.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_one(
    assigned: pd.DataFrame,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    family: str,
    representation: str,
    selection_protocol: str,
    seed: int,
    cpu_threads: int,
) -> tuple[Any, str, list[dict[str, float | str]], np.ndarray]:
    keep = assigned.split.ne("excluded").to_numpy()
    compact = assigned.loc[keep].reset_index(drop=True)
    compact_features = features[keep]
    labels = compact.dataset_label.to_numpy(dtype=int)
    train = compact.split.eq("train").to_numpy()
    validation = compact.split.eq("val").to_numpy()
    test = compact.split.eq("test").to_numpy()
    config = BenchmarkConfig(family=family, random_seed=seed, cpu_threads=cpu_threads)
    # Imputation belongs inside every candidate so pLDDT medians are learned from
    # training only. This wrapper is a no-op for complete embedding-only views.
    original_candidates = _candidates(config)
    wrapped = [
        (name, Pipeline([("impute", SimpleImputer()), ("candidate", candidate)]))
        for name, candidate in original_candidates
    ]
    if selection_protocol == "fixed":
        selected_name = FIXED_CANDIDATES[(representation, family)]
        selected = dict(wrapped)[selected_name]
        trials: list[dict[str, float | str]] = [
            {
                "candidate": selected_name,
                "selection": "locked_before_source_heldout_testing",
            }
        ]
    else:
        import protein_state_router.experiments.benchmark as benchmark_module

        original_factory = benchmark_module._candidates
        benchmark_module._candidates = lambda _config, progress_callback=None: wrapped
        try:
            selected_name, selected, trials = _select_candidate(
                config, compact_features, labels, train, validation
            )
        finally:
            benchmark_module._candidates = original_factory
    fitted = _fit_candidate(selected, compact_features[train | validation], labels[train | validation], config)
    probabilities = _predict(fitted, compact_features[test])
    if len(feature_names) != features.shape[1]:
        raise AssertionError("feature names do not match feature matrix")
    return fitted, selected_name, trials, probabilities


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _feature_views(
    frame: pd.DataFrame, representation: str, embeddings: np.ndarray
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    sequence_covariates, sequence_names = sequence_feature_matrix(frame)
    plddt = frame.alphafold_mean_plddt.to_numpy(dtype=np.float32).reshape(-1, 1)
    covariates = np.concatenate((sequence_covariates, plddt), axis=1)
    covariate_names = (*sequence_names, "alphafold_mean_plddt")
    _, embedding_names = representation_contract(representation, WIDTHS[representation])
    return {
        "covariates": (covariates, covariate_names),
        "embedding": (embeddings, embedding_names),
        "combined": (
            np.concatenate((covariates, embeddings), axis=1),
            (*covariate_names, *embedding_names),
        ),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the requested resumable source-held-out suite."""
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for representation in args.representations:
        catalog_path = Path(args.catalogs.get(representation, DEFAULT_CATALOGS[representation]))
        frame = pd.read_parquet(catalog_path).reset_index(drop=True)
        required = {"protein_id", "dataset_label", "homology_group_id", "embedding_path"}
        if missing := required - set(frame):
            raise ValueError(f"catalog lacks required columns: {sorted(missing)}")
        frame["origin_source"] = frame.protein_id.map(origin_source)
        cache = args.output / "feature_cache" / f"{representation}_pooled.npz"
        if cache.is_file():
            payload = np.load(cache)
            embeddings = payload["features"]
            if embeddings.shape[0] != len(frame):
                raise ValueError(f"stale pooled feature cache: {cache}")
        else:
            embeddings = pooled_features(frame, WIDTHS[representation])
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, features=embeddings)
        views = _feature_views(frame, representation, embeddings)
        blocks = source_blocks(frame, args.suite)
        for block_index, (block, test_mask) in enumerate(blocks.items()):
            assigned, split_report = assign_outer_split(frame, test_mask, args.seed)
            block_root = args.output / representation / block
            block_root.mkdir(parents=True, exist_ok=True)
            manifest_columns = [
                "protein_id", "dataset_label", "origin_source", "homology_group_id", "split"
            ]
            assigned[manifest_columns].to_parquet(block_root / "split_manifest.parquet", index=False)
            assigned.loc[assigned.split.eq("excluded"), manifest_columns].to_csv(
                block_root / "purged_homologs.csv", index=False
            )
            _write_json(block_root / "split_report.json", split_report)
            for family in args.families:
                for view_name in args.feature_views:
                    model_root = block_root / f"{family}_{view_name}"
                    metrics_path = model_root / "metrics.json"
                    predictions_path = model_root / "test_predictions.parquet"
                    if metrics_path.is_file() and predictions_path.is_file() and not args.replace:
                        metrics = json.loads(metrics_path.read_text())
                    else:
                        began = time.monotonic()
                        fitted, selected, trials, probabilities = _fit_one(
                            assigned,
                            *views[view_name],
                            family=family,
                            representation=representation,
                            selection_protocol=args.selection_protocol,
                            seed=args.seed,
                            cpu_threads=args.cpu_threads,
                        )
                        test = assigned.split.eq("test")
                        labels = assigned.loc[test, "dataset_label"].to_numpy(dtype=int)
                        metrics = evaluation_metrics(labels, probabilities)
                        intervals = bootstrap_intervals(
                            labels,
                            probabilities,
                            seed=args.seed + block_index,
                            replicates=args.bootstrap_replicates,
                        )
                        predictions = assigned.loc[
                            test, ["protein_id", "dataset_label", "origin_source", "homology_group_id"]
                        ].copy()
                        predictions["probability"] = probabilities
                        model_root.mkdir(parents=True, exist_ok=True)
                        predictions.to_parquet(predictions_path, index=False)
                        _write_json(metrics_path, metrics)
                        _write_json(model_root / "bootstrap_95ci.json", intervals)
                        _write_json(
                            model_root / "validation_selection.json",
                            {
                                "primary_metric": "auroc" if args.selection_protocol == "nested" else None,
                                "selection_protocol": args.selection_protocol,
                                "selected_candidate": selected,
                                "trials": trials,
                            },
                        )
                        _write_json(
                            model_root / "manifest.json",
                            {
                                "representation": representation,
                                "feature_view": view_name,
                                "family": family,
                                "seed": args.seed,
                                "cpu_threads": args.cpu_threads,
                                "catalog": str(catalog_path),
                                "catalog_sha256": _sha256(catalog_path),
                                "homology_purge": True,
                                "selection_protocol": args.selection_protocol,
                                "split": split_report,
                                "elapsed_seconds": time.monotonic() - began,
                            },
                        )
                        if args.save_models:
                            joblib.dump(fitted, model_root / "model.joblib")
                        del fitted
                        gc.collect()
                    record = {
                        "representation": representation,
                        "block": block,
                        "block_kind": block.split("_", 1)[0],
                        "family": family,
                        "feature_view": view_name,
                        **metrics,
                        **{f"split_{key}": value for key, value in split_report.items() if isinstance(value, (int, float))},
                    }
                    records.append(record)
                    pd.DataFrame(records).to_csv(args.output / "summary_metrics.csv", index=False)
                    _write_json(
                        args.output / "progress.json",
                        {
                            "status": "running",
                            "completed_models": len(records),
                            "total_models": (
                                len(args.representations)
                                * len(source_blocks(frame, args.suite))
                                * len(args.families)
                                * len(args.feature_views)
                            ),
                            "representations": args.representations,
                            "suite": args.suite,
                            "last_completed": record,
                        },
                    )
                    print(json.dumps({"event": "model_complete", **record}), flush=True)
        del embeddings, views
        gc.collect()
    result = pd.DataFrame(records)
    result.to_csv(args.output / "summary_metrics.csv", index=False)
    _write_json(
        args.output / "progress.json",
        {"status": "completed", "completed_models": len(result), "suite": args.suite},
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ml/results/source_heldout"))
    parser.add_argument("--representations", nargs="+", choices=tuple(WIDTHS), default=list(WIDTHS))
    parser.add_argument("--suite", choices=("classic", "paired", "all"), default="all")
    parser.add_argument("--families", nargs="+", choices=("linear", "tree"), default=["linear", "tree"])
    parser.add_argument(
        "--feature-views", nargs="+", choices=("covariates", "embedding", "combined"),
        default=["covariates", "embedding", "combined"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--selection-protocol",
        choices=("fixed", "nested"),
        default="fixed",
        help="Use predeclared frozen candidates (default) or retune within every development set.",
    )
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--esmfold-catalog", type=Path, default=DEFAULT_CATALOGS["esmfold"]
    )
    parser.add_argument("--bioemu-catalog", type=Path, default=DEFAULT_CATALOGS["bioemu"])
    args = parser.parse_args()
    args.catalogs = {"esmfold": args.esmfold_catalog, "bioemu": args.bioemu_catalog}
    if args.cpu_threads < 1 or args.bootstrap_replicates < 1:
        parser.error("cpu threads and bootstrap replicates must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        result = run(args)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
