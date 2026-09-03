# ruff: noqa: E402 - direct execution needs the repository root on sys.path.
"""Measure homology-aware linear recoverability of dataset provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.run_source_heldout_benchmark import origin_source
from ml.train_frozen_8598_models import pooled_features
from protein_state_router.evaluation.inference import benjamini_hochberg
from protein_state_router.experiments.benchmark import sequence_feature_matrix

REPRESENTATION_WIDTHS = {
    "esmfold": 1024,
    "bioemu": 384,
    "bioemu_no_msa": 384,
    "esm2_3b": 2560,
}
DEFAULT_CATALOGS = {
    "esmfold": Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    "bioemu": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/bioemu_8572_catalog.parquet"
    ),
    "bioemu_no_msa": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/"
        "bioemu_no_msa_8572_catalog.parquet"
    ),
    "esm2_3b": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/esm2_3b_8566_catalog.parquet"
    ),
}
DEFAULT_EMBEDDING_MANIFESTS = {
    "esmfold": Path("data/lifecycle/final/initial_8598_dataset/embedding_manifest.csv"),
    "bioemu": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/bioemu_8572_embedding_manifest.csv"
    ),
    "bioemu_no_msa": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/"
        "bioemu_no_msa_8572_embedding_manifest.csv"
    ),
    "esm2_3b": Path(
        "data/lifecycle/final/initial_8598_dataset/homology35_seed42/"
        "esm2_3b_8566_embedding_manifest.csv"
    ),
}
EXPECTED_SOURCES = {
    "all": ("atlas", "dynamicmpnn", "pathpre", "promise", "rcsb"),
    "dynamic_only": ("dynamicmpnn", "pathpre", "promise"),
    "static_only": ("atlas", "pathpre", "rcsb"),
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_cohorts(catalog: pd.DataFrame) -> dict[str, np.ndarray]:
    source = catalog.protein_id.map(origin_source)
    labels = catalog.dataset_label.to_numpy(dtype=int)
    cohorts = {
        "all": np.ones(len(catalog), dtype=bool),
        "dynamic_only": labels == 1,
        "static_only": labels == 0,
    }
    for name, mask in cohorts.items():
        actual = tuple(sorted(source.loc[mask].unique()))
        if actual != EXPECTED_SOURCES[name]:
            raise ValueError(
                f"{name} source classes are {actual}, expected {EXPECTED_SOURCES[name]}"
            )
    return cohorts


def metrics(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "macro_ovr_auroc": float(
            roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro")
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
    }


def permutation_tests(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    observed = metrics(labels, probabilities, classes)
    rng = np.random.default_rng(seed)
    selected = ("balanced_accuracy", "macro_f1", "macro_ovr_auroc")
    null = {name: np.empty(permutations, dtype=float) for name in selected}
    for index in range(permutations):
        shuffled = rng.permutation(labels)
        values = metrics(shuffled, probabilities, classes)
        for name in selected:
            null[name][index] = values[name]
    return {
        name: {
            "method": "fixed_oof_prediction_label_permutation",
            "observed": observed[name],
            "null_mean": float(values.mean()),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "permutation_p_greater": float(
                (1 + np.count_nonzero(values >= observed[name])) / (permutations + 1)
            ),
        }
        for name, values in null.items()
    }


def evaluate_view(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    cohort: str,
    feature_view: str,
    output: Path,
    folds: int,
    permutations: int,
    seed: int,
    cpu_threads: int,
) -> dict[str, Any]:
    labels = frame.origin_source.to_numpy(dtype=str)
    groups = frame.homology_group_id.astype(str).to_numpy()
    classes = np.asarray(sorted(np.unique(labels)))
    if tuple(classes) != EXPECTED_SOURCES[cohort]:
        raise ValueError(f"unexpected classes for {cohort}: {classes.tolist()}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full((len(frame), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=int)
    fold_records: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups)):
        if set(labels[train]) != set(classes) or set(labels[test]) != set(classes):
            raise ValueError(f"fold {fold} does not contain every source class")
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=seed + fold,
                    ),
                ),
            ]
        )
        with threadpool_limits(limits=cpu_threads):
            model.fit(features[train], labels[train])
            fold_probability = model.predict_proba(features[test])
        if not np.array_equal(model.classes_, classes):
            raise ValueError("classifier source class order changed")
        probabilities[test] = fold_probability
        fold_ids[test] = fold
        fold_records.append(
            {
                "cohort": cohort,
                "feature_view": feature_view,
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                **metrics(labels[test], fold_probability, classes),
            }
        )
    if not np.isfinite(probabilities).all() or (fold_ids < 0).any():
        raise AssertionError("out-of-fold source predictions are incomplete")
    predicted = classes[np.argmax(probabilities, axis=1)]
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame[
        ["protein_id", "dataset_label", "homology_group_id", "origin_source"]
    ].copy()
    prediction_frame["fold"] = fold_ids
    prediction_frame["predicted_source"] = predicted
    for index, name in enumerate(classes):
        prediction_frame[f"probability_{name}"] = probabilities[:, index]
    prediction_frame.to_parquet(output / "oof_predictions.parquet", index=False)
    pd.DataFrame(fold_records).to_csv(output / "fold_metrics.csv", index=False)
    raw_confusion = confusion_matrix(labels, predicted, labels=classes)
    normalized = confusion_matrix(labels, predicted, labels=classes, normalize="true")
    pd.DataFrame(raw_confusion, index=classes, columns=classes).to_csv(
        output / "confusion_counts.csv"
    )
    pd.DataFrame(normalized, index=classes, columns=classes).to_csv(
        output / "confusion_recall_normalized.csv"
    )
    overall = metrics(labels, probabilities, classes)
    majority = pd.Series(labels).value_counts().idxmax()
    baseline = np.repeat(majority, len(labels))
    report: dict[str, Any] = {
        "status": "complete",
        "cohort": cohort,
        "feature_view": feature_view,
        "rows": len(frame),
        "classes": classes.tolist(),
        "class_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
        "folds": folds,
        "model": {
            "type": "class_balanced_multinomial_logistic_regression",
            "C": 0.1,
            "preprocessing": "train_fold_standard_scaler",
        },
        "metrics": overall,
        "majority_baseline": {
            "source": majority,
            "accuracy": float(accuracy_score(labels, baseline)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, baseline)),
            "macro_f1": float(f1_score(labels, baseline, average="macro")),
        },
        "permutation_tests": permutation_tests(
            labels,
            probabilities,
            classes,
            permutations=permutations,
            seed=seed,
        ),
        "per_source_recall": {
            name: float(value)
            for name, value in zip(
                classes, recall_score(labels, predicted, labels=classes, average=None), strict=True
            )
        },
    }
    atomic_json(output / "summary.json", report)
    return report


def completed_view_report(
    output: Path,
    *,
    cohort: str,
    feature_view: str,
    rows: int,
    folds: int,
) -> dict[str, Any] | None:
    """Return a verified completed view report, if one already exists.

    Source-prediction views are self-contained.  Reusing a completed view makes a
    restarted run finalize safely after an interruption without refitting models.
    """
    path = output / "summary.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        report.get("status") != "complete"
        or report.get("cohort") != cohort
        or report.get("feature_view") != feature_view
        or report.get("rows") != rows
        or report.get("folds") != folds
    ):
        return None
    return report


def run(
    catalog_path: Path,
    embedding_manifest_path: Path,
    output: Path,
    *,
    representation: str = "esmfold",
    expected_rows: int | None = None,
    folds: int = 5,
    permutations: int = 1000,
    seed: int = 42,
    cpu_threads: int = 2,
) -> dict[str, Any]:
    if representation not in REPRESENTATION_WIDTHS:
        raise ValueError(f"unsupported representation: {representation}")
    catalog = pd.read_parquet(catalog_path).copy()
    required = {"protein_id", "dataset_label", "homology_group_id", "sequence"}
    if missing := required - set(catalog):
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    if expected_rows is not None and len(catalog) != expected_rows:
        raise ValueError(f"expected {expected_rows:,} catalog rows, found {len(catalog):,}")
    if catalog.protein_id.duplicated().any():
        raise ValueError("catalog contains duplicate protein IDs")
    manifest = pd.read_csv(embedding_manifest_path)
    if manifest.protein_id.duplicated().any() or set(manifest.protein_id) != set(
        catalog.protein_id
    ):
        raise ValueError("embedding manifest must exactly cover the source-prediction catalog")
    catalog = catalog.drop(columns="embedding_path", errors="ignore").merge(
        manifest[["protein_id", "embedding_path"]], on="protein_id", validate="one_to_one"
    )
    catalog["origin_source"] = catalog.protein_id.map(origin_source)
    masks = source_cohorts(catalog)
    sequence, sequence_names = sequence_feature_matrix(catalog)
    pooled = pooled_features(catalog, REPRESENTATION_WIDTHS[representation])
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    records: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []
    feature_views = {
        "sequence_covariates": (sequence, tuple(sequence_names)),
        f"pooled_{representation}": (
            pooled,
            tuple(f"pooled_{index}" for index in range(pooled.shape[1])),
        ),
    }
    for cohort, mask in masks.items():
        frame = catalog.loc[mask].reset_index(drop=True)
        for feature_view, (values, names) in feature_views.items():
            selected = values[mask]
            if selected.shape != (len(frame), len(names)):
                raise AssertionError("source-prediction feature matrix is misaligned")
            view_output = output / cohort / feature_view
            report = completed_view_report(
                view_output,
                cohort=cohort,
                feature_view=feature_view,
                rows=len(frame),
                folds=folds,
            )
            if report is None:
                report = evaluate_view(
                    frame,
                    selected,
                    cohort=cohort,
                    feature_view=feature_view,
                    output=view_output,
                    folds=folds,
                    permutations=permutations,
                    seed=seed,
                    cpu_threads=cpu_threads,
                )
            records.append(
                {
                    "cohort": cohort,
                    "feature_view": feature_view,
                    "rows": report["rows"],
                    **report["metrics"],
                    "majority_accuracy": report["majority_baseline"]["accuracy"],
                }
            )
            for metric_name, test in report["permutation_tests"].items():
                permutation_records.append(
                    {
                        "cohort": cohort,
                        "feature_view": feature_view,
                        "metric": metric_name,
                        **test,
                    }
                )
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "updated_at_utc": now(),
                    "completed_experiments": len(records),
                    "total_experiments": len(masks) * len(feature_views),
                    "active_cohort": cohort,
                    "active_feature_view": feature_view,
                },
            )
    pd.DataFrame(records).to_csv(output / "summary_metrics.csv", index=False)
    permutation_frame = pd.DataFrame(permutation_records)
    permutation_frame["fdr_bh"] = benjamini_hochberg(
        permutation_frame.permutation_p_greater.to_numpy(dtype=float)
    )
    permutation_frame.to_csv(output / "permutation_tests.csv", index=False)
    final = {
        "status": "complete",
        "updated_at_utc": now(),
        "catalog": str(catalog_path),
        "embedding_manifest": str(embedding_manifest_path),
        "representation": representation,
        "embedding_width": REPRESENTATION_WIDTHS[representation],
        "folds": folds,
        "permutations": permutations,
        "experiments": records,
        "interpretation": (
            "Source recoverability indicates provenance-specific signal. It does not establish "
            "that a dynamics classifier relies on that signal; the PathPre-only benchmark is the "
            "direct source-controlled dynamics evaluation."
        ),
    }
    atomic_json(progress_path, final)
    atomic_json(output / "summary.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=tuple(REPRESENTATION_WIDTHS), default="esmfold")
    parser.add_argument(
        "--catalog", type=Path
    )
    parser.add_argument(
        "--embedding-manifest", type=Path
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/results/pathpre_only_source_control/source_prediction"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    if args.catalog is None:
        args.catalog = DEFAULT_CATALOGS[args.representation]
    if args.embedding_manifest is None:
        args.embedding_manifest = DEFAULT_EMBEDDING_MANIFESTS[args.representation]
    if args.expected_rows is None:
        args.expected_rows = 8598 if args.representation == "esmfold" else 8572
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(args.cpu_threads)
    print(
        json.dumps(
            run(
                args.catalog,
                args.embedding_manifest,
                args.output,
                representation=args.representation,
                expected_rows=args.expected_rows,
                folds=args.folds,
                permutations=args.permutations,
                seed=args.seed,
                cpu_threads=args.cpu_threads,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
