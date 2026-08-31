"""Test whether ESMFold adds predictive value beyond length, sequence features, and pLDDT."""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from protein_state_router.evaluation.inference import benjamini_hochberg, paired_sign_flip_test
from protein_state_router.experiments.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    sequence_feature_matrix,
)
from protein_state_router.representations.registry import representation_choices
from train_suite import _load_features

ROOT = Path("data/lifecycle/final/initial_8598_dataset")
CATALOG_PATH = ROOT / "homology35_seed42/catalog.parquet"
SPLITS = Path("ml/results/homology35_repeated_splits")
OUTPUT = Path("ml/results/homology35_plddt_confounder_benchmark")
EMBEDDING_MANIFEST = ROOT / "embedding_manifest.csv"
SEEDS = tuple(range(10, 20))
REPRESENTATION_NAME = "esmfold"
EXPECTED_ROWS: int | None = 7032
FEATURE_VIEWS = ("covariates", "embedding", f"covariates_plus_{REPRESENTATION_NAME}")
CENTRAL = ZoneInfo("America/Chicago")


def _now() -> str:
    return datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M:%S %Z")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _write_report(records: list[dict[str, object]], completed: int) -> None:
    table = pd.DataFrame(records)
    html = (
        "<!doctype html><meta charset='utf-8'><title>pLDDT confounder benchmark</title>"
        "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;color:#17324d}"
        "table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d8e0e8;padding:7px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style>"
        "<h1>pLDDT confounder benchmark</h1>"
        "<p>Ten paired saved splits. Covariates are log sequence length, amino-acid fractions, "
        "sequence entropy, and AlphaFold DB mean pLDDT. Models use CPU-only validation selection.</p>"
        f"<p>Completed splits: {completed}/{len(SEEDS)}</p>"
        + table.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    )
    (OUTPUT / "live_report.html").write_text(html)


def _dataset_for_seed(frame: pd.DataFrame, seed: int) -> Path:
    split = pd.read_parquet(SPLITS / f"split_{seed}.parquet")[["protein_id", "split"]]
    dataset = frame.drop(columns="split", errors="ignore").merge(
        split, on="protein_id", how="inner"
    )
    if len(dataset) != len(frame) or set(dataset.split) != {"train", "val", "test"}:
        raise ValueError(f"saved split {seed} does not cover the pLDDT-observed subset")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"split_{seed}.parquet"
    dataset.to_parquet(path, index=False)
    return path


def _covariates(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    sequence, names = sequence_feature_matrix(frame)
    plddt = frame.alphafold_mean_plddt.to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate((sequence, plddt), axis=1), (*names, "alphafold_mean_plddt")


def _record(seed: int, family: str, view: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "seed": seed,
        "family": family,
        "feature_view": view,
        "test_accuracy": metrics["accuracy"],
        "test_auroc": metrics["auroc"],
        "test_auprc": metrics["auprc"],
    }


def _candidate_progress(seed: int, family: str, view: str, payload: dict[str, object]) -> None:
    event = {
        **payload,
        "seed": seed,
        "family": family,
        "feature_view": view,
    }
    print(json.dumps(event), flush=True)
    _write_json(OUTPUT / "candidate_progress.json", {**event, "updated_at_central": _now()})


def _paired_reports(results: pd.DataFrame) -> list[dict[str, object]]:
    """Return paired A, B, and A+B comparisons within each model family."""
    if not {"covariates", "embedding", f"covariates_plus_{REPRESENTATION_NAME}"}.issubset(
        set(results.feature_view)
    ):
        return []
    reports: list[dict[str, object]] = []
    metrics = ("test_accuracy", "test_auroc", "test_auprc")
    for family in ("linear", "tree"):
        paired = results.loc[results.family.eq(family)].pivot(
            index="seed",
            columns="feature_view",
            values=list(metrics),
        )
        for comparison, baseline_view, enriched_view in (
            ("embedding_vs_covariates", "covariates", "embedding"),
            ("combined_vs_embedding", "embedding", f"covariates_plus_{REPRESENTATION_NAME}"),
            ("combined_vs_covariates", "covariates", f"covariates_plus_{REPRESENTATION_NAME}"),
        ):
            report: dict[str, object] = {
                "comparison": comparison,
                "feature_view": "within_family",
                "family": family,
                "n_splits": len(paired),
            }
            for metric in metrics:
                name = metric.removeprefix("test_")
                baseline = paired[(metric, baseline_view)]
                enriched = paired[(metric, enriched_view)]
                test = paired_sign_flip_test((enriched - baseline).to_numpy())
                report[f"mean_{baseline_view}_{name}"] = baseline.mean()
                report[f"mean_{enriched_view}_{name}"] = enriched.mean()
                report[f"paired_{name}_p"] = test["permutation_p_two_sided"]
                report[f"paired_{name}_method"] = test["permutation_method"]
            reports.append(report)

    combined = results.loc[results.feature_view.eq(f"covariates_plus_{REPRESENTATION_NAME}")].pivot(
        index="seed", columns="family", values=list(metrics)
    )
    report = {
        "comparison": "tree_vs_linear",
        "feature_view": f"covariates_plus_{REPRESENTATION_NAME}",
        "family": "tree_vs_linear",
        "n_splits": len(combined),
    }
    for metric in metrics:
        name = metric.removeprefix("test_")
        linear = combined[(metric, "linear")]
        tree = combined[(metric, "tree")]
        test = paired_sign_flip_test((tree - linear).to_numpy())
        report[f"mean_linear_{name}"] = linear.mean()
        report[f"mean_tree_{name}"] = tree.mean()
        report[f"mean_tree_minus_linear_{name}"] = (tree - linear).mean()
        report[f"paired_{name}_p"] = test["permutation_p_two_sided"]
        report[f"paired_{name}_method"] = test["permutation_method"]
    reports.append(report)
    p_keys = sorted({key for report in reports for key in report if key.endswith("_p")})
    values = np.asarray([report.get(key, 1.0) for report in reports for key in p_keys])
    adjusted = benjamini_hochberg(values).reshape(len(reports), len(p_keys))
    for row_index, report in enumerate(reports):
        for column_index, key in enumerate(p_keys):
            report[key.removesuffix("_p") + "_fdr"] = float(adjusted[row_index, column_index])
    return reports


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(CATALOG_PATH)
    catalog = catalog.loc[catalog.alphafold_mean_plddt.notna()].copy()
    if (
        EXPECTED_ROWS is not None and len(catalog) != EXPECTED_ROWS
    ) or catalog.dataset_label.nunique() != 2:
        raise ValueError("expected the requested pLDDT-observed binary subset with both classes")
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        dataset = _dataset_for_seed(catalog, seed)
        frame = pd.read_parquet(dataset)
        covariates, covariate_names = _covariates(frame)
        embeddings, embedding_names = _load_features(
            dataset, f"{REPRESENTATION_NAME}_single", "linear", EMBEDDING_MANIFEST
        )
        all_views = {
            "covariates": (covariates, covariate_names),
            "embedding": (embeddings, embedding_names),
            f"covariates_plus_{REPRESENTATION_NAME}": (
                np.concatenate((covariates, embeddings), axis=1),
                (*covariate_names, *embedding_names),
            ),
        }
        views = {view: all_views[view] for view in FEATURE_VIEWS}
        for family in ("linear", "tree"):
            for view, (features, names) in views.items():
                model_dir = OUTPUT / f"seed_{seed}" / f"{family}_{view}"
                metrics_path = model_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                else:
                    metrics = run_benchmark(
                        dataset,
                        model_dir,
                        BenchmarkConfig(
                            family=family,
                            random_seed=seed,
                            device="cpu",
                            search=os.environ.get("PLDDT_BENCHMARK_SEARCH", "standard"),
                            save_model=False,
                        ),
                        features=features,
                        feature_names=names,
                        progress_callback=lambda payload, seed=seed, family=family, view=view: (
                            _candidate_progress(seed, family, view, payload)
                        ),
                    )
                records.append(_record(seed, family, view, metrics))
                print(
                    json.dumps(
                        {
                            "event": "confounder_model_complete",
                            "seed": seed,
                            "family": family,
                            "feature_view": view,
                            "completed_models": len(records),
                            "total_models": len(SEEDS) * len(views) * 2,
                        }
                    ),
                    flush=True,
                )
                pd.DataFrame(records).to_csv(OUTPUT / "live_metrics.csv", index=False)
                _write_json(
                    OUTPUT / "progress.json",
                    {
                        "status": "running",
                        "updated_at_central": _now(),
                        "completed_models": len(records),
                        "total_models": len(SEEDS) * len(views) * 2,
                        "completed_splits": seed - SEEDS[0] + 1,
                    },
                )
                _write_report(records, seed - SEEDS[0] + 1)
        del covariates, embeddings, features, frame, views
        gc.collect()
    results = pd.DataFrame(records)
    results.to_csv(OUTPUT / "all_metrics.csv", index=False)
    pd.DataFrame(_paired_reports(results)).to_csv(
        OUTPUT / "paired_permutation_tests.csv", index=False
    )
    _write_json(
        OUTPUT / "progress.json",
        {"status": "completed", "updated_at_central": _now(), "completed_models": len(records)},
    )
    _write_report(records, len(SEEDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "homology35_seed42/catalog.parquet")
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--embedding-manifest", type=Path, default=EMBEDDING_MANIFEST)
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    ROOT = args.catalog.parent.parent
    CATALOG_PATH = args.catalog
    SPLITS = args.splits
    OUTPUT = args.output
    EMBEDDING_MANIFEST = args.embedding_manifest
    SEEDS = tuple(args.seeds)
    REPRESENTATION_NAME = args.representation_name
    EXPECTED_ROWS = args.expected_rows
    main()
