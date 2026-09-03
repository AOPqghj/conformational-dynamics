"""Compare mean pLDDT with per-residue pLDDT distribution summaries."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = Path(__file__).resolve().parent
for path in (REPOSITORY_ROOT, ML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_plddt_only_benchmark as mean_benchmark  # noqa: E402
from protein_state_router.evaluation.inference import (  # noqa: E402
    benjamini_hochberg,
    paired_sign_flip_test,
)
from protein_state_router.experiments.benchmark import (  # noqa: E402
    BenchmarkConfig,
    run_benchmark,
)

DEFAULT_FEATURES = (
    REPOSITORY_ROOT
    / "data/lifecycle/final/initial_8598_dataset/analysis/alphafold_plddt_distribution_features.parquet"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ml/results/homology35_plddt_distribution_baseline"
MEAN_ONLY = ("plddt_mean",)
DISTRIBUTION_FEATURES = (
    "plddt_mean",
    "plddt_std",
    "plddt_q10",
    "plddt_q25",
    "plddt_median",
    "plddt_q75",
    "plddt_q90",
    "plddt_fraction_below_50",
    "plddt_fraction_below_70",
    "plddt_fraction_below_90",
)
VIEWS = {"mean_only": MEAN_ONLY, "distribution": DISTRIBUTION_FEATURES}


def _load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"protein_id", *DISTRIBUTION_FEATURES}
    if missing := required - set(frame):
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != mean_benchmark.EXPECTED_ROWS or frame.protein_id.duplicated().any():
        raise ValueError("pLDDT feature table must contain 7,032 unique proteins")
    if not np.isfinite(frame[list(DISTRIBUTION_FEATURES)].to_numpy(float)).all():
        raise ValueError("pLDDT distribution features contain non-finite values")
    return frame.set_index("protein_id")


def _feature_matrix(
    dataset: pd.DataFrame, feature_table: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    if not set(dataset.protein_id).issubset(feature_table.index):
        raise ValueError("pLDDT feature table does not cover the split dataset")
    matrix = feature_table.loc[dataset.protein_id, list(columns)].to_numpy(dtype=np.float32)
    if matrix.shape != (len(dataset), len(columns)) or not np.isfinite(matrix).all():
        raise ValueError("pLDDT feature matrix is not aligned to the split dataset")
    return matrix


def _paired_view_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repeated = metrics.loc[metrics.evaluation.eq("repeated")]
    for family in mean_benchmark.FAMILIES:
        values = repeated.loc[repeated.family.eq(family)].pivot(
            index="seed",
            columns="feature_view",
            values=[f"test_{m}" for m in mean_benchmark.METRICS],
        )
        row: dict[str, object] = {"family": family, "n_splits": len(values)}
        for metric in mean_benchmark.METRICS:
            baseline = values[(f"test_{metric}", "mean_only")]
            distribution = values[(f"test_{metric}", "distribution")]
            difference = distribution.to_numpy() - baseline.to_numpy()
            test = paired_sign_flip_test(difference)
            row[f"mean_only_{metric}"] = float(baseline.mean())
            row[f"distribution_{metric}"] = float(distribution.mean())
            row[f"mean_difference_{metric}"] = float(difference.mean())
            row[f"paired_{metric}_p"] = test["permutation_p_two_sided"]
            row[f"paired_{metric}_method"] = test["permutation_method"]
        rows.append(row)
    table = pd.DataFrame(rows)
    p_columns = [column for column in table if column.endswith("_p")]
    adjusted = benjamini_hochberg(table[p_columns].to_numpy(float).ravel()).reshape(
        len(table), len(p_columns)
    )
    for index, column in enumerate(p_columns):
        table[column.removesuffix("_p") + "_fdr"] = adjusted[:, index]
    return table


def _reference_comparisons(metrics: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    observed = metrics.loc[
        metrics.evaluation.eq("repeated") & metrics.feature_view.eq("distribution")
    ]
    rows = []
    for family in mean_benchmark.FAMILIES:
        distribution = observed.loc[observed.family.eq(family)].set_index("seed")
        for reference_view in ("covariates", "embedding"):
            comparator = reference.loc[
                reference.family.eq(family) & reference.feature_view.eq(reference_view)
            ].set_index("seed")
            if set(distribution.index) != set(comparator.index):
                raise ValueError(f"{family} {reference_view} does not use the same split seeds")
            row: dict[str, object] = {
                "family": family,
                "comparison": f"plddt_distribution_vs_{reference_view}",
                "n_splits": len(distribution),
            }
            for metric in mean_benchmark.METRICS:
                observed_values = distribution[f"test_{metric}"].sort_index()
                reference_values = comparator[f"test_{metric}"].sort_index()
                difference = observed_values.to_numpy() - reference_values.to_numpy()
                test = paired_sign_flip_test(difference)
                row[f"distribution_{metric}"] = float(observed_values.mean())
                row[f"{reference_view}_{metric}"] = float(reference_values.mean())
                row[f"mean_difference_{metric}"] = float(difference.mean())
                row[f"paired_{metric}_p"] = test["permutation_p_two_sided"]
            rows.append(row)
    table = pd.DataFrame(rows)
    p_columns = [column for column in table if column.endswith("_p")]
    adjusted = benjamini_hochberg(table[p_columns].to_numpy(float).ravel()).reshape(
        len(table), len(p_columns)
    )
    for index, column in enumerate(p_columns):
        table[column.removesuffix("_p") + "_fdr"] = adjusted[:, index]
    return table


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for feature_view, group in metrics.groupby("feature_view", sort=False):
        summary = mean_benchmark._summarize(group)
        summary.insert(2, "feature_view", feature_view)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("this benchmark requires exactly ten unique repeated-split seeds")
    feature_table = _load_features(args.features.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeated = mean_benchmark._load_repeated_splits(args.split_root.resolve(), tuple(args.seeds))
    frozen = mean_benchmark._frozen_dataset(args.catalog.resolve(), output)
    jobs = [
        ("frozen_seed42", 42, frozen),
        *[("repeated", seed, repeated[seed]) for seed in args.seeds],
    ]
    records: list[dict[str, object]] = []
    total = len(jobs) * len(mean_benchmark.FAMILIES) * len(VIEWS)
    for evaluation, seed, dataset_path in jobs:
        dataset = pd.read_parquet(dataset_path)
        for family in mean_benchmark.FAMILIES:
            for view, columns in VIEWS.items():
                model_output = output / evaluation / f"seed_{seed}" / family / view
                metrics_path = model_output / "metrics.json"
                if metrics_path.is_file():
                    result = json.loads(metrics_path.read_text())
                    event = "model_reused"
                else:
                    result = run_benchmark(
                        dataset_path,
                        model_output,
                        BenchmarkConfig(
                            family=family,
                            random_seed=seed,
                            primary_metric="auroc",
                            search=args.search,
                            save_model=False,
                            cpu_threads=args.cpu_threads,
                        ),
                        features=_feature_matrix(dataset, feature_table, columns),
                        feature_names=columns,
                        dataset_reference=str(dataset_path.relative_to(REPOSITORY_ROOT)),
                    )
                    event = "model_completed"
                records.append(
                    {
                        "evaluation": evaluation,
                        "seed": seed,
                        "family": family,
                        "feature_view": view,
                        "test_n": int(result["sample_count"]),
                        "test_accuracy": result["accuracy"],
                        "test_auroc": result["auroc"],
                        "test_auprc": result["auprc"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": event,
                            "evaluation": evaluation,
                            "seed": seed,
                            "family": family,
                            "feature_view": view,
                            "test_auroc": result["auroc"],
                            "completed_models": len(records),
                            "total_models": total,
                        }
                    ),
                    flush=True,
                )
                pd.DataFrame(records).to_csv(output / "live_metrics.csv", index=False)
                mean_benchmark._write_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "updated_at_utc": datetime.now(UTC).isoformat(),
                        "completed_models": len(records),
                        "total_models": total,
                    },
                )
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    _summarize(metrics).to_csv(output / "summary.csv", index=False)
    _paired_view_comparisons(metrics).to_csv(output / "paired_view_comparisons.csv", index=False)
    _reference_comparisons(metrics, args.reference.resolve()).to_csv(
        output / "paired_reference_comparisons.csv", index=False
    )
    mean_benchmark._write_json(
        output / "progress.json",
        {
            "status": "completed",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "completed_models": len(records),
            "total_models": total,
            "repeated_split_seeds": args.seeds,
            "distribution_features": list(DISTRIBUTION_FEATURES),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=mean_benchmark.DEFAULT_CATALOG)
    parser.add_argument("--split-root", type=Path, default=mean_benchmark.DEFAULT_SPLITS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--reference", type=Path, default=mean_benchmark.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=mean_benchmark.DEFAULT_SEEDS)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--search", choices=("fast", "standard"), default="standard")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
