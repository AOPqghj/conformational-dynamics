"""Evaluate compact single-structure geometry on frozen and repeated splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "ml"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_plddt_only_benchmark as benchmark_helpers  # noqa: E402
from protein_state_router.evaluation.inference import (  # noqa: E402
    benjamini_hochberg,
    paired_sign_flip_test,
)
from protein_state_router.experiments.benchmark import BenchmarkConfig, run_benchmark  # noqa: E402
from scripts.cache_alphafold_geometry_features import FEATURE_NAMES  # noqa: E402

DEFAULT_FEATURES = (
    REPOSITORY_ROOT / "data/lifecycle/final/initial_8598_dataset/analysis/"
    "alphafold_single_structure_geometry_features.parquet"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ml/results/homology35_single_structure_geometry_baseline"


def _load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"protein_id", "status", *FEATURE_NAMES}
    if missing := required - set(frame):
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) != 7_032 or frame.protein_id.nunique() != 7_032:
        raise ValueError("geometry table must cover 7,032 unique proteins")
    available = frame.status.eq("ok")
    if available.sum() < 0.99 * len(frame):
        raise ValueError("geometry availability is below the prespecified 99% threshold")
    values = frame.loc[available, list(FEATURE_NAMES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("available geometry features contain non-finite values")
    return frame.set_index("protein_id")


def _feature_matrix(
    dataset: pd.DataFrame, feature_table: pd.DataFrame
) -> tuple[np.ndarray, tuple[str, ...]]:
    aligned = feature_table.loc[dataset.protein_id]
    available = aligned.status.eq("ok").to_numpy()
    raw = aligned[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    train = dataset.split.eq("train").to_numpy() & available
    if not train.any():
        raise ValueError("training partition has no available geometry")
    medians = np.nanmedian(raw[train], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("training geometry medians are non-finite")
    missing = ~np.isfinite(raw)
    raw[missing] = np.broadcast_to(medians, raw.shape)[missing]
    matrix = np.column_stack([raw, available.astype(np.float32)])
    names = (*FEATURE_NAMES, "geometry_available")
    return matrix, names


def _reference_comparisons(metrics: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    observed = metrics.loc[metrics.evaluation.eq("repeated")]
    rows = []
    for family in benchmark_helpers.FAMILIES:
        geometry = observed.loc[observed.family.eq(family)].set_index("seed")
        for reference_view in ("covariates", "embedding"):
            comparator = reference.loc[
                reference.family.eq(family) & reference.feature_view.eq(reference_view)
            ].set_index("seed")
            if set(geometry.index) != set(comparator.index):
                raise ValueError(f"{family} {reference_view} does not use the same split seeds")
            row: dict[str, object] = {
                "family": family,
                "comparison": f"single_structure_geometry_vs_{reference_view}",
                "n_splits": len(geometry),
            }
            for metric in benchmark_helpers.METRICS:
                geometry_values = geometry[f"test_{metric}"].sort_index()
                reference_values = comparator[f"test_{metric}"].sort_index()
                difference = geometry_values.to_numpy() - reference_values.to_numpy()
                test = paired_sign_flip_test(difference)
                row[f"geometry_{metric}"] = float(geometry_values.mean())
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


def run(args: argparse.Namespace) -> None:
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("this benchmark requires exactly ten unique repeated-split seeds")
    feature_table = _load_features(args.features.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeated = benchmark_helpers._load_repeated_splits(args.split_root.resolve(), tuple(args.seeds))
    frozen = benchmark_helpers._frozen_dataset(args.catalog.resolve(), output)
    jobs = [("frozen_seed42", 42, frozen), *[("repeated", s, repeated[s]) for s in args.seeds]]
    records = []
    total = len(jobs) * len(benchmark_helpers.FAMILIES)
    for evaluation, seed, dataset_path in jobs:
        dataset = pd.read_parquet(dataset_path)
        features, names = _feature_matrix(dataset, feature_table)
        for family in benchmark_helpers.FAMILIES:
            model_output = output / evaluation / f"seed_{seed}" / family
            metrics_path = model_output / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text())
                event = "model_reused"
            else:
                metrics = run_benchmark(
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
                    features=features,
                    feature_names=names,
                    dataset_reference=str(dataset_path.relative_to(REPOSITORY_ROOT)),
                )
                event = "model_completed"
            records.append(
                {
                    "evaluation": evaluation,
                    "seed": seed,
                    "family": family,
                    "feature_view": "single_structure_geometry",
                    "test_n": int(metrics["sample_count"]),
                    "test_accuracy": metrics["accuracy"],
                    "test_auroc": metrics["auroc"],
                    "test_auprc": metrics["auprc"],
                }
            )
            print(
                json.dumps(
                    {
                        "event": event,
                        "evaluation": evaluation,
                        "seed": seed,
                        "family": family,
                        "test_auroc": metrics["auroc"],
                        "completed_models": len(records),
                        "total_models": total,
                    }
                ),
                flush=True,
            )
            pd.DataFrame(records).to_csv(output / "live_metrics.csv", index=False)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    benchmark_helpers._summarize(metrics).to_csv(output / "summary.csv", index=False)
    _reference_comparisons(metrics, args.reference.resolve()).to_csv(
        output / "paired_reference_comparisons.csv", index=False
    )
    benchmark_helpers._write_json(
        output / "progress.json",
        {
            "status": "completed",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "completed_models": len(records),
            "total_models": total,
            "features": [*FEATURE_NAMES, "geometry_available"],
            "available_proteins": int(feature_table.status.eq("ok").sum()),
            "repeated_split_seeds": args.seeds,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=benchmark_helpers.DEFAULT_CATALOG)
    parser.add_argument("--split-root", type=Path, default=benchmark_helpers.DEFAULT_SPLITS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--reference", type=Path, default=benchmark_helpers.DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=benchmark_helpers.DEFAULT_SEEDS)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--search", choices=("fast", "standard"), default="standard")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
