"""Evaluate mean pLDDT alone on frozen and repeated homology-aware splits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from protein_state_router.evaluation.inference import (  # noqa: E402
    benjamini_hochberg,
    paired_sign_flip_test,
)
from protein_state_router.experiments.benchmark import (  # noqa: E402
    BenchmarkConfig,
    run_benchmark,
)

DEFAULT_CATALOG = (
    REPOSITORY_ROOT / "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
)
DEFAULT_SPLITS = REPOSITORY_ROOT / "ml/results/homology35_confounder_rerun/splits"
DEFAULT_REFERENCE = (
    REPOSITORY_ROOT / "ml/results/homology35_confounder_rerun/pooled_confounder/all_metrics.csv"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ml/results/homology35_plddt_only_baseline"
DEFAULT_SEEDS = tuple(range(10, 20))
EXPECTED_ROWS = 7_032
FAMILIES = ("linear", "tree")
METRICS = ("accuracy", "auroc", "auprc")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_frame(frame: pd.DataFrame, source: Path) -> None:
    required = {
        "protein_id",
        "sequence",
        "sequence_length",
        "dataset_label",
        "split",
        "homology_group_id",
        "alphafold_mean_plddt",
    }
    if missing := required - set(frame):
        raise ValueError(f"{source} missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS or frame.protein_id.nunique() != EXPECTED_ROWS:
        raise ValueError(f"{source} must contain exactly {EXPECTED_ROWS:,} unique proteins")
    if frame.alphafold_mean_plddt.isna().any():
        raise ValueError(f"{source} contains missing mean pLDDT")
    if not np.isfinite(frame.alphafold_mean_plddt.to_numpy(float)).all():
        raise ValueError(f"{source} contains non-finite mean pLDDT")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError(f"{source} must contain train, val, and test partitions")
    if (frame.groupby("homology_group_id").split.nunique() > 1).any():
        raise ValueError(f"{source} has a homology group crossing partitions")
    for partition in ("train", "val", "test"):
        if frame.loc[frame.split.eq(partition), "dataset_label"].nunique() != 2:
            raise ValueError(f"{source} {partition} partition lacks both labels")


def _load_repeated_splits(split_root: Path, seeds: tuple[int, ...]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    reference: pd.DataFrame | None = None
    for seed in seeds:
        path = split_root / f"split_{seed}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        _validate_frame(frame, path)
        identity = frame[["protein_id", "dataset_label", "alphafold_mean_plddt"]].sort_values(
            "protein_id", ignore_index=True
        )
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError(f"{path} does not preserve the repeated-split cohort and labels")
        paths[seed] = path
    return paths


def _frozen_dataset(catalog_path: Path, output: Path) -> Path:
    catalog = pd.read_parquet(catalog_path)
    if "alphafold_mean_plddt" not in catalog:
        raise ValueError("catalog lacks alphafold_mean_plddt")
    frame = catalog.loc[catalog.alphafold_mean_plddt.notna()].copy()
    _validate_frame(frame, catalog_path)
    path = output / "frozen_seed42_catalog.parquet"
    frame.to_parquet(path, index=False)
    return path


def _plddt_features(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    values = frame.alphafold_mean_plddt.to_numpy(dtype=np.float32)[:, None]
    return values, ("alphafold_mean_plddt",)


def _record(
    evaluation: str,
    seed: int,
    family: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "evaluation": evaluation,
        "seed": seed,
        "family": family,
        "feature_view": "mean_plddt_only",
        "test_n": int(metrics["sample_count"]),
        "test_accuracy": metrics["accuracy"],
        "test_auroc": metrics["auroc"],
        "test_auprc": metrics["auprc"],
    }


def _confidence_interval(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    half_width = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
    mean = float(array.mean())
    return mean - half_width, mean + half_width


def _summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (evaluation, family), group in records.groupby(["evaluation", "family"], sort=False):
        row: dict[str, object] = {
            "evaluation": evaluation,
            "family": family,
            "n_splits": len(group),
            "test_n_mean": float(group.test_n.mean()),
        }
        for metric in METRICS:
            values = group[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            if len(values) > 1:
                lower, upper = _confidence_interval(values)
                row[f"{metric}_ci95_lower"] = lower
                row[f"{metric}_ci95_upper"] = upper
            else:
                row[f"{metric}_ci95_lower"] = np.nan
                row[f"{metric}_ci95_upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_comparisons(records: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = pd.read_csv(reference_path)
    required = {"seed", "family", "feature_view", *[f"test_{metric}" for metric in METRICS]}
    if missing := required - set(reference):
        raise ValueError(f"{reference_path} missing columns: {sorted(missing)}")
    observed = records.loc[records.evaluation.eq("repeated")].copy()
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        plddt = observed.loc[observed.family.eq(family)].set_index("seed")
        for reference_view in ("covariates", "embedding"):
            comparator = reference.loc[
                reference.family.eq(family) & reference.feature_view.eq(reference_view)
            ].set_index("seed")
            if set(plddt.index) != set(comparator.index):
                raise ValueError(f"{family} {reference_view} does not use the same split seeds")
            row: dict[str, object] = {
                "family": family,
                "comparison": f"mean_plddt_only_vs_{reference_view}",
                "n_splits": len(plddt),
            }
            for metric in METRICS:
                plddt_values = plddt[f"test_{metric}"].sort_index()
                reference_values = comparator[f"test_{metric}"].sort_index()
                differences = plddt_values.to_numpy() - reference_values.to_numpy()
                test = paired_sign_flip_test(differences)
                row[f"mean_plddt_only_{metric}"] = float(plddt_values.mean())
                row[f"mean_{reference_view}_{metric}"] = float(reference_values.mean())
                row[f"mean_difference_{metric}"] = float(differences.mean())
                row[f"paired_{metric}_p"] = test["permutation_p_two_sided"]
                row[f"paired_{metric}_method"] = test["permutation_method"]
            rows.append(row)
    table = pd.DataFrame(rows)
    p_columns = [column for column in table if column.endswith("_p")]
    adjusted = benjamini_hochberg(table[p_columns].to_numpy(dtype=float).ravel()).reshape(
        len(table), len(p_columns)
    )
    for index, column in enumerate(p_columns):
        table[column.removesuffix("_p") + "_fdr"] = adjusted[:, index]
    return table


def _run_one(
    dataset: Path,
    model_output: Path,
    evaluation: str,
    seed: int,
    family: str,
    cpu_threads: int,
    search: str,
) -> dict[str, object]:
    metrics_path = model_output / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        event = "model_reused"
    else:
        frame = pd.read_parquet(dataset)
        features, names = _plddt_features(frame)

        def progress(payload: dict[str, object]) -> None:
            print(
                json.dumps(
                    {
                        **payload,
                        "evaluation": evaluation,
                        "seed": seed,
                        "family": family,
                    }
                ),
                flush=True,
            )

        metrics = run_benchmark(
            dataset,
            model_output,
            BenchmarkConfig(
                family=family,
                random_seed=seed,
                primary_metric="auroc",
                device="cpu",
                search=search,
                save_model=False,
                cpu_threads=cpu_threads,
            ),
            features=features,
            feature_names=names,
            progress_callback=progress,
            dataset_reference=str(dataset.relative_to(REPOSITORY_ROOT)),
        )
        event = "model_completed"
    print(
        json.dumps(
            {
                "event": event,
                "evaluation": evaluation,
                "seed": seed,
                "family": family,
                "test_accuracy": metrics["accuracy"],
                "test_auroc": metrics["auroc"],
            }
        ),
        flush=True,
    )
    return _record(evaluation, seed, family, metrics)


def run(args: argparse.Namespace) -> None:
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("this benchmark requires exactly ten unique repeated-split seeds")
    if args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeated_paths = _load_repeated_splits(args.split_root.resolve(), tuple(args.seeds))
    frozen_path = _frozen_dataset(args.catalog.resolve(), output)
    _write_json(
        output / "confidence_feature_audit.json",
        {
            "mean_plddt_available_proteins": EXPECTED_ROWS,
            "residue_level_plddt_cached": False,
            "distribution_features_evaluated": False,
            "reason": (
                "The catalog stores AlphaFold mean pLDDT and residue count, while the "
                "cached ESMFold bundles store representations but no residue-level pLDDT vector."
            ),
        },
    )
    records: list[dict[str, object]] = []
    jobs = [
        ("frozen_seed42", 42, frozen_path),
        *[("repeated", seed, repeated_paths[seed]) for seed in args.seeds],
    ]
    total = len(jobs) * len(FAMILIES)
    for evaluation, seed, dataset in jobs:
        for family in FAMILIES:
            model_output = output / evaluation / f"seed_{seed}" / family
            records.append(
                _run_one(
                    dataset,
                    model_output,
                    evaluation,
                    seed,
                    family,
                    args.cpu_threads,
                    args.search,
                )
            )
            frame = pd.DataFrame(records)
            frame.to_csv(output / "live_metrics.csv", index=False)
            _write_json(
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
    _reference_comparisons(metrics, args.reference.resolve()).to_csv(
        output / "paired_reference_comparisons.csv", index=False
    )
    _write_json(
        output / "progress.json",
        {
            "status": "completed",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "completed_models": len(records),
            "total_models": total,
            "repeated_split_seeds": args.seeds,
        },
    )
    print(
        json.dumps(
            {
                "event": "benchmark_completed",
                "output": str(output),
                "models": len(records),
            }
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--search", choices=("fast", "standard"), default="standard")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
