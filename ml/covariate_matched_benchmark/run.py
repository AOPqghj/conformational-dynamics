"""Run a resumable, covariate-matched ESMFold classification benchmark."""

# ruff: noqa: E402 - repository and thread setup must precede numerical imports.

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPOSITORY_ROOT / "ml"
for import_root in (REPOSITORY_ROOT, ML_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from protein_state_router.evaluation.inference import (
    benjamini_hochberg,
    paired_sign_flip_test,
)
from protein_state_router.experiments.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    sequence_feature_matrix,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_suite import _pooled_single_features

DATASET_ROOT = REPOSITORY_ROOT / "data/lifecycle/final/initial_8598_dataset"
CATALOG = DATASET_ROOT / "homology35_seed42/catalog.parquet"
MANIFEST = DATASET_ROOT / "embedding_manifest.csv"
SPLIT_ROOT = REPOSITORY_ROOT / "ml/results/homology35_confounder_rerun/splits"
PATHPRE_CATALOG = (
    DATASET_ROOT / "homology35_seed42/pathpre_only_controls/pathpre_matched_4395_catalog.parquet"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ml/results/covariate_matched_benchmark"
FULL_SEEDS = tuple(range(10, 20))
PATHPRE_SEEDS = (10, 11, 12)
FEATURE_VIEWS = ("covariates", "embedding", "combined")
FAMILIES = ("linear", "tree")
EXPECTED_CATALOG_ROWS = 8_598
EXPECTED_PLDDT_ROWS = 7_032
EXPECTED_PATHPRE_ROWS = 4_395
PROPENSITY_BINS = 20
CALIPER_STANDARD_DEVIATIONS = 0.20
BALANCE_TARGET = 0.10
BALANCE_GATE = 0.15
MINIMUM_PAIR_RETENTION = 0.40


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def build_plan(output: Path) -> dict[str, Any]:
    inputs = (CATALOG, MANIFEST, PATHPRE_CATALOG)
    missing = [str(path) for path in inputs if not path.is_file()]
    split_paths = [SPLIT_ROOT / f"split_{seed}.parquet" for seed in FULL_SEEDS]
    missing.extend(str(path) for path in split_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    plan = {
        "schema_version": 1,
        "created_at_utc": now(),
        "experiment": "covariate_matched_esmfold",
        "runner_sha256": sha256(Path(__file__)),
        "output_root": relative(output),
        "cpu_threads": 2,
        "catalog": {"path": relative(CATALOG), "sha256": sha256(CATALOG)},
        "embedding_manifest": {"path": relative(MANIFEST), "sha256": sha256(MANIFEST)},
        "pathpre_catalog": {
            "path": relative(PATHPRE_CATALOG),
            "sha256": sha256(PATHPRE_CATALOG),
        },
        "splits": [
            {"seed": seed, "path": relative(path), "sha256": sha256(path)}
            for seed, path in zip(FULL_SEEDS, split_paths, strict=True)
        ],
        "cohorts": [
            {"name": "full", "seeds": list(FULL_SEEDS)},
            {"name": "pathpre", "seeds": list(PATHPRE_SEEDS)},
        ],
        "matching": {
            "method": "training_fitted_propensity_score_coarsened_exact_matching",
            "propensity_bins": PROPENSITY_BINS,
            "caliper_standard_deviations": CALIPER_STANDARD_DEVIATIONS,
            "target_max_absolute_smd": BALANCE_TARGET,
            "hard_max_absolute_smd": BALANCE_GATE,
            "minimum_pair_retention": MINIMUM_PAIR_RETENTION,
            "partition_local": True,
            "covariates": [
                "log1p_sequence_length",
                "amino_acid_fractions",
                "sequence_entropy",
                "alphafold_mean_plddt",
            ],
        },
        "models": {
            "families": list(FAMILIES),
            "feature_views": list(FEATURE_VIEWS),
            "search": "standard",
            "selection_metric": "validation_auroc",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run_plan.json", plan)
    print(json.dumps({"event": "plan_prepared", "path": str(output / "run_plan.json")}, indent=2))
    return plan


def validate_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or plan.get("experiment") != "covariate_matched_esmfold":
        raise ValueError("unsupported or incorrect run plan")
    if plan.get("runner_sha256") != sha256(Path(__file__)):
        raise ValueError("runner changed after the plan was prepared")
    checks = [plan["catalog"], plan["embedding_manifest"], plan["pathpre_catalog"]]
    checks.extend(plan["splits"])
    for item in checks:
        path = resolve(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"planned input is missing or changed: {path}")
    if plan["cpu_threads"] != 2:
        raise ValueError("the reviewed plan must remain bounded to two CPU threads")
    return plan


def claim_lock(output: Path, plan_path: Path) -> Path:
    lock = output / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another runner owns {lock}") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at_utc": now(),
                "plan": str(plan_path),
                "plan_sha256": sha256(plan_path),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return lock


def covariates(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    sequence, names = sequence_feature_matrix(frame)
    plddt = frame.alphafold_mean_plddt.to_numpy(dtype=np.float32)[:, None]
    values = np.concatenate((sequence, plddt), axis=1)
    if not np.isfinite(values).all():
        raise ValueError("matching covariates must be finite")
    return values, (*names, "alphafold_mean_plddt")


def fit_propensity(
    frame: pd.DataFrame, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = frame.split.eq("train").to_numpy()
    labels = frame.dataset_label.to_numpy(dtype=np.int64)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(values[train], labels[train])
    scaled = model.named_steps["scale"].transform(values).astype(np.float32)
    probabilities = np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities))
    quantiles = np.linspace(0.0, 1.0, PROPENSITY_BINS + 1)
    edges = np.unique(np.quantile(logits[train], quantiles))
    if len(edges) < 4:
        raise ValueError("propensity model produced too few distinct matching bins")
    edges[0], edges[-1] = -np.inf, np.inf
    return logits, edges, scaled


def match_partition(
    frame: pd.DataFrame,
    logits: np.ndarray,
    scaled_covariates: np.ndarray,
    edges: np.ndarray,
    partition: str,
    seed: int,
    caliper: float,
) -> pd.DataFrame:
    subset = frame.loc[frame.split.eq(partition)].copy()
    subset["propensity_logit"] = logits[frame.split.eq(partition).to_numpy()]
    subset["propensity_bin"] = np.digitize(subset.propensity_logit, edges[1:-1])
    pairs: list[pd.DataFrame] = []
    pair_number = 0
    for bin_number, values in subset.groupby("propensity_bin", sort=True):
        classes = {
            label: values.loc[values.dataset_label.eq(label)].sort_values(
                ["propensity_logit", "protein_id"]
            )
            for label in (0, 1)
        }
        count = min(len(classes[0]), len(classes[1]))
        if not count:
            continue
        left_positions, right_positions = linear_sum_assignment(
            cdist(
                scaled_covariates[classes[0].index.to_numpy()],
                scaled_covariates[classes[1].index.to_numpy()],
                metric="euclidean",
            )
        )
        for left_position, right_position in zip(left_positions, right_positions, strict=True):
            left = classes[0].iloc[left_position]
            right = classes[1].iloc[right_position]
            distance = abs(float(left.propensity_logit) - float(right.propensity_logit))
            if distance > caliper:
                continue
            pair_id = f"{seed}:{partition}:{int(bin_number)}:{pair_number}"
            pair_number += 1
            pair = pd.DataFrame([left, right])
            pair["match_pair_id"] = pair_id
            pair["propensity_distance"] = distance
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"matching retained no pairs for {partition}")
    matched = pd.concat(pairs, ignore_index=True)
    if matched.protein_id.duplicated().any() or matched.groupby("match_pair_id").size().ne(2).any():
        raise ValueError("matching reused a protein or produced an incomplete pair")
    if matched.groupby("match_pair_id").dataset_label.nunique().ne(2).any():
        raise ValueError("every match pair must contain one protein from each class")
    return matched


def standardized_difference(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    first, second = values[labels == 0], values[labels == 1]
    pooled = np.sqrt((first.var(axis=0) + second.var(axis=0)) / 2.0)
    return np.divide(
        second.mean(axis=0) - first.mean(axis=0),
        pooled,
        out=np.zeros_like(pooled),
        where=pooled > 0,
    )


def prune_for_balance(
    matched: pd.DataFrame,
    feature_lookup: pd.DataFrame,
    *,
    target: float = BALANCE_TARGET,
    minimum_pair_retention: float = MINIMUM_PAIR_RETENTION,
) -> pd.DataFrame:
    """Deterministically remove the pairs driving residual covariate imbalance."""
    pair_ids = matched.match_pair_id.drop_duplicates().tolist()
    minimum_pairs = max(2, int(np.ceil(len(pair_ids) * minimum_pair_retention)))
    selected = matched.copy()
    while len(pair_ids) > minimum_pairs:
        values = feature_lookup.loc[selected.protein_id].to_numpy()
        labels = selected.dataset_label.to_numpy()
        differences = standardized_difference(values, labels)
        current_maximum = float(np.max(np.abs(differences)))
        if current_maximum <= target:
            break
        pairs = selected.pivot(index="match_pair_id", columns="dataset_label", values="protein_id")
        class_zero = feature_lookup.loc[pairs[0]].to_numpy()
        class_one = feature_lookup.loc[pairs[1]].to_numpy()
        remaining = len(pairs) - 1
        means_zero = (class_zero.sum(axis=0) - class_zero) / remaining
        means_one = (class_one.sum(axis=0) - class_one) / remaining
        variances_zero = (
            np.square(class_zero).sum(axis=0) - np.square(class_zero)
        ) / remaining - np.square(means_zero)
        variances_one = (
            np.square(class_one).sum(axis=0) - np.square(class_one)
        ) / remaining - np.square(means_one)
        pooled = np.sqrt(np.maximum((variances_zero + variances_one) / 2.0, 0.0))
        leave_one_out = np.divide(
            means_one - means_zero,
            pooled,
            out=np.zeros_like(pooled),
            where=pooled > 0,
        )
        candidate_maxima = np.max(np.abs(leave_one_out), axis=1)
        remove_position = int(np.argmin(candidate_maxima))
        if float(candidate_maxima[remove_position]) >= current_maximum - 1e-9:
            break
        remove_id = pairs.index[remove_position]
        selected = selected.loc[selected.match_pair_id.ne(remove_id)].copy()
        pair_ids.remove(remove_id)
    return selected


def make_matched_dataset(
    frame: pd.DataFrame,
    seed: int,
    feature_values: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logits, edges, scaled = fit_propensity(frame, feature_values)
    train_logits = logits[frame.split.eq("train").to_numpy()]
    caliper = CALIPER_STANDARD_DEVIATIONS * float(np.std(train_logits, ddof=1))
    matched = pd.concat(
        [
            match_partition(frame, logits, scaled, edges, partition, seed, caliper)
            for partition in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    feature_lookup = pd.DataFrame(feature_values, index=frame.protein_id, columns=feature_names)
    matched = pd.concat(
        [
            prune_for_balance(matched.loc[matched.split.eq(partition)].copy(), feature_lookup)
            for partition in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    balance: list[dict[str, Any]] = []
    for partition in ("train", "val", "test"):
        original = frame.loc[frame.split.eq(partition)]
        selected = matched.loc[matched.split.eq(partition)]
        before_values = feature_lookup.loc[original.protein_id].to_numpy()
        after_values = feature_lookup.loc[selected.protein_id].to_numpy()
        before = standardized_difference(before_values, original.dataset_label.to_numpy())
        after = standardized_difference(after_values, selected.dataset_label.to_numpy())
        for name, before_value, after_value in zip(feature_names, before, after, strict=True):
            balance.append(
                {
                    "seed": seed,
                    "split": partition,
                    "feature": name,
                    "smd_before": float(before_value),
                    "smd_after": float(after_value),
                    "absolute_smd_before": abs(float(before_value)),
                    "absolute_smd_after": abs(float(after_value)),
                    "original_n": len(original),
                    "matched_n": len(selected),
                    "matched_pairs": len(selected) // 2,
                }
            )
    balance_frame = pd.DataFrame(balance)
    test_balance = balance_frame.loc[balance_frame.split.eq("test")]
    if test_balance.absolute_smd_after.max() > BALANCE_GATE:
        raise ValueError(
            f"test matching failed the predeclared absolute standardized-difference gate of {BALANCE_GATE}"
        )
    return matched, balance_frame


def cohort_for_seed(
    catalog: pd.DataFrame,
    pathpre_ids: set[str],
    split_path: Path,
    cohort_name: str,
) -> pd.DataFrame:
    assignments = pd.read_parquet(split_path)[["protein_id", "split"]]
    frame = catalog.drop(columns="split", errors="ignore").merge(
        assignments, on="protein_id", how="inner", validate="one_to_one"
    )
    if cohort_name == "pathpre":
        frame = frame.loc[frame.protein_id.isin(pathpre_ids)].copy()
    frame = frame.loc[frame.alphafold_mean_plddt.notna()].reset_index(drop=True)
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError(f"{cohort_name} cohort lacks a required split")
    if frame.groupby("homology_group_id").split.nunique().max() != 1:
        raise ValueError("homology groups cross matched benchmark partitions")
    if frame.groupby("split").dataset_label.nunique().min() != 2:
        raise ValueError(f"{cohort_name} split lacks both classes")
    return frame


def completed_metrics(model_dir: Path) -> dict[str, float] | None:
    required = (
        "metrics.json",
        "validation_selection.json",
        "test_predictions.parquet",
        "manifest.json",
    )
    if not all((model_dir / name).is_file() for name in required):
        return None
    return json.loads((model_dir / "metrics.json").read_text())


def run_model(
    dataset: Path,
    model_dir: Path,
    family: str,
    view: str,
    seed: int,
    features: np.ndarray,
    names: tuple[str, ...],
    update_progress,
) -> dict[str, float]:
    if metrics := completed_metrics(model_dir):
        return metrics
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_dir.name}.tmp-", dir=model_dir.parent))
    metrics = run_benchmark(
        dataset,
        temporary,
        BenchmarkConfig(
            family=family,
            random_seed=seed,
            primary_metric="auroc",
            device="cpu",
            search="standard",
            save_model=False,
            cpu_threads=2,
        ),
        features=features,
        feature_names=names,
        progress_callback=update_progress,
        dataset_reference=relative(dataset),
    )
    if model_dir.exists():
        raise RuntimeError(f"refusing to replace existing incomplete output: {model_dir}")
    os.replace(temporary, model_dir)
    return metrics


def summarize(output: Path, records: list[dict[str, Any]]) -> None:
    results = pd.DataFrame(records)
    atomic_csv(output / "all_metrics.csv", results)
    if results.empty:
        return
    metric_columns = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auroc",
        "test_auprc",
        "test_mcc",
        "test_brier",
    ]
    summary = results.groupby(["cohort", "family", "feature_view"], as_index=False).agg(
        splits=("seed", "nunique"),
        **{f"{column}_mean": (column, "mean") for column in metric_columns},
    )
    atomic_csv(output / "summary.csv", summary)
    comparisons = []
    for (cohort, family), values in results.groupby(["cohort", "family"]):
        if not {"covariates", "embedding", "combined"}.issubset(set(values.feature_view)):
            continue
        pivot = values.pivot(index="seed", columns="feature_view", values=metric_columns)
        for comparison, baseline, target in (
            ("embedding_vs_covariates", "covariates", "embedding"),
            ("combined_vs_embedding", "embedding", "combined"),
        ):
            report: dict[str, Any] = {
                "cohort": cohort,
                "family": family,
                "comparison": comparison,
                "n_splits": len(pivot),
            }
            if len(pivot) < 2:
                continue
            for metric in metric_columns:
                differences = (pivot[(metric, target)] - pivot[(metric, baseline)]).to_numpy()
                short = metric.removeprefix("test_")
                report[f"mean_difference_{short}"] = float(np.mean(differences))
                finite = differences[np.isfinite(differences)]
                if len(finite) < 2:
                    report[f"p_{short}"] = np.nan
                    report[f"method_{short}"] = "not_estimable_incomplete_splits"
                else:
                    test = paired_sign_flip_test(finite, seed=42)
                    report[f"p_{short}"] = test["permutation_p_two_sided"]
                    report[f"method_{short}"] = test["permutation_method"]
            comparisons.append(report)
    tests = pd.DataFrame(comparisons)
    p_columns = [name for name in tests if name.startswith("p_")]
    if p_columns:
        raw = tests[p_columns].to_numpy().ravel()
        adjusted = np.full(raw.shape, np.nan, dtype=float)
        finite = np.isfinite(raw)
        if finite.any():
            adjusted[finite] = benjamini_hochberg(raw[finite])
        adjusted = adjusted.reshape(len(tests), len(p_columns))
        for index, column in enumerate(p_columns):
            tests[f"fdr_{column.removeprefix('p_')}"] = adjusted[:, index]
    atomic_csv(output / "paired_tests.csv", tests)


def run(plan_path: Path) -> None:
    plan = validate_plan(plan_path)
    output = resolve(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    claim_lock(output, plan_path)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    catalog = pd.read_parquet(resolve(plan["catalog"]["path"]))
    if len(catalog) != EXPECTED_CATALOG_ROWS or catalog.protein_id.duplicated().any():
        raise ValueError("canonical catalog count or identity invariant failed")
    if int(catalog.alphafold_mean_plddt.notna().sum()) != EXPECTED_PLDDT_ROWS:
        raise ValueError("unexpected pLDDT-observed cohort size")
    pathpre = pd.read_parquet(resolve(plan["pathpre_catalog"]["path"]))
    if len(pathpre) != EXPECTED_PATHPRE_ROWS or pathpre.protein_id.duplicated().any():
        raise ValueError("unexpected PathPre cohort")
    pathpre_ids = set(pathpre.protein_id.astype(str))
    eligible = catalog.loc[catalog.alphafold_mean_plddt.notna()].reset_index(drop=True)
    manifest = resolve(plan["embedding_manifest"]["path"])
    manifest_frame = pd.read_csv(manifest)
    if manifest_frame.protein_id.duplicated().any() or not set(eligible.protein_id).issubset(
        set(manifest_frame.protein_id)
    ):
        raise ValueError("embedding manifest does not uniquely cover the matched cohort")
    eligible_manifest = manifest_frame.loc[manifest_frame.protein_id.isin(eligible.protein_id)]
    missing_files = [
        value for value in eligible_manifest.embedding_path if not resolve(str(value)).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"embedding manifest has {len(missing_files)} missing files")
    print(json.dumps({"event": "pooling_started", "proteins": len(eligible)}), flush=True)
    pooling_started = time.monotonic()
    pooling_finished = threading.Event()

    def pooling_heartbeat() -> None:
        while not pooling_finished.wait(30):
            elapsed = round(time.monotonic() - pooling_started, 1)
            atomic_json(
                output / "progress.json",
                {
                    "status": "running",
                    "stage": "pooling_embeddings",
                    "updated_at_utc": now(),
                    "elapsed_seconds": elapsed,
                    "completed_models": 0,
                },
            )
            print(
                json.dumps({"event": "pooling_heartbeat", "elapsed_seconds": elapsed}), flush=True
            )

    heartbeat_thread = threading.Thread(target=pooling_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with threadpool_limits(limits=2):
            pooled, embedding_names = _pooled_single_features(eligible, manifest, "esmfold")
    finally:
        pooling_finished.set()
        heartbeat_thread.join()
    master_covariates, covariate_names = covariates(eligible)
    master_index = pd.Series(np.arange(len(eligible)), index=eligible.protein_id)
    split_specs = {int(item["seed"]): resolve(item["path"]) for item in plan["splits"]}
    total_models = (
        sum(len(item["seeds"]) for item in plan["cohorts"]) * len(FAMILIES) * len(FEATURE_VIEWS)
    )
    records: list[dict[str, Any]] = []
    started = time.monotonic()

    def publish(**extra: Any) -> None:
        completed = len(records)
        elapsed = time.monotonic() - started
        eta_seconds = elapsed / completed * (total_models - completed) if completed else None
        atomic_json(
            output / "progress.json",
            {
                "status": "running",
                "updated_at_utc": now(),
                "completed_models": completed,
                "total_models": total_models,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(eta_seconds, 1) if eta_seconds else None,
                **extra,
            },
        )

    for cohort_spec in plan["cohorts"]:
        cohort_name = str(cohort_spec["name"])
        for seed in cohort_spec["seeds"]:
            seed = int(seed)
            frame = cohort_for_seed(catalog, pathpre_ids, split_specs[seed], cohort_name)
            indices = master_index.loc[frame.protein_id].to_numpy(dtype=int)
            frame_covariates = master_covariates[indices]
            matched, balance = make_matched_dataset(frame, seed, frame_covariates, covariate_names)
            seed_root = output / cohort_name / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            dataset = seed_root / "matched_catalog.parquet"
            balance_path = seed_root / "covariate_balance.csv"
            if dataset.is_file():
                existing = pd.read_parquet(dataset)
                if not existing.protein_id.equals(matched.protein_id):
                    raise ValueError(f"matched cohort changed on resume: {cohort_name} seed {seed}")
            else:
                atomic_parquet(dataset, matched)
            atomic_csv(balance_path, balance)
            matched_indices = master_index.loc[matched.protein_id].to_numpy(dtype=int)
            views = {
                "covariates": (master_covariates[matched_indices], covariate_names),
                "embedding": (pooled[matched_indices], embedding_names),
                "combined": (
                    np.concatenate(
                        (master_covariates[matched_indices], pooled[matched_indices]), axis=1
                    ),
                    (*covariate_names, *embedding_names),
                ),
            }
            for family in FAMILIES:
                for view in FEATURE_VIEWS:
                    model_dir = seed_root / f"{family}_{view}"

                    def heartbeat(
                        event: dict[str, Any],
                        cohort: str = cohort_name,
                        current_seed: int = seed,
                        current_family: str = family,
                        current_view: str = view,
                    ) -> None:
                        publish(
                            cohort=cohort,
                            seed=current_seed,
                            family=current_family,
                            feature_view=current_view,
                            candidate_event=event,
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "candidate_progress",
                                    "cohort": cohort,
                                    "seed": current_seed,
                                    "family": current_family,
                                    "view": current_view,
                                    **event,
                                }
                            ),
                            flush=True,
                        )

                    metrics = run_model(
                        dataset,
                        model_dir,
                        family,
                        view,
                        seed,
                        *views[view],
                        heartbeat,
                    )
                    records.append(
                        {
                            "cohort": cohort_name,
                            "seed": seed,
                            "family": family,
                            "feature_view": view,
                            "train_n": int(matched.split.eq("train").sum()),
                            "val_n": int(matched.split.eq("val").sum()),
                            "test_n": int(matched.split.eq("test").sum()),
                            **{f"test_{key}": value for key, value in metrics.items()},
                        }
                    )
                    summarize(output, records)
                    publish(cohort=cohort_name, seed=seed, family=family, feature_view=view)
                    print(
                        json.dumps(
                            {
                                "event": "model_complete",
                                "completed_models": len(records),
                                "total_models": total_models,
                                "cohort": cohort_name,
                                "seed": seed,
                                "family": family,
                                "view": view,
                                "test_auroc": metrics["auroc"],
                            }
                        ),
                        flush=True,
                    )
    summarize(output, records)
    atomic_json(
        output / "progress.json",
        {
            "status": "completed",
            "updated_at_utc": now(),
            "completed_models": len(records),
            "total_models": total_models,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="write a reviewed immutable run plan")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    execute = subparsers.add_parser("run", help="validate and execute a prepared plan")
    execute.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        build_plan(args.output)
    else:
        run(args.plan)


if __name__ == "__main__":
    main()
