"""Audit pLDDT shortcut risk with inference strata and residualized embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from protein_state_router.evaluation.inference import benjamini_hochberg, paired_sign_flip_test
from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.experiments.benchmark import (
    BenchmarkConfig,
    run_benchmark,
    sequence_feature_matrix,
)
from protein_state_router.representations.registry import representation_choices
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from train_suite import _load_features

ROOT = Path("data/lifecycle/final/initial_8598_dataset")
CATALOG_PATH = ROOT / "homology35_seed42/catalog.parquet"
SOURCE = Path("ml/results/homology35_plddt_confounder_benchmark")
OUTPUT = Path("ml/results/homology35_plddt_followup")
STRATIFIED_OUTPUT = OUTPUT / "stratified"
EMBEDDING_MANIFEST = ROOT / "embedding_manifest.csv"
REPRESENTATION_NAME = "esmfold"
EXPECTED_ROWS: int | None = 7032
SEEDS = (10,)
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
BIN_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
STRATUM_EDGES = (-np.inf, 70.0, 90.0, np.inf)
STRATUM_LABELS = ("low_below_70", "medium_70_to_90", "high_90_or_above")
# Subset selection must not change STRATUM_LABELS: pd.cut requires one label
# per fixed bin edge.  Runners may change SELECTED_STRATA instead.
SELECTED_STRATA = STRATUM_LABELS
STRATIFIED_VIEWS = (
    "metadata",
    "embedding",
    "combined",
    "residual_plddt",
    "residual_full_covariates",
)
CENTRAL = ZoneInfo("America/Chicago")


def _now() -> str:
    return datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M:%S %Z")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_readme(residual_only: bool) -> None:
    path = OUTPUT / "README.md"
    if path.exists():
        return
    path.write_text(
        "# pLDDT follow-up artifacts\n\n"
        "`residual_metrics.csv` contains raw and residualized pooled-embedding test metrics.\n"
        "`residual_permutation_tests.csv` contains paired sign-flip comparisons against raw embeddings.\n"
        "`live_report.html` and `progress.json` refresh while the residual benchmark runs.\n"
        + (
            "This run intentionally skips the separate pLDDT-bin inference summary.\n"
            if residual_only
            else "`bin_metrics.csv` contains per-seed, pLDDT-stratified held-out inference from the completed confounder models.\n"
        )
    )


def plddt_bin(values: pd.Series) -> pd.Categorical:
    return pd.cut(values, BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)


def plddt_stratum(values: pd.Series) -> pd.Categorical:
    return pd.cut(values, STRATUM_EDGES, labels=STRATUM_LABELS, right=False)


def bin_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels):
        return {
            "sample_count": 0.0,
            "negative_count": 0.0,
            "positive_count": 0.0,
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    result = {
        "sample_count": float(len(labels)),
        "negative_count": float((labels == 0).sum()),
        "positive_count": float((labels == 1).sum()),
        "accuracy": float(((probabilities >= 0.5) == labels).mean()),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if np.unique(labels).size == 2:
        result.update(
            {key: classification_metrics(labels, probabilities)[key] for key in ("auroc", "auprc")}
        )
    return result


def _catalog() -> pd.DataFrame:
    frame = pd.read_parquet(CATALOG_PATH)
    frame = frame.loc[frame.alphafold_mean_plddt.notna()].copy()
    # The pooled benchmark may have excluded a small number of proteins for
    # representation-specific reasons (for example, a missing embedding).
    # Stratified retraining must use exactly that saved cohort, rather than
    # every catalog row with an available pLDDT value.  Otherwise the saved
    # homology split cannot cover the requested data and a restart is both
    # impossible and conceptually inconsistent with the pooled benchmark.
    saved_split_ids: list[set[str]] = []
    for seed in SEEDS:
        split_path = SOURCE / f"split_{seed}.parquet"
        if split_path.is_file():
            saved_split_ids.append(set(pd.read_parquet(split_path, columns=["protein_id"]).protein_id))
    if saved_split_ids:
        reference_ids = saved_split_ids[0]
        if any(ids != reference_ids for ids in saved_split_ids[1:]):
            raise ValueError("saved pLDDT splits do not share one eligible protein cohort")
        frame = frame.loc[frame.protein_id.isin(reference_ids)].copy()
    if (
        EXPECTED_ROWS is not None and len(frame) != EXPECTED_ROWS
    ) or frame.dataset_label.nunique() != 2:
        raise ValueError("expected the requested pLDDT-observed binary subset with both classes")
    frame["plddt_normalized"] = frame.alphafold_mean_plddt / 100.0
    return frame


def _require_source_complete() -> None:
    status = json.loads((SOURCE / "progress.json").read_text())
    if status.get("status") != "completed":
        raise RuntimeError(
            "pLDDT confounder benchmark is not complete; rerun after its ten splits finish"
        )
    missing = [
        f"seed_{seed}/{family}_{view}/test_predictions.parquet"
        for seed in SEEDS
        for family in ("linear", "tree")
        for view in ("covariates", "embedding", f"covariates_plus_{REPRESENTATION_NAME}")
        if not (SOURCE / f"seed_{seed}" / f"{family}_{view}" / "test_predictions.parquet").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing completed pLDDT predictions: {missing[:3]}")


def _require_split_inputs() -> None:
    missing = [
        str(SOURCE / f"split_{seed}.parquet")
        for seed in SEEDS
        if not (SOURCE / f"split_{seed}.parquet").is_file()
    ]
    if missing:
        raise RuntimeError(f"missing saved split inputs for follow-up tests: {missing[:3]}")


def _wait_for_source() -> None:
    while True:
        status_path = SOURCE / "progress.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text()).get("status")
            if status == "completed":
                return
        time.sleep(60)


def _wait_for_residual() -> None:
    deadline = time.monotonic() + 24 * 60 * 60
    while time.monotonic() < deadline:
        status_path = OUTPUT / "progress.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text()).get("status")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError("residual benchmark failed; stratified benchmark will not start")
        time.sleep(60)
    raise TimeoutError("residual benchmark did not complete within 24 hours")


def _wait_for_stratified() -> None:
    deadline = time.monotonic() + 48 * 60 * 60
    while time.monotonic() < deadline:
        status_path = STRATIFIED_OUTPUT / "progress.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text()).get("status")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError("stratified benchmark failed; residual expansion will not start")
        time.sleep(60)
    raise TimeoutError("stratified benchmark did not complete within 48 hours")


def run_bin_inference(catalog: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metadata = catalog[["protein_id", "dataset_label", "plddt_normalized"]].copy()
    for seed in SEEDS:
        for family in ("linear", "tree"):
            for view in ("covariates", "embedding", f"covariates_plus_{REPRESENTATION_NAME}"):
                path = SOURCE / f"seed_{seed}" / f"{family}_{view}" / "test_predictions.parquet"
                predictions = pd.read_parquet(path)[["protein_id", "dataset_label", "probability"]]
                joined = predictions.merge(
                    metadata,
                    on="protein_id",
                    suffixes=("_prediction", "_catalog"),
                    validate="one_to_one",
                )
                if len(joined) != len(predictions) or not joined.dataset_label_prediction.equals(
                    joined.dataset_label_catalog
                ):
                    raise ValueError(
                        f"prediction metadata mismatch for seed {seed} {family} {view}"
                    )
                joined["bin"] = plddt_bin(joined.plddt_normalized)
                for name in BIN_LABELS:
                    values = joined.loc[joined["bin"].eq(name)]
                    records.append(
                        {
                            "seed": seed,
                            "family": family,
                            "feature_view": view,
                            "plddt_bin": name,
                            **bin_metrics(
                                values.dataset_label_catalog.to_numpy(),
                                values.probability.to_numpy(),
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _dataset(catalog: pd.DataFrame, seed: int) -> Path:
    split = pd.read_parquet(SOURCE / f"split_{seed}.parquet")[["protein_id", "split"]]
    frame = catalog.drop(columns="split", errors="ignore").merge(
        split, on="protein_id", how="inner", validate="one_to_one"
    )
    if len(frame) != len(catalog) or set(frame.split) != {"train", "val", "test"}:
        raise ValueError(f"saved split {seed} does not cover the pLDDT subset")
    path = OUTPUT / f"split_{seed}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def covariate_matrix(frame: pd.DataFrame, mode: str) -> tuple[np.ndarray, tuple[str, ...]]:
    if mode == "plddt":
        return frame.alphafold_mean_plddt.to_numpy(dtype=np.float32).reshape(-1, 1), ("mean_plddt",)
    sequence, names = sequence_feature_matrix(frame)
    plddt = frame.alphafold_mean_plddt.to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.concatenate((sequence, plddt), axis=1), (*names, "mean_plddt")


class _FastResidualizer:
    """Train-only OLS projection with a compact, fast predict interface."""

    def __init__(self, scaler: StandardScaler, coefficients: np.ndarray) -> None:
        self.scaler = scaler
        self.coefficients = coefficients

    @property
    def named_steps(self) -> dict[str, object]:
        """Expose the historical fitted-model inspection interface."""
        regression = type("RegressionView", (), {})()
        regression.coef_ = self.coefficients[1:].T
        regression.intercept_ = self.coefficients[0]
        return {"scale": self.scaler, "regression": regression}

    def predict(self, covariates: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(covariates).astype(np.float64, copy=False)
        design = np.column_stack((np.ones(len(scaled)), scaled))
        return design @ self.coefficients


def residualize_embeddings(
    embeddings: np.ndarray, covariates: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, _FastResidualizer]:
    """Fit H ~ Z on train only, then return H - H-hat for all split rows."""
    scaler = StandardScaler().fit(covariates[train_mask])
    scaled_train = scaler.transform(covariates[train_mask]).astype(np.float64, copy=False)
    design_train = np.column_stack((np.ones(len(scaled_train)), scaled_train))
    target_train = embeddings[train_mask].astype(np.float64, copy=False)
    with threadpool_limits(limits=2):
        # ``lstsq`` remains well-defined when sequence-derived covariates are
        # collinear or constant in a small split. Normal equations previously
        # failed on singular designs and are less numerically stable.
        coefficients, _, _, _ = np.linalg.lstsq(design_train, target_train, rcond=None)
    residualizer = _FastResidualizer(scaler, coefficients)
    predicted = np.asarray(residualizer.predict(covariates), dtype=np.float32)
    np.subtract(embeddings, predicted, out=predicted)
    return predicted, residualizer


def _residual_views(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    residualizers: dict[str, _FastResidualizer] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, _FastResidualizer]]:
    """Build both residual views, repairing incomplete cached fit objects."""
    residualizers = dict(residualizers or {})
    views: dict[str, np.ndarray] = {}
    for mode, view in (("plddt", "residual_plddt"), ("full", "residual_full_covariates")):
        covariates, _ = covariate_matrix(frame, mode)
        fitted = residualizers.get(mode)
        if fitted is None:
            residual, fitted = residualize_embeddings(embeddings, covariates, train_mask)
            residualizers[mode] = fitted
        else:
            predicted = np.asarray(fitted.predict(covariates), dtype=np.float32)
            residual = embeddings - predicted
        if residual.shape != embeddings.shape or not np.isfinite(residual).all():
            raise ValueError(f"invalid {view} residual shape or values")
        views[view] = residual
    return views, residualizers


def _summary(values: pd.DataFrame, groups: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    return values.groupby(groups, as_index=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
        splits=("seed", "count"),
    )


def residual_permutation_tests(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in ("linear", "tree"):
        for residual_view in ("residual_plddt", "residual_full_covariates"):
            paired = records.loc[records.family.eq(family)].pivot(
                index="seed",
                columns="feature_view",
                values=["test_accuracy", "test_auroc", "test_auprc"],
            )
            for metric in ("test_accuracy", "test_auroc", "test_auprc"):
                raw = paired[(metric, f"raw_{REPRESENTATION_NAME}")]
                residual = paired[(metric, residual_view)]
                test = paired_sign_flip_test((residual - raw).to_numpy())
                rows.append(
                    {
                        "family": family,
                        "comparison": f"{residual_view}_minus_raw_{REPRESENTATION_NAME}",
                        "metric": metric.removeprefix("test_"),
                        "raw_mean": raw.mean(),
                        "residual_mean": residual.mean(),
                        "mean_difference": (residual - raw).mean(),
                        "paired_p": test["permutation_p_two_sided"],
                        "paired_method": test["permutation_method"],
                        "n_splits": len(raw),
                    }
                )
    result = pd.DataFrame(rows)
    result["paired_fdr"] = benjamini_hochberg(result.paired_p.to_numpy())
    return result


def _stratum_models(
    subset: pd.DataFrame,
    subset_path: Path,
    destination: Path,
    stratum: str,
    seed: int,
    homology_ids: set[str] | None = None,
    residualizers: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    covariates, covariate_names = covariate_matrix(subset, "full")
    embeddings, embedding_names = _load_features(
        subset_path, f"{REPRESENTATION_NAME}_single", "linear", EMBEDDING_MANIFEST
    )
    train_mask = subset.split.eq("train").to_numpy()
    residual_views, residualizers = _residual_views(embeddings, subset, train_mask, residualizers)
    records = []
    for view in STRATIFIED_VIEWS:
        if view == "metadata":
            features, names = covariates, covariate_names
        elif view == "embedding":
            features, names = embeddings, embedding_names
        elif view == "combined":
            features = np.concatenate((covariates, embeddings), axis=1)
            names = (*covariate_names, *embedding_names)
        else:
            features = residual_views[view]
            names = tuple(f"{view}_{index}" for index in range(features.shape[1]))
        for family in ("linear", "tree"):
            model_dir = destination / f"{family}_{view}"
            metrics_path = model_dir / "metrics.json"
            metrics = (
                json.loads(metrics_path.read_text())
                if metrics_path.is_file()
                else run_benchmark(
                    subset_path,
                    model_dir,
                    BenchmarkConfig(
                        family=family,
                        random_seed=seed,
                        device="cpu",
                        search="fast",
                        save_model=False,
                    ),
                    features=features,
                    feature_names=names,
                )
            )
            record = {
                "seed": seed,
                "plddt_stratum": stratum,
                "family": family,
                "feature_view": view,
                "sample_count": len(subset),
                **{f"test_{key}": metrics[key] for key in ("accuracy", "auroc", "auprc")},
            }
            if homology_ids is not None:
                predictions = pd.read_parquet(model_dir / "test_predictions.parquet")
                filtered = predictions.loc[predictions.protein_id.isin(homology_ids)]
                if filtered.dataset_label.nunique() != 2:
                    raise ValueError(f"homology subset lacks both classes for {stratum}")
                homology = classification_metrics(
                    filtered.dataset_label.to_numpy(), filtered.probability.to_numpy()
                )
                record.update(
                    homology35_sample_count=len(filtered),
                    **{
                        f"homology35_{key}": homology[key] for key in ("accuracy", "auroc", "auprc")
                    },
                )
            records.append(record)
        if view in ("combined", "residual_plddt", "residual_full_covariates"):
            del features
            gc.collect()
    del covariates, embeddings, residual_views
    gc.collect()
    return records


def _validate_stratum(subset: pd.DataFrame, description: str) -> None:
    counts = subset.groupby("split").dataset_label.nunique()
    if set(counts.index) != {"train", "val", "test"} or not counts.eq(2).all():
        raise ValueError(f"{description} does not contain both classes per split")


def run_stratified_benchmark(catalog: pd.DataFrame) -> None:
    """Retrain paired models within fixed AlphaFold confidence strata."""
    STRATIFIED_OUTPUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    models_per_split = len(SELECTED_STRATA) * len(STRATIFIED_VIEWS) * 2
    total = (len(SEEDS) + 1) * models_per_split

    def checkpoint() -> None:
        pd.DataFrame(records).to_csv(STRATIFIED_OUTPUT / "live_metrics.csv", index=False)
        _write_json(
            STRATIFIED_OUTPUT / "progress.json",
            {
                "status": "running",
                "updated_at_central": _now(),
                "completed_models": len(records),
                "total_models": total,
            },
        )

    for seed in SEEDS:
        dataset = _dataset(catalog, seed)
        frame = pd.read_parquet(dataset)
        frame["plddt_stratum"] = plddt_stratum(frame.alphafold_mean_plddt)
        for stratum in SELECTED_STRATA:
            subset = frame.loc[frame.plddt_stratum.eq(stratum)].copy()
            _validate_stratum(subset, f"{stratum} seed {seed}")
            subset_path = STRATIFIED_OUTPUT / f"split_{seed}_{stratum}.parquet"
            subset.to_parquet(subset_path, index=False)
            records.extend(
                _stratum_models(
                    subset,
                    subset_path,
                    STRATIFIED_OUTPUT / f"seed_{seed}" / stratum,
                    stratum,
                    seed,
                    residualizers=joblib.load(OUTPUT / f"seed_{seed}_residualizers.joblib"),
                )
            )
            checkpoint()

    if catalog.groupby("homology_group_id").split.nunique().max() != 1:
        raise ValueError("homology group crosses the canonical split")
    homology_ids = set(catalog.loc[catalog.split.eq("test"), "protein_id"])
    canonical = catalog.copy()
    canonical["plddt_stratum"] = plddt_stratum(canonical.alphafold_mean_plddt)
    canonical_records = []
    for stratum in SELECTED_STRATA:
        subset = canonical.loc[canonical.plddt_stratum.eq(stratum)].copy()
        _validate_stratum(subset, f"canonical {stratum}")
        subset_path = STRATIFIED_OUTPUT / f"canonical_{stratum}.parquet"
        subset.to_parquet(subset_path, index=False)
        values = _stratum_models(
            subset,
            subset_path,
            STRATIFIED_OUTPUT / "canonical" / stratum,
            stratum,
            42,
            homology_ids,
        )
        canonical_records.extend(values)
        records.extend(values)
        checkpoint()
    pd.DataFrame(canonical_records).to_csv(
        STRATIFIED_OUTPUT / "homology_grouped_test_metrics.csv", index=False
    )

    results = pd.DataFrame(records[: len(SEEDS) * models_per_split])
    results.to_csv(STRATIFIED_OUTPUT / "all_metrics.csv", index=False)
    results.groupby(["plddt_stratum", "family", "feature_view"], as_index=False).agg(
        splits=("seed", "count"),
        sample_count=("sample_count", "first"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        auroc_mean=("test_auroc", "mean"),
        auroc_std=("test_auroc", "std"),
        auprc_mean=("test_auprc", "mean"),
        auprc_std=("test_auprc", "std"),
    ).to_csv(STRATIFIED_OUTPUT / "summary.csv", index=False)
    comparisons = []
    for stratum in SELECTED_STRATA:
        for family in ("linear", "tree"):
            values = results.loc[results.plddt_stratum.eq(stratum) & results.family.eq(family)]
            for view in STRATIFIED_VIEWS[1:]:
                paired = values.loc[values.feature_view.isin(("metadata", view))].pivot(
                    index="seed",
                    columns="feature_view",
                    values=["test_accuracy", "test_auroc", "test_auprc"],
                )
                for metric in ("test_accuracy", "test_auroc", "test_auprc"):
                    difference = paired[(metric, view)] - paired[(metric, "metadata")]
                    test = paired_sign_flip_test(difference.to_numpy())
                    comparisons.append(
                        {
                            "plddt_stratum": stratum,
                            "family": family,
                            "comparison": f"{view}_minus_metadata",
                            "metric": metric.removeprefix("test_"),
                            "mean_difference": (
                                paired[(metric, view)] - paired[(metric, "metadata")]
                            ).mean(),
                            "paired_p": test["permutation_p_two_sided"],
                            "paired_method": test["permutation_method"],
                            "n_splits": len(paired),
                        }
                    )
    comparison_table = pd.DataFrame(comparisons)
    # An embedding-only run intentionally has no metadata baseline to compare
    # against.  Still emit the empty report rather than failing after all
    # requested models have completed.
    if not comparison_table.empty:
        comparison_table["paired_fdr"] = benjamini_hochberg(
            comparison_table.paired_p.to_numpy()
        )
    comparison_table.to_csv(STRATIFIED_OUTPUT / "paired_permutation_tests.csv", index=False)
    _write_json(
        STRATIFIED_OUTPUT / "progress.json",
        {
            "status": "completed",
            "updated_at_central": _now(),
            "completed_models": len(records),
            "total_models": total,
        },
    )


def _report(bin_values: pd.DataFrame, residuals: pd.DataFrame, completed: int) -> None:
    bin_summary = (
        _summary(
            bin_values, ["family", "feature_view", "plddt_bin"], ("accuracy", "auroc", "auprc")
        )
        if len(bin_values)
        else pd.DataFrame()
    )
    residual_summary = (
        _summary(
            residuals, ["family", "feature_view"], ("test_accuracy", "test_auroc", "test_auprc")
        )
        if len(residuals)
        else pd.DataFrame()
    )
    tests = (
        residual_permutation_tests(residuals)
        if len(residuals) == 6 * len(SEEDS)
        else pd.DataFrame()
    )
    html = (
        "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='30'><title>pLDDT follow-up</title>"
        "<style>body{font:15px system-ui;max-width:1400px;margin:32px auto;color:#17324d}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d8e0e8;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}</style>"
        "<h1>pLDDT shortcut-risk follow-up</h1>"
        "<p>Bin results are inference-only from the completed paired confounder models. Residualizers fit only training rows.</p>"
        f"<p>Residual embedding models completed: {completed}/{6 * len(SEEDS)}</p>"
        "<h2>pLDDT-bin inference</h2>"
        + (
            bin_summary.to_html(index=False, float_format=lambda value: f"{value:.5f}")
            if len(bin_summary)
            else "<p>Skipped during the residual-only stage. The separate low/medium/high retraining benchmark runs after residualization completes.</p>"
        )
        + "<h2>Raw and residualized pooled ESMFold embeddings</h2>"
        + residual_summary.to_html(index=False, float_format=lambda value: f"{value:.5f}")
        + "<h2>Paired residualization tests</h2>"
        + tests.to_html(index=False, float_format=lambda value: f"{value:.5g}")
    )
    (OUTPUT / "live_report.html").write_text(html)


def main(
    residual_only: bool = False,
    wait_for_source: bool = False,
    stratified_only: bool = False,
    wait_for_residual: bool = False,
    wait_for_stratified: bool = False,
    seeds: tuple[int, ...] = SEEDS,
) -> None:
    global SEEDS
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique saved split seeds")
    SEEDS = tuple(seeds)
    if stratified_only:
        if wait_for_residual:
            _wait_for_residual()
        _require_split_inputs()
        run_stratified_benchmark(_catalog())
        return
    if wait_for_source:
        _wait_for_source()
    if wait_for_stratified:
        _wait_for_stratified()
    if residual_only:
        _require_split_inputs()
    else:
        _require_source_complete()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_readme(residual_only)
    catalog = _catalog()
    bin_values = pd.DataFrame()
    if not residual_only:
        bin_values = run_bin_inference(catalog)
        bin_values.to_csv(OUTPUT / "bin_metrics.csv", index=False)
        _summary(
            bin_values, ["family", "feature_view", "plddt_bin"], ("accuracy", "auroc", "auprc")
        ).to_csv(OUTPUT / "bin_summary.csv", index=False)
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        dataset = _dataset(catalog, seed)
        frame = pd.read_parquet(dataset)
        embeddings, names = _load_features(
            dataset, f"{REPRESENTATION_NAME}_single", "linear", EMBEDDING_MANIFEST
        )
        train_mask = frame.split.eq("train").to_numpy()
        residualizers = {}
        for view, mode in (
            (f"raw_{REPRESENTATION_NAME}", None),
            ("residual_plddt", "plddt"),
            ("residual_full_covariates", "full"),
        ):
            if mode is None:
                features, feature_names = embeddings, names
            else:
                covariates, _ = covariate_matrix(frame, mode)
                features, residualizer = residualize_embeddings(embeddings, covariates, train_mask)
                feature_names = tuple(f"{view}_{index}" for index in range(features.shape[1]))
                residualizers[mode] = residualizer
            for family in ("linear", "tree"):
                destination = OUTPUT / f"seed_{seed}" / f"{family}_{view}"
                metrics_path = destination / "metrics.json"
                metrics = (
                    json.loads(metrics_path.read_text())
                    if metrics_path.exists()
                    else run_benchmark(
                        dataset,
                        destination,
                        BenchmarkConfig(
                            family=family,
                            random_seed=seed,
                            device="cpu",
                            search="fast",
                            save_model=False,
                        ),
                        features=features,
                        feature_names=feature_names,
                    )
                )
                records.append(
                    {
                        "seed": seed,
                        "family": family,
                        "feature_view": view,
                        **{f"test_{key}": metrics[key] for key in ("accuracy", "auroc", "auprc")},
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "residual_model_complete",
                            "seed": seed,
                            "family": family,
                            "feature_view": view,
                            "completed_models": len(records),
                            "total_models": 6 * len(SEEDS),
                        }
                    ),
                    flush=True,
                )
                residuals = pd.DataFrame(records)
                residuals.to_csv(OUTPUT / "residual_metrics.csv", index=False)
                _write_json(
                    OUTPUT / "progress.json",
                    {
                        "status": "running",
                        "updated_at_central": _now(),
                        "bin_inference": "skipped_residual_only" if residual_only else "completed",
                        "completed_residual_models": len(records),
                        "total_residual_models": 6 * len(SEEDS),
                    },
                )
                _report(bin_values, residuals, len(records))
            if mode is not None:
                del features, covariates
                gc.collect()
        joblib.dump(residualizers, OUTPUT / f"seed_{seed}_residualizers.joblib")
        del embeddings, frame, residualizers
        gc.collect()
    residuals = pd.DataFrame(records)
    residuals.to_csv(OUTPUT / "residual_metrics.csv", index=False)
    residual_permutation_tests(residuals).to_csv(
        OUTPUT / "residual_permutation_tests.csv", index=False
    )
    _write_json(
        OUTPUT / "progress.json",
        {
            "status": "completed",
            "updated_at_central": _now(),
            "bin_inference": "skipped_residual_only" if residual_only else "completed",
            "completed_residual_models": len(records),
            "total_residual_models": 6 * len(SEEDS),
        },
    )
    _report(bin_values, residuals, len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help="Run only residualized embedding models, without duplicate pLDDT-bin inference.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Saved split seeds to use. Defaults to four established splits.",
    )
    parser.add_argument(
        "--wait-for-source",
        action="store_true",
        help="Wait for the active ten-split benchmark before starting.",
    )
    parser.add_argument(
        "--stratified-only",
        action="store_true",
        help="Run only the low, medium, and high pLDDT retraining benchmark.",
    )
    parser.add_argument(
        "--wait-for-residual",
        action="store_true",
        help="Wait up to 24 hours for the residual benchmark before stratified training.",
    )
    parser.add_argument(
        "--wait-for-stratified",
        action="store_true",
        help="Wait up to 48 hours for stratified training before residual expansion.",
    )
    parser.add_argument("--catalog", type=Path, default=ROOT / "homology35_seed42/catalog.parquet")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--embedding-manifest", type=Path, default=EMBEDDING_MANIFEST)
    parser.add_argument(
        "--representation-name", choices=representation_choices(), default="esmfold"
    )
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    ROOT = args.catalog.parent.parent
    CATALOG_PATH = args.catalog
    SOURCE = args.source
    OUTPUT = args.output
    STRATIFIED_OUTPUT = OUTPUT / "stratified"
    EMBEDDING_MANIFEST = args.embedding_manifest
    REPRESENTATION_NAME = args.representation_name
    EXPECTED_ROWS = args.expected_rows
    main(
        residual_only=args.residual_only,
        wait_for_source=args.wait_for_source,
        stratified_only=args.stratified_only,
        wait_for_residual=args.wait_for_residual,
        wait_for_stratified=args.wait_for_stratified,
        seeds=tuple(args.seeds),
    )
