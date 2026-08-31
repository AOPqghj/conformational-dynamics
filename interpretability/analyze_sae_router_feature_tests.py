"""Test frozen SAE features against router labels, confounders, motifs, and ablations.

The SAE is fit only on Seed-42 train proteins.  Validation selects candidates;
the frozen Seed-42 test partition is used once for the primary label and
classifier-ablation results.  Feature 2722 is retained as an explicitly
exploratory transition-selected candidate.
"""

# ruff: noqa: E402 - direct execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/protein-state-router-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/protein-state-router-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from protein_state_router.representations.registry import representation_choices

from interpretability.analyze_sae_transition_residue_associations import (
    bh_adjust,
    load_frozen_sae,
    sha256_file,
)
from interpretability.contracts import load_residue_matrix, pool_residue_matrix
from interpretability.model import FrozenPooledModel

DEFAULT_CATALOG = Path(
    "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
)
DEFAULT_SAE = Path("ml/results/homology35_frozen_saes/esmfold_matrix_topk64_seed42")
DEFAULT_MODELS = Path("ml/results/homology35_frozen_models")
DEFAULT_ANNOTATIONS = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/annotated_catalog.parquet"
)
DEFAULT_DISPLACEMENT = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/"
    "homology35_dynamic_transition_residue_ca_displacements.csv"
)
DEFAULT_PRS = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/homology35_dynamic_transition_prs_scores.csv"
)
DEFAULT_OUTPUT = Path("interpretability/results/homology35_sae_router_feature_tests")
AGGREGATES = ("mean_activation", "max_activation", "fraction_active")
FEATURES = 4096
EXPLORATORY_FEATURE = 2722
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class RunConfig:
    """Stable configuration for a resumable router-feature analysis."""

    seed: int = 42
    device: str = "auto"
    feature_batch_size: int = 4
    random_controls: int = 100
    bootstrap_draws: int = 2_000
    motif_half_window: int = 7
    fdr_alpha: float = 0.05
    run_subtype: bool = False
    representation_name: str = "esmfold"

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def status(event: str, **details: object) -> None:
    """Emit compact line-buffered progress for terminal monitoring."""
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; use --device auto or --device cpu")
    return torch.device(device)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _open_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.is_file():
        array = np.lib.format.open_memmap(path, mode="r+")
        if array.shape != shape:
            raise ValueError(f"checkpoint shape differs for {path.name}: {array.shape} != {shape}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _require_catalog(catalog: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
        "dataset_label",
        "source_dataset",
        "split",
        "embedding_path",
    }
    if required - set(catalog) or catalog.empty or catalog.protein_id.duplicated().any():
        raise ValueError("expected a non-empty unique frozen Seed-42 catalog")
    if set(catalog.split) != {"train", "val", "test"}:
        raise ValueError("frozen catalog must declare train, val, and test partitions")


def _load_centered_latents(
    row: object, model: torch.nn.Module, center: np.ndarray, device: torch.device
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    matrix = load_residue_matrix(
        Path(row.embedding_path),
        protein_id=row.protein_id,
        sequence=row.sequence,
        sequence_sha256=row.sequence_sha256,
        sequence_length=int(row.sequence_length),
        expected_width=len(center),
    )
    centered = torch.from_numpy(matrix.astype(np.float32, copy=False) - center).to(device)
    with torch.inference_mode():
        reconstruction, latents = model(centered)
    return matrix, reconstruction, latents


def aggregate_partition(
    partition: pd.DataFrame,
    name: str,
    model: torch.nn.Module,
    center: np.ndarray,
    device: torch.device,
    output: Path,
    config: RunConfig,
) -> dict[str, np.ndarray]:
    """Stream one partition into compact protein-by-feature aggregate arrays."""
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    shape = (len(partition), FEATURES)
    arrays = {
        aggregate: _open_memmap(work / f"{name}_{aggregate}.npy", shape) for aggregate in AGGREGATES
    }
    progress_path = work / f"{name}_aggregate_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if progress and progress.get("config_hash") != config.config_hash:
        raise ValueError(f"{name} aggregate checkpoint has a different configuration")
    completed = int(progress.get("completed", 0))
    for index, row in enumerate(partition.itertuples(index=False)):
        if index < completed:
            continue
        _, _, latents = _load_centered_latents(row, model, center, device)
        arrays["mean_activation"][index] = latents.mean(dim=0).cpu().numpy()
        arrays["max_activation"][index] = latents.max(dim=0).values.cpu().numpy()
        arrays["fraction_active"][index] = (latents > 0).float().mean(dim=0).cpu().numpy()
        for values in arrays.values():
            values.flush()
        completed = index + 1
        atomic_json(
            progress_path,
            {"config_hash": config.config_hash, "completed": completed, "total": len(partition)},
        )
        if completed == 1 or completed % 25 == 0 or completed == len(partition):
            status(
                "aggregating_features", partition=name, completed=completed, total=len(partition)
            )
    result = {name: np.asarray(values) for name, values in arrays.items()}
    np.savez_compressed(output / f"{name}_feature_aggregates.npz", **result)
    atomic_json(
        progress_path,
        {
            "config_hash": config.config_hash,
            "completed": len(partition),
            "total": len(partition),
            "status": "complete",
        },
    )
    return result


def enrichment_table(
    aggregates: dict[str, np.ndarray], labels: np.ndarray, partition: str
) -> pd.DataFrame:
    """Test every feature and aggregation at the protein, never residue, unit."""
    labels = np.asarray(labels, dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("enrichment requires both router labels")
    positive = labels == 1
    negative = ~positive
    rows: list[dict[str, object]] = []
    for aggregate, values in aggregates.items():
        for feature in range(values.shape[1]):
            dynamic = values[positive, feature]
            static = values[negative, feature]
            if np.ptp(values[:, feature]) == 0:
                statistic = len(dynamic) * len(static) / 2
                p_value = 1.0
                rank_biserial = 0.0
            else:
                statistic, p_value = mannwhitneyu(dynamic, static, alternative="two-sided")
                p_value = float(p_value) if np.isfinite(p_value) else 1.0
                rank_biserial = 2 * float(statistic) / (len(dynamic) * len(static)) - 1
            rows.append(
                {
                    "partition": partition,
                    "feature_id": feature,
                    "aggregate": aggregate,
                    "dynamic_mean": float(dynamic.mean()),
                    "static_mean": float(static.mean()),
                    "mean_difference_dynamic_minus_static": float(dynamic.mean() - static.mean()),
                    "rank_biserial_dynamic_vs_static": rank_biserial,
                    "mann_whitney_p": float(p_value),
                    "dynamic_proteins": int(positive.sum()),
                    "static_proteins": int(negative.sum()),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = bh_adjust(result.mann_whitney_p.to_numpy(dtype=float))
    return result


def select_candidates(validation: pd.DataFrame, fdr_alpha: float) -> pd.DataFrame:
    """Lock a balanced validation-selected candidate set plus exploratory F2722."""
    best = validation.assign(
        abs_effect=validation.rank_biserial_dynamic_vs_static.abs()
    ).sort_values(["feature_id", "fdr", "abs_effect"], ascending=[True, True, False])
    best = best.groupby("feature_id", as_index=False).first()
    selected: list[pd.DataFrame] = []
    for direction, count in (("multistate_enriched", 10), ("static_enriched", 10)):
        sign = (
            best.rank_biserial_dynamic_vs_static.gt(0)
            if direction.startswith("multi")
            else best.rank_biserial_dynamic_vs_static.lt(0)
        )
        ranked = (
            best.loc[sign].sort_values(["fdr", "abs_effect"], ascending=[True, False]).head(count)
        )
        ranked = ranked.assign(
            selection_reason=direction, validation_fdr_significant=ranked.fdr.le(fdr_alpha)
        )
        selected.append(ranked)
    result = pd.concat(selected, ignore_index=True).drop_duplicates("feature_id")
    if len(result) < 20:
        fallback = (
            best.loc[~best.feature_id.isin(result.feature_id)]
            .sort_values(["fdr", "abs_effect"], ascending=[True, False])
            .head(20 - len(result))
        )
        result = pd.concat(
            [
                result,
                fallback.assign(
                    selection_reason="validation_fallback",
                    validation_fdr_significant=fallback.fdr.le(fdr_alpha),
                ),
            ],
            ignore_index=True,
        )
    if EXPLORATORY_FEATURE not in set(result.feature_id):
        external = best.loc[best.feature_id.eq(EXPLORATORY_FEATURE)].copy()
        external["selection_reason"] = "exploratory_transition_feature"
        external["validation_fdr_significant"] = external.fdr.le(fdr_alpha)
        result = pd.concat([result, external], ignore_index=True)
    return result.sort_values(["selection_reason", "feature_id"]).reset_index(drop=True)


def select_random_controls(
    fraction_active: np.ndarray, candidate_ids: list[int], count: int, seed: int
) -> pd.DataFrame:
    """Sample shared controls stratified by validation activation-frequency deciles."""
    frequency = fraction_active.mean(axis=0)
    bins = pd.qcut(pd.Series(frequency), q=10, duplicates="drop", labels=False).to_numpy()
    candidates = np.asarray(candidate_ids, dtype=int)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    candidate_bins = bins[candidates]
    for bin_value in np.unique(candidate_bins):
        target = max(1, round(count * float((candidate_bins == bin_value).mean())))
        pool = np.flatnonzero((bins == bin_value) & ~np.isin(np.arange(FEATURES), candidates))
        if len(pool):
            selected.extend(rng.choice(pool, size=min(target, len(pool)), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(FEATURES), np.asarray([*candidates, *selected]))
    if len(selected) < count:
        selected.extend(
            rng.choice(
                remaining, size=min(count - len(selected), len(remaining)), replace=False
            ).tolist()
        )
    selected = list(dict.fromkeys(selected))
    if len(selected) > count:
        selected = rng.choice(selected, size=count, replace=False).tolist()
    return pd.DataFrame(
        {
            "feature_id": selected,
            "activation_frequency_validation": frequency[selected],
            "activation_frequency_bin": bins[selected],
            "role": "shared_random_control",
        }
    )


def _pool_torch(values: torch.Tensor) -> np.ndarray:
    pooled = torch.cat(
        (values.mean(dim=1), values.std(dim=1, unbiased=False), values.max(dim=1).values), dim=1
    )
    return pooled.detach().cpu().numpy().astype(np.float32)


def ablate_reconstruction(
    reconstruction: torch.Tensor,
    latents: torch.Tensor,
    decoder_vectors: torch.Tensor,
) -> torch.Tensor:
    """Return reconstructions with one selected latent zeroed per batch row."""
    if reconstruction.ndim != 2 or latents.ndim != 2 or decoder_vectors.ndim != 2:
        raise ValueError("reconstruction, latents, and decoder vectors must be matrices")
    if (
        reconstruction.shape[0] != latents.shape[0]
        or reconstruction.shape[1] != decoder_vectors.shape[1]
    ):
        raise ValueError("ablation tensors have incompatible dimensions")
    if latents.shape[1] != decoder_vectors.shape[0]:
        raise ValueError("selected latent count differs from decoder vectors")
    return reconstruction.unsqueeze(0) - latents.T.unsqueeze(-1) * decoder_vectors.unsqueeze(1)


def _score_features(model: FrozenPooledModel, features: np.ndarray) -> np.ndarray:
    scores = model.model.predict_proba(features)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError(f"frozen {model.name} returned non-finite probabilities")
    return scores.astype(np.float32)


def evaluate_ablations(
    test: pd.DataFrame,
    feature_ids: list[int],
    model: torch.nn.Module,
    center: np.ndarray,
    device: torch.device,
    output: Path,
    config: RunConfig,
) -> dict[str, np.ndarray]:
    """Read every test matrix once and score all selected feature ablations in batches."""
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "ablation_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if progress and progress.get("config_hash") != config.config_hash:
        raise ValueError("ablation checkpoint has a different configuration")
    linear = FrozenPooledModel(DEFAULT_MODELS, f"{config.representation_name}_single_linear")
    tree = FrozenPooledModel(DEFAULT_MODELS, f"{config.representation_name}_single_tree")
    feature_ids = list(feature_ids)
    n = len(test)
    shape = (len(feature_ids), n)
    predictions = {
        "linear_feature": _open_memmap(work / "linear_feature_predictions.npy", shape),
        "tree_feature": _open_memmap(work / "tree_feature_predictions.npy", shape),
        "linear_native": _open_memmap(work / "linear_native_predictions.npy", (n,)),
        "tree_native": _open_memmap(work / "tree_native_predictions.npy", (n,)),
        "linear_reconstruction": _open_memmap(work / "linear_reconstruction_predictions.npy", (n,)),
        "tree_reconstruction": _open_memmap(work / "tree_reconstruction_predictions.npy", (n,)),
    }
    completed = int(progress.get("completed", 0))
    feature_tensor = torch.tensor(feature_ids, dtype=torch.long, device=device)
    center_tensor = torch.from_numpy(center).to(device)
    decoder = model.decoder.weight[:, feature_tensor].T.detach()
    for index, row in enumerate(test.itertuples(index=False)):
        if index < completed:
            continue
        matrix, reconstruction, latents = _load_centered_latents(row, model, center, device)
        reconstruction = reconstruction + center_tensor
        native = pool_residue_matrix(matrix).reshape(1, -1)
        reconstruction_features = _pool_torch(reconstruction.unsqueeze(0))
        predictions["linear_native"][index] = _score_features(linear, native)[0]
        predictions["tree_native"][index] = _score_features(tree, native)[0]
        predictions["linear_reconstruction"][index] = _score_features(
            linear, reconstruction_features
        )[0]
        predictions["tree_reconstruction"][index] = _score_features(tree, reconstruction_features)[
            0
        ]
        for start in range(0, len(feature_ids), config.feature_batch_size):
            stop = min(start + config.feature_batch_size, len(feature_ids))
            ids = feature_tensor[start:stop]
            ablated = ablate_reconstruction(reconstruction, latents[:, ids], decoder[start:stop])
            pooled = _pool_torch(ablated)
            predictions["linear_feature"][start:stop, index] = _score_features(linear, pooled)
            predictions["tree_feature"][start:stop, index] = _score_features(tree, pooled)
        for values in predictions.values():
            values.flush()
        completed = index + 1
        atomic_json(
            progress_path, {"config_hash": config.config_hash, "completed": completed, "total": n}
        )
        if completed == 1 or completed % 25 == 0 or completed == n:
            status("ablating_features", completed=completed, total=n, features=len(feature_ids))
    result = {name: np.asarray(values) for name, values in predictions.items()}
    np.savez_compressed(
        output / "ablation_predictions.npz", feature_ids=np.asarray(feature_ids), **result
    )
    atomic_json(
        progress_path,
        {"config_hash": config.config_hash, "completed": n, "total": n, "status": "complete"},
    )
    return result


def _metric_row(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def paired_metric_drop_interval(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    ablated_scores: np.ndarray,
    draws: int,
    seed: int,
    metric: str,
) -> tuple[float, float]:
    """Stratified protein bootstrap interval for a paired metric drop."""
    metric_function = roc_auc_score if metric == "auroc" else average_precision_score
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        values[draw] = metric_function(labels[indices], baseline_scores[indices]) - metric_function(
            labels[indices], ablated_scores[indices]
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def ablation_metrics(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
    feature_ids: list[int],
    candidate_ids: set[int],
    controls: set[int],
    config: RunConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in ("linear", "tree"):
        baseline = predictions[f"{family}_native"]
        reconstruction = predictions[f"{family}_reconstruction"]
        native_auc, native_ap = _metric_row(labels, baseline)
        reconstruction_auc, reconstruction_ap = _metric_row(labels, reconstruction)
        rows.extend(
            [
                {
                    "model": family,
                    "condition": "native",
                    "feature_id": np.nan,
                    "auroc": native_auc,
                    "auprc": native_ap,
                    "delta_auroc_from_reconstruction": native_auc - reconstruction_auc,
                    "delta_auprc_from_reconstruction": native_ap - reconstruction_ap,
                    "auroc_drop_from_reconstruction": reconstruction_auc - native_auc,
                    "auprc_drop_from_reconstruction": reconstruction_ap - native_ap,
                },
                {
                    "model": family,
                    "condition": "full_sae_reconstruction",
                    "feature_id": np.nan,
                    "auroc": reconstruction_auc,
                    "auprc": reconstruction_ap,
                    "delta_auroc_from_reconstruction": 0.0,
                    "delta_auprc_from_reconstruction": 0.0,
                    "auroc_drop_from_reconstruction": 0.0,
                    "auprc_drop_from_reconstruction": 0.0,
                },
            ]
        )
        for index, feature in enumerate(feature_ids):
            auc, ap = _metric_row(labels, predictions[f"{family}_feature"][index])
            row = {
                "model": family,
                "condition": "candidate" if feature in candidate_ids else "shared_random_control",
                "feature_id": feature,
                "auroc": auc,
                "auprc": ap,
                "delta_auroc_from_reconstruction": auc - reconstruction_auc,
                "delta_auprc_from_reconstruction": ap - reconstruction_ap,
                "auroc_drop_from_reconstruction": reconstruction_auc - auc,
                "auprc_drop_from_reconstruction": reconstruction_ap - ap,
            }
            if feature in candidate_ids:
                auc_low, auc_high = paired_metric_drop_interval(
                    labels,
                    reconstruction,
                    predictions[f"{family}_feature"][index],
                    config.bootstrap_draws,
                    config.seed + feature,
                    "auroc",
                )
                ap_low, ap_high = paired_metric_drop_interval(
                    labels,
                    reconstruction,
                    predictions[f"{family}_feature"][index],
                    config.bootstrap_draws,
                    config.seed + 10_000 + feature,
                    "auprc",
                )
                row["auroc_drop_ci_low"] = auc_low
                row["auroc_drop_ci_high"] = auc_high
                row["auprc_drop_ci_low"] = ap_low
                row["auprc_drop_ci_high"] = ap_high
            rows.append(row)
    result = pd.DataFrame(rows)
    for family in ("linear", "tree"):
        for metric in ("auroc_drop_from_reconstruction", "auprc_drop_from_reconstruction"):
            control = result.loc[
                (result.model == family) & result.feature_id.isin(controls), metric
            ]
            mask = (result.model == family) & result.feature_id.isin(candidate_ids)
            result.loc[mask, f"{metric}_control_percentile"] = result.loc[mask, metric].map(
                lambda value, baseline=control: float((baseline <= value).mean())
            )
            result.loc[mask, f"{metric}_control_p"] = result.loc[mask, metric].map(
                lambda value, baseline=control: float(
                    (1 + np.count_nonzero(baseline.to_numpy() >= value)) / (len(baseline) + 1)
                )
            )
            result.loc[mask, f"{metric}_control_fdr"] = bh_adjust(
                result.loc[mask, f"{metric}_control_p"].to_numpy(dtype=float)
            )
    return result


def _metadata_fields(catalog: pd.DataFrame, annotations_path: Path) -> pd.DataFrame:
    result = catalog[
        ["protein_id", "sequence_length", "source_dataset", "alphafold_mean_plddt"]
    ].copy()
    result["family_id"] = pd.NA
    result["disorder_or_heterogeneity"] = pd.NA
    for index, value in enumerate(catalog.source_metadata_json):
        try:
            records = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            records = []
        records = records if isinstance(records, list) else [records]
        for record in records:
            if isinstance(record, dict):
                for field in ("family_id", "disorder_or_heterogeneity"):
                    if pd.isna(result.at[index, field]) and record.get(field) not in (
                        None,
                        "",
                        "unknown",
                    ):
                        result.at[index, field] = str(record[field])
    if annotations_path.is_file():
        annotations = (
            pd.read_parquet(annotations_path)
            if annotations_path.suffix.lower() == ".parquet"
            else pd.read_csv(annotations_path, low_memory=False)
        )
        available = [name for name in ("protein_id", "uniprot_family") if name in annotations]
        if len(available) == 2:
            result = result.merge(
                annotations[available], on="protein_id", how="left", validate="one_to_one"
            )
    if "uniprot_family" not in result:
        result["uniprot_family"] = pd.NA
    return result


def _categorical_effect(values: np.ndarray, groups: pd.Series) -> tuple[float, float, int, int]:
    valid = groups.notna() & groups.astype(str).ne("unknown")
    grouped = pd.DataFrame({"value": values, "group": groups.astype(str)}).loc[valid]
    counts = grouped.group.value_counts()
    grouped = grouped.loc[grouped.group.isin(counts[counts.ge(5)].index)]
    pieces = [part.value.to_numpy() for _, part in grouped.groupby("group")]
    if len(pieces) < 2:
        return float("nan"), float("nan"), int(len(grouped)), len(pieces)
    overall = float(grouped.value.mean())
    total = float(((grouped.value - overall) ** 2).sum())
    if total == 0:
        return 0.0, float("nan"), int(len(grouped)), len(pieces)
    between = sum(
        len(part) * float((part.value.mean() - overall) ** 2)
        for _, part in grouped.groupby("group")
    )
    try:
        _, p_value = kruskal(*pieces)
    except ValueError:
        p_value = 1.0
    return between / total, float(p_value), int(len(grouped)), len(pieces)


def confounder_table(
    candidates: list[int], aggregates: dict[str, np.ndarray], metadata: pd.DataFrame
) -> pd.DataFrame:
    """Report coverage and association of selected features with known confounders."""
    rows: list[dict[str, object]] = []
    continuous = ("sequence_length", "alphafold_mean_plddt")
    categorical = ("source_dataset", "uniprot_family", "family_id", "disorder_or_heterogeneity")
    for aggregate, values in aggregates.items():
        for feature in candidates:
            for name in continuous:
                valid = metadata[name].notna()
                feature_values = values[valid, feature]
                target_values = metadata.loc[valid, name]
                rho, p_value = (
                    spearmanr(feature_values, target_values)
                    if valid.sum() >= 3
                    and np.ptp(feature_values) > 0
                    and np.ptp(target_values.to_numpy(dtype=float)) > 0
                    else (np.nan, np.nan)
                )
                rows.append(
                    {
                        "feature_id": feature,
                        "aggregate": aggregate,
                        "confounder": name,
                        "kind": "continuous",
                        "effect": rho,
                        "p_value": p_value,
                        "rows": int(valid.sum()),
                        "categories": np.nan,
                    }
                )
            for name in categorical:
                effect, p_value, rows_used, categories = _categorical_effect(
                    values[:, feature], metadata[name]
                )
                rows.append(
                    {
                        "feature_id": feature,
                        "aggregate": aggregate,
                        "confounder": name,
                        "kind": "categorical_eta_squared_kruskal_p",
                        "effect": effect,
                        "p_value": p_value,
                        "rows": rows_used,
                        "categories": categories,
                    }
                )
    result = pd.DataFrame(rows)
    valid = result.p_value.notna()
    result["fdr"] = np.nan
    if valid.any():
        result.loc[valid, "fdr"] = bh_adjust(result.loc[valid, "p_value"].to_numpy(dtype=float))
    return result


def subtype_table(
    catalog: pd.DataFrame, aggregates: dict[str, np.ndarray], run_subtype: bool
) -> pd.DataFrame:
    """Keep subtype analysis implemented but inactive until source labels are approved."""
    if not run_subtype:
        return pd.DataFrame([{"status": "skipped", "reason": "--run-subtype was not supplied"}])
    subtypes = []
    for value in catalog.source_metadata_json:
        subtype = None
        try:
            records = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            records = []
        for record in records if isinstance(records, list) else [records]:
            if isinstance(record, dict) and record.get("positive_subtype"):
                subtype = str(record["positive_subtype"])
                break
        subtypes.append(subtype)
    values = pd.Series(subtypes)
    counts = values.value_counts()
    usable = catalog.dataset_label.eq(1) & values.isin(counts[counts.ge(5)].index)
    if usable.sum() == 0 or values[usable].nunique() < 2:
        return pd.DataFrame(
            [{"status": "skipped", "reason": "fewer than two usable subtype groups"}]
        )
    rows: list[dict[str, object]] = []
    for aggregate, matrix in aggregates.items():
        for feature in range(FEATURES):
            groups = [
                matrix[usable.to_numpy() & values.eq(name).to_numpy(), feature]
                for name in values[usable].unique()
            ]
            try:
                statistic, p_value = kruskal(*groups)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rows.append(
                {
                    "status": "complete",
                    "feature_id": feature,
                    "aggregate": aggregate,
                    "f_statistic": statistic,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = bh_adjust(result.p_value.to_numpy(dtype=float))
    return result


def structural_consistency(
    test: pd.DataFrame,
    candidates: list[int],
    model: torch.nn.Module,
    center: np.ndarray,
    device: torch.device,
    displacement_path: Path,
    prs_path: Path,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one motif hotspot per protein and candidate without structure downloads."""
    hotspot_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.seed)
    for completed, row in enumerate(test.itertuples(index=False), start=1):
        _, _, latents = _load_centered_latents(row, model, center, device)
        selected = latents[:, candidates].cpu().numpy()
        for index, feature in enumerate(candidates):
            position = int(np.argmax(selected[:, index]))
            random_position = int(rng.integers(0, len(row.sequence)))
            half = config.motif_half_window
            lower, upper = max(0, position - half), min(len(row.sequence), position + half + 1)
            hotspot_rows.append(
                {
                    "feature_id": feature,
                    "protein_id": row.protein_id,
                    "sequence_sha256": row.sequence_sha256,
                    "dataset_label": row.dataset_label,
                    "source_dataset": row.source_dataset,
                    "position_zero_based": position,
                    "canonical_residue_number": position + 1,
                    "relative_position": position / max(len(row.sequence) - 1, 1),
                    "activation": float(selected[position, index]),
                    "sequence_window": row.sequence[lower:upper],
                    "window_center_offset": position - lower,
                    "matched_random_position": random_position,
                    "matched_random_relative_position": random_position
                    / max(len(row.sequence) - 1, 1),
                }
            )
        if completed == 1 or completed % 25 == 0 or completed == len(test):
            status("collecting_structural_hotspots", completed=completed, total=len(test))
    hotspots = pd.DataFrame(hotspot_rows)
    motif_rows: list[dict[str, object]] = []
    for feature, values in hotspots.groupby("feature_id"):
        for offset in range(-config.motif_half_window, config.motif_half_window + 1):
            amino = []
            for row in values.itertuples(index=False):
                index = row.window_center_offset + offset
                amino.append(
                    row.sequence_window[index] if 0 <= index < len(row.sequence_window) else "-"
                )
            for residue in "-" + AMINO_ACIDS:
                motif_rows.append(
                    {
                        "feature_id": feature,
                        "offset": offset,
                        "amino_acid": residue,
                        "frequency": float(np.mean(np.asarray(amino) == residue)),
                        "proteins": len(values),
                    }
                )
    mapped = hotspots.loc[hotspots.dataset_label.eq(1)].copy()
    if displacement_path.is_file() and prs_path.is_file():
        key = ["protein_id", "sequence_sha256", "canonical_residue_number"]
        displacement = pd.read_csv(displacement_path, low_memory=False)[
            key + ["ca_displacement_after_global_kabsch_angstrom"]
        ]
        prs = pd.read_csv(prs_path, low_memory=False)[key + ["prs_max_overlap"]]
        mapped = mapped.merge(
            displacement.merge(prs, on=key, validate="one_to_one"), on=key, how="left"
        )
    return hotspots, pd.DataFrame(motif_rows), mapped


def structural_consistency_summary(hotspots: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    """Summarize sequence-position controls and available mapped transition scores."""
    summary = hotspots.groupby("feature_id", as_index=False).agg(
        proteins=("protein_id", "nunique"),
        mean_activation=("activation", "mean"),
        mean_hotspot_relative_position=("relative_position", "mean"),
        mean_random_relative_position=("matched_random_relative_position", "mean"),
    )
    summary["hotspot_minus_random_relative_position"] = (
        summary.mean_hotspot_relative_position - summary.mean_random_relative_position
    )
    score_columns = [
        name
        for name in ("ca_displacement_after_global_kabsch_angstrom", "prs_max_overlap")
        if name in mapped
    ]
    if score_columns:
        transition = mapped.groupby("feature_id", as_index=False).agg(
            mapped_hotspots=("protein_id", "count"),
            **{f"hotspot_mean_{name}": (name, "mean") for name in score_columns},
        )
        summary = summary.merge(transition, on="feature_id", how="left", validate="one_to_one")
    return summary


def write_html(
    output: Path,
    summary: dict[str, object],
    candidates: pd.DataFrame,
    test_enrichment: pd.DataFrame,
    ablations: pd.DataFrame,
    confounders: pd.DataFrame,
) -> None:
    candidate_ids = candidates.feature_id.tolist()
    enrichment = (
        test_enrichment.loc[test_enrichment.feature_id.isin(candidate_ids)]
        .sort_values("fdr")
        .head(25)
    )
    ablation = (
        ablations.loc[ablations.condition.eq("candidate")]
        .sort_values("auroc_drop_from_reconstruction", ascending=False)
        .head(30)
    )
    strongest = confounders.sort_values("fdr").head(30)

    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, float_format=lambda value: f"{value:.4g}")

    page = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>SAE router feature tests</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#15212b}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 28px}}th,td{{padding:6px;border-bottom:1px solid #d6e0e7;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf3f6}}.note{{padding:14px;background:#eef6ff;border-left:4px solid #286fb5}}</style></head><body>
<h1>Frozen SAE router-feature tests</h1><p class=\"note\">Validation selected router-enriched candidates. Test results are held out. The legacy ESMFold transition feature is included only when it exists in the selected SAE feature space. Structural consistency uses mapped residue scores and sequence motifs only; no new coordinate downloads were performed.</p>
<p>Test proteins: {summary["test_proteins"]:,}. SAE features: {FEATURES:,}. Shared random ablations: {summary["random_controls"]:,}.</p>
<h2>Selected candidates</h2>{table(candidates)}
<h2>Held-out label enrichment</h2>{table(enrichment)}
<h2>Classifier ablations</h2>{table(ablation)}
<h2>Strongest specificity/confounder associations</h2>{table(strongest)}
<h2>Artifacts</h2><ul><li><a href=\"test_feature_enrichment.csv\">All test enrichment tests</a></li><li><a href=\"ablation_metrics.csv\">Ablation metrics</a></li><li><a href=\"feature_specificity_confounders.csv\">Specificity/confounder table</a></li><li><a href=\"structural_consistency_summary.csv\">Structural-consistency summary</a></li><li><a href=\"structural_hotspots.parquet\">Hotspot positions</a></li><li><a href=\"motif_profiles.csv\">Motif profiles</a></li></ul></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def write_figures(output: Path, enrichment: pd.DataFrame, ablations: pd.DataFrame) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grouped = enrichment.groupby("feature_id", as_index=False).agg(
        effect=("rank_biserial_dynamic_vs_static", lambda x: x.iloc[np.argmax(np.abs(x))]),
        fdr=("fdr", "min"),
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(grouped.effect, -np.log10(grouped.fdr.clip(lower=1e-300)), s=5, alpha=0.45)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Validation/test rank-biserial effect",
        ylabel="-log10 FDR",
        title="SAE feature router-label enrichment",
    )
    fig.tight_layout()
    fig.savefig(figures / "label_enrichment_volcano.png", dpi=180)
    plt.close(fig)
    values = ablations.loc[ablations.condition.eq("candidate")]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for model, subset in values.groupby("model"):
        axis.scatter(
            subset.feature_id, subset.auroc_drop_from_reconstruction, label=model, alpha=0.8
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="SAE feature",
        ylabel="AUROC drop from full SAE reconstruction",
        title="Candidate feature ablations",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "candidate_ablation_auroc.png", dpi=180)
    plt.close(fig)


def run(
    catalog_path: Path,
    sae_root: Path,
    models_root: Path,
    annotations_path: Path,
    displacement_path: Path,
    prs_path: Path,
    output: Path,
    config: RunConfig,
) -> dict[str, object]:
    global DEFAULT_MODELS, FEATURES
    DEFAULT_MODELS = models_root
    device = resolve_device(config.device)
    catalog = pd.read_parquet(catalog_path)
    _require_catalog(catalog)
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "summary.json"
    if complete.is_file() and (output / "progress.json").is_file():
        progress = json.loads((output / "progress.json").read_text())
        if (
            progress.get("status") == "complete"
            and progress.get("config_hash") == config.config_hash
        ):
            status("already_complete", output=str(output))
            return json.loads(complete.read_text())
    model, center, manifest = load_frozen_sae(sae_root, device)
    FEATURES = int(model.config.latent_dim)
    if manifest.get("catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("frozen SAE catalog checksum does not match the frozen Seed-42 catalog")
    val = catalog.loc[catalog.split.eq("val")].reset_index(drop=True)
    test = catalog.loc[catalog.split.eq("test")].reset_index(drop=True)
    status("loaded_inputs", validation=len(val), test=len(test), device=str(device))
    atomic_json(
        output / "run_manifest.json",
        {
            "config": asdict(config),
            "config_hash": config.config_hash,
            "catalog": str(catalog_path),
            "catalog_sha256": sha256_file(catalog_path),
            "sae": str(sae_root),
            "sae_manifest_sha256": sha256_file(sae_root / "manifest.json"),
            "models": str(models_root),
            "status": "running",
        },
    )
    val_aggregates = aggregate_partition(val, "validation", model, center, device, output, config)
    test_aggregates = aggregate_partition(test, "test", model, center, device, output, config)
    validation_enrichment = enrichment_table(
        val_aggregates, val.dataset_label.to_numpy(), "validation"
    )
    test_enrichment = enrichment_table(test_aggregates, test.dataset_label.to_numpy(), "test")
    validation_enrichment.to_csv(output / "validation_feature_enrichment.csv", index=False)
    test_enrichment.to_csv(output / "test_feature_enrichment.csv", index=False)
    validation_enrichment.to_parquet(output / "validation_feature_enrichment.parquet", index=False)
    test_enrichment.to_parquet(output / "test_feature_enrichment.parquet", index=False)
    candidates = select_candidates(validation_enrichment, config.fdr_alpha)
    candidate_ids = candidates.feature_id.astype(int).tolist()
    controls = select_random_controls(
        val_aggregates["fraction_active"], candidate_ids, config.random_controls, config.seed
    )
    controls.to_csv(output / "shared_random_controls.csv", index=False)
    candidates.to_csv(output / "selected_candidates.csv", index=False)
    ablation_ids = [*candidate_ids, *controls.feature_id.astype(int).tolist()]
    metrics_path = output / "ablation_metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        status("reusing_ablation_metrics", rows=len(metrics))
    else:
        predictions = evaluate_ablations(test, ablation_ids, model, center, device, output, config)
        metrics = ablation_metrics(
            test.dataset_label.to_numpy(),
            predictions,
            ablation_ids,
            set(candidate_ids),
            set(controls.feature_id),
            config,
        )
        metrics.to_csv(metrics_path, index=False)
    metadata = _metadata_fields(test.reset_index(drop=True), annotations_path)
    confounders = confounder_table(candidate_ids, test_aggregates, metadata)
    confounders.to_csv(output / "feature_specificity_confounders.csv", index=False)
    subtype = subtype_table(test.reset_index(drop=True), test_aggregates, config.run_subtype)
    subtype.to_csv(output / "subtype_enrichment.csv", index=False)
    hotspots, motifs, mapped = structural_consistency(
        test, candidate_ids, model, center, device, displacement_path, prs_path, config
    )
    hotspots.to_parquet(output / "structural_hotspots.parquet", index=False)
    motifs.to_csv(output / "motif_profiles.csv", index=False)
    mapped.to_csv(output / "mapped_hotspot_transition_scores.csv", index=False)
    structural_consistency_summary(hotspots, mapped).to_csv(
        output / "structural_consistency_summary.csv", index=False
    )
    summary = {
        "config_hash": config.config_hash,
        "validation_proteins": len(val),
        "test_proteins": len(test),
        "test_dynamic_proteins": int(test.dataset_label.sum()),
        "test_static_proteins": int((test.dataset_label == 0).sum()),
        "features": FEATURES,
        "candidates": candidate_ids,
        "random_controls": len(controls),
        "exploratory_transition_feature": (
            EXPLORATORY_FEATURE if EXPLORATORY_FEATURE < FEATURES else None
        ),
        "subtype_status": str(subtype.iloc[0].get("status", "complete")),
    }
    write_figures(output, test_enrichment, metrics)
    write_html(output, summary, candidates, test_enrichment, metrics, confounders)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": "complete", "config_hash": config.config_hash})
    status("complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--displacement", type=Path, default=DEFAULT_DISPLACEMENT)
    parser.add_argument("--prs", type=Path, default=DEFAULT_PRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--run-subtype", action="store_true")
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    args = parser.parse_args()
    if min(args.feature_batch_size, args.random_controls, args.bootstrap_draws) < 1:
        parser.error("batch size, random controls, and bootstrap draws must be positive")
    config = RunConfig(
        device=args.device,
        feature_batch_size=args.feature_batch_size,
        random_controls=args.random_controls,
        bootstrap_draws=args.bootstrap_draws,
        run_subtype=args.run_subtype,
        representation_name=args.representation_name,
    )
    print(
        json.dumps(
            run(
                args.catalog,
                args.sae_root,
                args.models_root,
                args.annotations,
                args.displacement,
                args.prs,
                args.output,
                config,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
