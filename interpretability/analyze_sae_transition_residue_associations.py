"""Associate frozen SAE features with residue displacement and PRS on seed-42 test data."""

# ruff: noqa: E402 - direct execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
from scipy.stats import rankdata
from scipy.stats import t as student_t
from torch.torch_version import TorchVersion

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.freeze_seed42_sae import configured_latent_dim

from interpretability.contracts import load_residue_matrix
from interpretability.methods import SparseAutoencoder, SparseAutoencoderConfig

DATA_ROOT = Path("data/lifecycle/final/initial_8598_dataset")
DEFAULT_OUTPUT = Path("interpretability/results/homology35_sae_transition_residue_associations")
TARGETS = {
    "displacement": "ca_displacement_after_global_kabsch_angstrom",
    "prs": "prs_max_overlap",
}


@dataclass(frozen=True)
class AnalysisConfig:
    statistic_version: str = "within_protein_percentile_rank_v2"
    seed: int = 42
    n_permutations: int = 1_000
    permutation_batch_size: int = 16
    inference_batch_size: int = 512
    permutation_row_batch_size: int = 4_096
    checkpoint_every: int = 25
    minimum_active_residues_per_protein: int = 5
    fdr_alpha: float = 0.05
    device: str = "auto"
    allow_incomplete: bool = False

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def status(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def resolve_device(device: str) -> torch.device:
    """Use MPS when available, otherwise retain an explicit CPU fallback."""
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; use --device auto or --device cpu")
    return torch.device(device)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be one finite vector in [0, 1]")
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def correlation_p_value(correlation: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(correlation, dtype=np.float64)
    clipped = np.clip(values, -1 + 1e-15, 1 - 1e-15)
    statistic = np.abs(clipped) * np.sqrt((n - 2) / np.maximum(1 - clipped**2, 1e-15))
    result = 2 * student_t.sf(statistic, df=max(n - 2, 1))
    result[~np.isfinite(values)] = np.nan
    return result


def weighted_correlation_from_sums(
    sum_weight: float,
    sum_x: np.ndarray,
    sum_x2: np.ndarray,
    sum_y: float,
    sum_y2: float,
    sum_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = sum_xy - sum_x * sum_y / sum_weight
    variance_x = sum_x2 - sum_x**2 / sum_weight
    variance_y = sum_y2 - sum_y**2 / sum_weight
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    slope = np.divide(
        covariance,
        variance_x,
        out=np.full_like(covariance, np.nan, dtype=np.float64),
        where=variance_x > 0,
    )
    return correlation, slope


def load_frozen_sae(root: Path, device: torch.device) -> tuple[SparseAutoencoder, np.ndarray, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("kind") != "full_matrix_residue_sae"
        or manifest.get("fit_partition") != "seed_42_train"
    ):
        raise ValueError("SAE is not the frozen seed-42 train-fit residue model")
    for name, expected in manifest.get("artifacts", {}).items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen SAE artifact checksum mismatch: {name}")
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    config = payload["config"]
    input_dim = int(config["input_dim"])
    latent_dim = configured_latent_dim(config)
    architecture = manifest.get("architecture", {})
    if (
        int(architecture.get("input_dim", input_dim)) != input_dim
        or int(architecture.get("latent_dim", latent_dim)) != latent_dim
    ):
        raise ValueError("frozen SAE manifest architecture differs from checkpoint configuration")
    model = SparseAutoencoder(
        SparseAutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            l1_coefficient=0.0,
            top_k=int(config["top_k"]),
        )
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    center = np.load(root / str(payload["center_artifact"])).astype(np.float32)
    if center.shape != (input_dim,):
        raise ValueError("SAE input center has an invalid shape")
    return model, center, manifest


def prepare_residue_index(
    catalog_path: Path,
    displacement_path: Path,
    prs_path: Path,
    *,
    allow_incomplete: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    catalog = pd.read_parquet(catalog_path)
    eligible = catalog.loc[catalog.split.eq("test") & catalog.dataset_label.eq(1)].copy()
    if eligible.empty or eligible.protein_id.duplicated().any():
        raise ValueError("homology-grouped dynamic test partition is empty or duplicated")
    displacement = pd.read_csv(displacement_path, low_memory=False)
    prs = pd.read_csv(prs_path, low_memory=False)
    key = ["protein_id", "sequence_sha256", "canonical_residue_number"]
    all_residues = displacement[key + [TARGETS["displacement"]]].merge(
        prs[key + [TARGETS["prs"]]], on=key, validate="one_to_one"
    )
    if all_residues.duplicated(key).any():
        raise ValueError("transition residue tables contain duplicate canonical residue keys")
    eligible_ids = set(eligible.protein_id)
    scored_ids = set(all_residues.protein_id)
    missing_proteins = eligible_ids - scored_ids
    extra_proteins = scored_ids - eligible_ids
    if (extra_proteins or missing_proteins) and not allow_incomplete:
        raise ValueError(
            "transition scores must cover all homology-grouped dynamic test proteins; "
            f"missing={len(missing_proteins)} extra={len(extra_proteins)}"
        )
    residues = all_residues.loc[all_residues.protein_id.isin(eligible_ids)].copy()
    completed = residues.protein_id.nunique()
    if completed == 0:
        raise ValueError("transition score tables contain no eligible frozen test proteins")
    metadata = eligible[
        [
            "protein_id",
            "sequence_sha256",
            "sequence",
            "sequence_length",
            "embedding_path",
            "source_dataset",
        ]
    ]
    residues = residues.merge(
        metadata, on=["protein_id", "sequence_sha256"], validate="many_to_one"
    )
    residues = residues.sort_values(["protein_id", "canonical_residue_number"]).reset_index(
        drop=True
    )
    proteins = (
        residues[["protein_id"]]
        .drop_duplicates()
        .merge(metadata, on="protein_id", validate="one_to_one")
        .reset_index(drop=True)
    )
    counts = residues.groupby("protein_id", sort=False).size().reindex(proteins.protein_id)
    starts = np.concatenate(([0], np.cumsum(counts.to_numpy(dtype=np.int64))))
    proteins["residue_start"] = starts[:-1]
    proteins["residue_stop"] = starts[1:]
    summary = {
        "eligible_dynamic_test_proteins": len(eligible),
        "completed_structure_prs_proteins": completed,
        "unavailable_transition_score_proteins": len(missing_proteins),
        "transition_score_proteins_outside_catalog": len(extra_proteins),
        "transition_score_scope": (
            "all homology-grouped Seed-42 dynamic test proteins"
            if not missing_proteins
            else "QC-complete homology-grouped Seed-42 dynamic test proteins"
        ),
        "aligned_residues": len(residues),
    }
    return proteins, residues, summary


def _initialize_memmap(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap:
    if path.is_file():
        return np.lib.format.open_memmap(path, mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def encode_sparse_activations(
    proteins: pd.DataFrame,
    residues: pd.DataFrame,
    model: SparseAutoencoder,
    center: np.ndarray,
    device: torch.device,
    output: Path,
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    top_k = int(model.config.top_k or model.config.latent_dim)
    indices_path = work / "feature_indices.npy"
    values_path = work / "activation_values.npy"
    indices = _initialize_memmap(indices_path, (len(residues), top_k), "uint16")
    values = _initialize_memmap(values_path, (len(residues), top_k), "float32")
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if progress and progress.get("config_hash") != config.config_hash:
        raise ValueError("existing SAE association checkpoint has a different configuration")
    completed = int(progress.get("encoded_proteins", 0))
    for protein_index, protein in enumerate(proteins.itertuples(index=False)):
        if protein_index < completed:
            continue
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        matrix = load_residue_matrix(
            Path(protein.embedding_path),
            protein_id=protein.protein_id,
            sequence=protein.sequence,
            sequence_sha256=protein.sequence_sha256,
            sequence_length=int(protein.sequence_length),
            expected_width=len(center),
        )
        positions = residues.iloc[start:stop].canonical_residue_number.to_numpy(dtype=np.int64) - 1
        if positions.min() < 0 or positions.max() >= len(matrix):
            raise ValueError(f"canonical residue index outside embedding: {protein.protein_id}")
        selected = matrix[positions].astype(np.float32, copy=False) - center
        for batch_start in range(0, len(selected), config.inference_batch_size):
            batch_stop = min(batch_start + config.inference_batch_size, len(selected))
            batch = torch.from_numpy(selected[batch_start:batch_stop]).to(device)
            with torch.inference_mode():
                latents = model.encode(batch)
                top_values, top_indices = torch.topk(latents, k=top_k, dim=1, sorted=False)
            target = slice(start + batch_start, start + batch_stop)
            indices[target] = top_indices.to(torch.int32).cpu().numpy().astype(np.uint16)
            values[target] = top_values.cpu().numpy().astype(np.float32)
        indices.flush()
        values.flush()
        completed = protein_index + 1
        atomic_json(
            progress_path,
            {
                "status": "encoding",
                "config_hash": config.config_hash,
                "encoded_proteins": completed,
                "total_proteins": len(proteins),
                "completed_permutations": int(progress.get("completed_permutations", 0)),
            },
        )
        if completed == 1 or completed % 25 == 0 or completed == len(proteins):
            status("encoding", completed=completed, total=len(proteins), device=str(device))
    return np.asarray(indices), np.asarray(values)


def sparse_feature_ranks(
    indices: np.ndarray,
    values: np.ndarray,
    feature_count: int,
    proteins: pd.DataFrame,
) -> np.ndarray:
    """Return sparse within-protein percentile-rank deltas.

    Inactive residues share the local zero rank.  Active entries store only
    their difference from that baseline, keeping the representation sparse
    while removing between-protein rank structure from the primary statistic.
    """
    rank_deltas = np.zeros_like(values, dtype=np.float32)
    for protein in proteins.itertuples(index=False):
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        length = stop - start
        local_indices = indices[start:stop].ravel()
        local_values = values[start:stop].ravel()
        local_deltas = rank_deltas[start:stop].ravel()
        positive_locations = np.flatnonzero(local_values > 0)
        order = np.argsort(local_indices[positive_locations], kind="stable")
        sorted_locations = positive_locations[order]
        sorted_features = local_indices[sorted_locations]
        boundaries = np.searchsorted(sorted_features, np.arange(feature_count + 1))
        for feature in np.unique(sorted_features):
            locations = sorted_locations[boundaries[feature] : boundaries[feature + 1]]
            active_count = len(locations)
            zero_rank = (length - active_count + 1) / 2
            active_ranks = (
                length - active_count + rankdata(local_values[locations], method="average")
            )
            local_deltas[locations] = ((active_ranks - zero_rank) / length).astype(np.float32)
    return rank_deltas


def within_protein_target_ranks(proteins: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for protein in proteins.itertuples(index=False):
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        length = stop - start
        ranks = rankdata(values[start:stop], method="average") / length
        result[start:stop] = ranks - ranks.mean()
    return result


def feature_sufficient_statistics(
    indices: np.ndarray,
    values: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_feature = indices.ravel().astype(np.int64)
    flat_value = values.ravel().astype(np.float64)
    flat_weight = np.repeat(weights, values.shape[1])
    flat_y = np.repeat(y, values.shape[1])
    positive = flat_value > 0
    feature = flat_feature[positive]
    value = flat_value[positive]
    weight = flat_weight[positive]
    target = flat_y[positive]
    count = np.bincount(feature, minlength=feature_count)
    sum_x = np.bincount(feature, weights=weight * value, minlength=feature_count)
    sum_x2 = np.bincount(feature, weights=weight * value**2, minlength=feature_count)
    sum_xy = np.bincount(feature, weights=weight * value * target, minlength=feature_count)
    return count, sum_x, sum_x2, sum_xy


def rank_sufficient_statistics(
    indices: np.ndarray,
    rank_deltas: np.ndarray,
    y_ranks: np.ndarray,
    weights: np.ndarray,
    feature_count: int,
    proteins: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_feature = indices.ravel().astype(np.int64)
    flat_delta = rank_deltas.ravel().astype(np.float64)
    flat_weight = np.repeat(weights, rank_deltas.shape[1])
    flat_y = np.repeat(y_ranks, rank_deltas.shape[1])
    active = flat_delta != 0
    feature = flat_feature[active]
    delta = flat_delta[active]
    weight = flat_weight[active]
    target = flat_y[active]
    count = np.bincount(feature, minlength=feature_count)
    sum_x = np.zeros(feature_count, dtype=np.float64)
    sum_x2 = np.zeros(feature_count, dtype=np.float64)
    for protein in proteins.itertuples(index=False):
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        length = stop - start
        present = indices[start:stop][rank_deltas[start:stop] != 0].astype(np.int64)
        active_counts = np.bincount(present, minlength=feature_count)
        baseline = (length - active_counts + 1) / (2 * length)
        sum_x += baseline
        sum_x2 += baseline**2
    sum_x += np.bincount(feature, weights=weight * delta, minlength=feature_count)
    baseline_by_active = np.empty(len(feature), dtype=np.float64)
    cursor = 0
    for protein in proteins.itertuples(index=False):
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        length = stop - start
        mask = rank_deltas[start:stop].ravel() != 0
        local_features = indices[start:stop].ravel()[mask].astype(np.int64)
        counts = np.bincount(local_features, minlength=feature_count)
        baseline = (length - counts + 1) / (2 * length)
        baseline_by_active[cursor : cursor + len(local_features)] = baseline[local_features]
        cursor += len(local_features)
    sum_x2 += np.bincount(
        feature,
        weights=weight * (2 * baseline_by_active * delta + delta**2),
        minlength=feature_count,
    )
    sum_xy = np.bincount(feature, weights=weight * delta * target, minlength=feature_count)
    return count, sum_x, sum_x2, sum_xy


def per_protein_effects(
    proteins: pd.DataFrame,
    residues: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    feature_count: int,
    minimum_active: int,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for target_name, target_column in TARGETS.items():
        effects = np.full((len(proteins), feature_count), np.nan, dtype=np.float32)
        slopes = np.full_like(effects, np.nan)
        for protein_index, protein in enumerate(proteins.itertuples(index=False)):
            start, stop = int(protein.residue_start), int(protein.residue_stop)
            n = stop - start
            y = residues.iloc[start:stop][target_column].to_numpy(dtype=np.float64)
            flat_feature = indices[start:stop].ravel().astype(np.int64)
            flat_value = values[start:stop].ravel().astype(np.float64)
            flat_y = np.repeat(y, values.shape[1])
            positive = flat_value > 0
            feature = flat_feature[positive]
            value = flat_value[positive]
            target = flat_y[positive]
            active_count = np.bincount(feature, minlength=feature_count)
            sum_x = np.bincount(feature, weights=value, minlength=feature_count)
            sum_x2 = np.bincount(feature, weights=value**2, minlength=feature_count)
            sum_xy = np.bincount(feature, weights=value * target, minlength=feature_count)
            corr, slope = weighted_correlation_from_sums(
                float(n), sum_x, sum_x2, float(y.sum()), float(y @ y), sum_xy
            )
            valid = active_count >= minimum_active
            effects[protein_index, valid] = corr[valid]
            slopes[protein_index, valid] = slope[valid]
        result[target_name] = {"correlation": effects, "slope": slopes}
    return result


def observed_statistics(
    proteins: pd.DataFrame,
    residues: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    rank_deltas: np.ndarray,
    feature_count: int,
    minimum_active: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = len(residues)
    lengths = (proteins.residue_stop - proteins.residue_start).to_numpy(dtype=np.int64)
    protein_weights = np.concatenate([np.full(length, 1 / length) for length in lengths])
    uniform_weights = np.ones(n, dtype=np.float64)
    activation_count, activation_sum, _, _ = feature_sufficient_statistics(
        indices, values, np.zeros(n), uniform_weights, feature_count
    )
    rows = pd.DataFrame(
        {
            "feature_id": np.arange(feature_count),
            "active_residue_count": activation_count,
            "activation_frequency": activation_count / n,
            "mean_activation": activation_sum / n,
            "mean_active_activation": np.divide(
                activation_sum,
                activation_count,
                out=np.zeros(feature_count),
                where=activation_count > 0,
            ),
        }
    )
    target_ranks: dict[str, np.ndarray] = {}
    primary_statistics: dict[str, np.ndarray] = {}
    for target_name, target_column in TARGETS.items():
        y = residues[target_column].to_numpy(dtype=np.float64)
        y_ranks = within_protein_target_ranks(proteins, y)
        target_ranks[target_name] = y_ranks
        for label, weights in (("global", uniform_weights), ("balanced", protein_weights)):
            _, sum_x, sum_x2, sum_xy = feature_sufficient_statistics(
                indices, values, y, weights, feature_count
            )
            pearson, slope = weighted_correlation_from_sums(
                float(weights.sum()),
                sum_x,
                sum_x2,
                float(weights @ y),
                float(weights @ (y**2)),
                sum_xy,
            )
            rows[f"{target_name}_{label}_pearson"] = pearson
            rows[f"{target_name}_{label}_slope"] = slope
            if label == "global":
                rows[f"{target_name}_pearson_p"] = correlation_p_value(pearson, n)
            else:
                _, rank_sum_x, rank_sum_x2, rank_sum_xy = rank_sufficient_statistics(
                    indices, rank_deltas, y_ranks, weights, feature_count, proteins
                )
                spearman, _ = weighted_correlation_from_sums(
                    float(weights.sum()),
                    rank_sum_x,
                    rank_sum_x2,
                    float(weights @ y_ranks),
                    float(weights @ (y_ranks**2)),
                    rank_sum_xy,
                )
                rows[f"{target_name}_{label}_spearman"] = spearman
                primary_statistics[target_name] = spearman
    effects = per_protein_effects(
        proteins, residues, indices, values, feature_count, minimum_active
    )
    for target_name, target_effects in effects.items():
        correlations = target_effects["correlation"]
        slopes = target_effects["slope"]
        valid_count = np.sum(np.isfinite(correlations), axis=0)
        rows[f"{target_name}_n_proteins_evaluable"] = valid_count
        rows[f"{target_name}_per_protein_mean_pearson"] = np.divide(
            np.nansum(correlations, axis=0),
            valid_count,
            out=np.full(feature_count, np.nan),
            where=valid_count > 0,
        )
        rows[f"{target_name}_per_protein_median_pearson"] = np.asarray(
            [
                np.nanmedian(correlations[:, feature]) if valid_count[feature] else np.nan
                for feature in range(feature_count)
            ]
        )
        rows[f"{target_name}_per_protein_mean_slope"] = np.divide(
            np.nansum(slopes, axis=0),
            valid_count,
            out=np.full(feature_count, np.nan),
            where=valid_count > 0,
        )
        sign = np.sign(primary_statistics[target_name])
        matches = np.sum(np.sign(correlations) == sign[None, :], axis=0)
        rows[f"{target_name}_fraction_proteins_same_sign"] = np.divide(
            matches, valid_count, out=np.full(feature_count, np.nan), where=valid_count > 0
        )
    active_proteins = np.zeros(feature_count, dtype=np.int64)
    for protein in proteins.itertuples(index=False):
        present = np.unique(
            indices[int(protein.residue_start) : int(protein.residue_stop)][
                values[int(protein.residue_start) : int(protein.residue_stop)] > 0
            ]
        )
        active_proteins[present.astype(np.int64)] += 1
    rows["n_proteins_active"] = active_proteins
    return rows, target_ranks, primary_statistics


def permutation_batch_statistics(
    indices: np.ndarray,
    rank_deltas: np.ndarray,
    weights: np.ndarray,
    permuted_targets: np.ndarray,
    feature_count: int,
    device: torch.device,
    row_batch_size: int,
    proteins: pd.DataFrame,
) -> np.ndarray:
    batch_size = permuted_targets.shape[1]
    sum_weight = float(weights.sum())
    sum_x = np.zeros(feature_count, dtype=np.float64)
    sum_x2 = np.zeros(feature_count, dtype=np.float64)
    flat_active = rank_deltas.ravel() != 0
    flat_features = indices.ravel()[flat_active].astype(np.int64)
    flat_deltas = rank_deltas.ravel()[flat_active].astype(np.float64)
    flat_weights = np.repeat(weights, rank_deltas.shape[1])[flat_active]
    baseline_by_active = np.empty(len(flat_features), dtype=np.float64)
    cursor = 0
    for protein in proteins.itertuples(index=False):
        start, stop = int(protein.residue_start), int(protein.residue_stop)
        length = stop - start
        mask = rank_deltas[start:stop].ravel() != 0
        local_features = indices[start:stop].ravel()[mask].astype(np.int64)
        counts = np.bincount(local_features, minlength=feature_count)
        baseline = (length - counts + 1) / (2 * length)
        sum_x += baseline
        sum_x2 += baseline**2
        baseline_by_active[cursor : cursor + len(local_features)] = baseline[local_features]
        cursor += len(local_features)
    sum_x += np.bincount(flat_features, weights=flat_weights * flat_deltas, minlength=feature_count)
    sum_x2 += np.bincount(
        flat_features,
        weights=flat_weights * (2 * baseline_by_active * flat_deltas + flat_deltas**2),
        minlength=feature_count,
    )
    target_tensor = torch.from_numpy(permuted_targets.astype(np.float32)).to(device)
    output = torch.zeros((feature_count, batch_size), dtype=torch.float32, device=device)
    for start in range(0, len(indices), row_batch_size):
        stop = min(start + row_batch_size, len(indices))
        feature = torch.from_numpy(indices[start:stop].astype(np.int32)).to(device).long()
        delta = torch.from_numpy(rank_deltas[start:stop]).to(device)
        weight = torch.from_numpy(weights[start:stop].astype(np.float32)).to(device)
        target = target_tensor[start:stop]
        contribution = delta[:, :, None] * weight[:, None, None] * target[:, None, :]
        contribution = contribution.reshape(-1, batch_size)
        feature = feature.reshape(-1, 1).expand_as(contribution)
        output.scatter_add_(0, feature, contribution)
    sum_y = weights @ permuted_targets
    sum_y2 = weights @ (permuted_targets**2)
    sum_xy = output.cpu().numpy().astype(np.float64)
    covariance = sum_xy - sum_x[:, None] * sum_y[None, :] / sum_weight
    variance_x = sum_x2 - sum_x**2 / sum_weight
    variance_y = sum_y2 - sum_y**2 / sum_weight
    denominator = np.sqrt(np.maximum(variance_x[:, None] * variance_y[None, :], 0))
    return np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
    ).T


def run_permutations(
    proteins: pd.DataFrame,
    indices: np.ndarray,
    rank_deltas: np.ndarray,
    target_ranks: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    output: Path,
    config: AnalysisConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    work = output / ".work"
    feature_count = len(next(iter(observed.values())))
    lengths = (proteins.residue_stop - proteins.residue_start).to_numpy(dtype=np.int64)
    weights = np.concatenate([np.full(length, 1 / length) for length in lengths]).astype(np.float64)
    nulls = {
        name: _initialize_memmap(
            work / f"{name}_permutation_null.npy",
            (config.n_permutations, feature_count),
            "float32",
        )
        for name in TARGETS
    }
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text())
    completed = int(progress.get("completed_permutations", 0))
    rng = np.random.default_rng(config.seed)
    # Advance deterministically to the checkpoint boundary.
    for _ in range(completed):
        for _name in TARGETS:
            for length in lengths:
                rng.permutation(length)
    while completed < config.n_permutations:
        stop = min(completed + config.permutation_batch_size, config.n_permutations)
        batch_size = stop - completed
        permutations = {
            name: np.empty((len(indices), batch_size), dtype=np.float64) for name in TARGETS
        }
        for batch_index in range(batch_size):
            for name in TARGETS:
                for protein in proteins.itertuples(index=False):
                    start, protein_stop = int(protein.residue_start), int(protein.residue_stop)
                    order = rng.permutation(protein_stop - start)
                    permutations[name][start:protein_stop, batch_index] = target_ranks[name][
                        start:protein_stop
                    ][order]
        for name in TARGETS:
            nulls[name][completed:stop] = permutation_batch_statistics(
                indices,
                rank_deltas,
                weights,
                permutations[name],
                feature_count,
                device,
                config.permutation_row_batch_size,
                proteins,
            ).astype(np.float32)
            nulls[name].flush()
        completed = stop
        atomic_json(
            progress_path,
            {
                "status": "permuting" if completed < config.n_permutations else "analyzing",
                "config_hash": config.config_hash,
                "encoded_proteins": len(proteins),
                "total_proteins": len(proteins),
                "completed_permutations": completed,
                "total_permutations": config.n_permutations,
            },
        )
        status("permutations", completed=completed, total=config.n_permutations, device=str(device))
    return {name: np.asarray(values) for name, values in nulls.items()}


def finalize_feature_table(
    table: pd.DataFrame,
    observed: dict[str, np.ndarray],
    nulls: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = table.copy()
    for name in TARGETS:
        statistic = observed[name]
        null = nulls[name]
        valid = np.isfinite(statistic)
        empirical = np.ones(len(statistic), dtype=np.float64)
        empirical[valid] = (
            1 + np.sum(np.abs(null[:, valid]) >= np.abs(statistic[valid])[None, :], axis=0)
        ) / (len(null) + 1)
        result[f"{name}_perm_p"] = empirical
        result[f"{name}_fdr"] = bh_adjust(empirical)
    return result


def feature_vector(feature: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(indices), dtype=np.float32)
    rows, columns = np.where((indices == feature) & (values > 0))
    result[rows] = values[rows, columns]
    return result


def source_robustness(
    features: list[int],
    residues: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        activation = feature_vector(feature, indices, values)
        for source, source_rows in residues.groupby("source_dataset"):
            positions = source_rows.index.to_numpy(dtype=np.int64)
            if len(positions) < 20 or np.std(activation[positions]) == 0:
                continue
            for target_name, target_column in TARGETS.items():
                corr = np.corrcoef(
                    rankdata(activation[positions]), rankdata(source_rows[target_column].to_numpy())
                )[0, 1]
                rows.append(
                    {
                        "feature_id": feature,
                        "source_dataset": source,
                        "target": target_name,
                        "residues": len(positions),
                        "spearman": float(corr),
                    }
                )
    return pd.DataFrame(rows)


def save_figures(
    table: pd.DataFrame,
    residues: pd.DataFrame,
    indices: np.ndarray,
    values: np.ndarray,
    nulls: dict[str, np.ndarray],
    output: Path,
    seed: int,
) -> dict[str, list[int]]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[int]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, (name, _target_column) in zip(axes, TARGETS.items(), strict=True):
        ranked = table.sort_values(f"{name}_balanced_spearman", ascending=False).head(20)
        selected[name] = ranked.feature_id.head(5).astype(int).tolist()
        axis.barh(
            [f"F{value}" for value in ranked.feature_id[::-1]],
            ranked[f"{name}_balanced_spearman"][::-1],
            color="#276FBF" if name == "displacement" else "#2A9D8F",
        )
        axis.set_title(f"Positive {name} association")
        axis.set_xlabel("Protein-balanced Spearman correlation")
    fig.tight_layout()
    fig.savefig(figures / "top_feature_correlations.png", dpi=180)
    plt.close(fig)
    rng = np.random.default_rng(seed)
    for name, target_column in TARGETS.items():
        feature = selected[name][0]
        activation = feature_vector(feature, indices, values)
        target = residues[target_column].to_numpy(dtype=np.float64)
        sample = np.sort(rng.choice(len(target), size=min(20_000, len(target)), replace=False))
        bins = pd.qcut(target, q=20, duplicates="drop")
        trend = (
            pd.DataFrame({"target": target, "activation": activation, "bin": bins})
            .groupby("bin", observed=True)
            .agg(target=("target", "mean"), activation=("activation", "mean"))
        )
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.scatter(target[sample], activation[sample], s=5, alpha=0.08, color="#566573")
        axis.plot(trend.target, trend.activation, color="#D1495B", marker="o", linewidth=2)
        axis.set_title(f"Feature {feature} versus {name}")
        axis.set_xlabel(
            "C-alpha displacement (Angstrom)" if name == "displacement" else "PRS max overlap"
        )
        axis.set_ylabel("SAE activation")
        fig.tight_layout()
        fig.savefig(figures / f"feature_{feature}_{name}_trend.png", dpi=180)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.hist(nulls[name][:, feature], bins=40, color="#AAB7B8", edgecolor="white")
        observed = float(
            table.loc[table.feature_id.eq(feature), f"{name}_balanced_spearman"].iloc[0]
        )
        axis.axvline(observed, color="#C0392B", linewidth=2, label=f"Observed = {observed:.3f}")
        axis.set_title(f"Feature {feature}: within-protein permutation null")
        axis.set_xlabel("Protein-balanced Spearman correlation")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / f"feature_{feature}_{name}_permutation_null.png", dpi=180)
        plt.close(fig)
    return selected


def _format_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}g}" if np.isfinite(number) else "-"


def _feature_html(rows: pd.DataFrame, target: str) -> str:
    body = []
    for row in rows.head(10).itertuples(index=False):
        body.append(
            "<tr>"
            f"<td>F{int(row.feature_id)}</td>"
            f"<td>{_format_number(getattr(row, f'{target}_balanced_spearman'))}</td>"
            f"<td>{_format_number(getattr(row, f'{target}_perm_p'))}</td>"
            f"<td>{_format_number(getattr(row, f'{target}_fdr'))}</td>"
            f"<td>{int(row.n_proteins_active):,}</td>"
            f"<td>{_format_number(getattr(row, f'{target}_fraction_proteins_same_sign'))}</td>"
            "</tr>"
        )
    return "".join(body)


def write_html(
    table: pd.DataFrame,
    selected: dict[str, list[int]],
    cohort: dict[str, object],
    source_table: pd.DataFrame,
    output: Path,
    config: AnalysisConfig,
) -> None:
    displacement = table.sort_values("displacement_balanced_spearman", ascending=False)
    prs = table.sort_values("prs_balanced_spearman", ascending=False)
    both = table.loc[
        (table.displacement_fdr <= config.fdr_alpha) & (table.prs_fdr <= config.fdr_alpha)
    ]
    source_consistency = {}
    for target in TARGETS:
        subset = (
            source_table.loc[source_table.target.eq(target)]
            if not source_table.empty
            else source_table
        )
        source_consistency[target] = (
            float((subset.spearman > 0).mean()) if not subset.empty else float("nan")
        )
    displacement_feature = selected["displacement"][0]
    prs_feature = selected["prs"][0]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAE residue transition associations</title>
<style>
body{{font:16px/1.55 system-ui,-apple-system,sans-serif;max-width:1120px;margin:42px auto;padding:0 22px;color:#17212b;background:#fbfcfd}}
h1{{font-size:34px;line-height:1.15}}h2{{margin-top:38px}}.lede{{font-size:19px;color:#43515d}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}}.card{{background:white;border:1px solid #dce3e8;border-radius:10px;padding:16px}}.value{{font-size:27px;font-weight:700}}
table{{border-collapse:collapse;width:100%;background:white;margin:14px 0 28px}}th,td{{padding:9px 11px;border-bottom:1px solid #dce3e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f6}}
img{{max-width:100%;height:auto;background:white;border:1px solid #dce3e8;border-radius:8px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.note{{background:#edf6ff;border-left:4px solid #276fbf;padding:14px 17px}}code{{font-family:ui-monospace,monospace}}a{{color:#175c9e}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>SAE features associated with multistate residues</h1>
<p class="lede">Frozen Seed-42 TopK SAE analysis across every structurally aligned residue with valid displacement and PRS evidence.</p>
<div class="cards"><div class="card"><div class="value">{int(cohort["completed_structure_prs_proteins"]):,}</div>proteins analyzed</div><div class="card"><div class="value">{int(cohort["aligned_residues"]):,}</div>aligned residues</div><div class="card"><div class="value">4,096</div>SAE features</div><div class="card"><div class="value">{config.n_permutations:,}</div>within-protein permutations</div></div>
<div class="note">The dynamic Seed-42 test cohort contains {int(cohort["eligible_dynamic_test_proteins"])} proteins. This run includes every protein with QC-complete displacement and PRS scores; {int(cohort["unavailable_transition_score_proteins"])} proteins are unavailable and no values were imputed. “Displacement” is per-residue Cα movement after global alignment, not protein-level RMSD.</div>
<h2>Main result</h2><p>{int((table.displacement_fdr <= config.fdr_alpha).sum()):,} features pass displacement FDR ≤ {config.fdr_alpha:.2f}; {int((table.prs_fdr <= config.fdr_alpha).sum()):,} pass PRS FDR; {len(both):,} pass both. Primary effects use within-protein percentile ranks and equal total weight per protein, so between-protein rank structure and long proteins do not dominate.</p>
<img src="figures/top_feature_correlations.png" alt="Top SAE feature correlations">
<h2>Leading displacement features</h2><table><thead><tr><th>Feature</th><th>Balanced ρ</th><th>Permutation p</th><th>FDR</th><th>Active proteins</th><th>Same-sign proteins</th></tr></thead><tbody>{_feature_html(displacement, "displacement")}</tbody></table>
<h2>Leading PRS features</h2><table><thead><tr><th>Feature</th><th>Balanced ρ</th><th>Permutation p</th><th>FDR</th><th>Active proteins</th><th>Same-sign proteins</th></tr></thead><tbody>{_feature_html(prs, "prs")}</tbody></table>
<div class="grid"><div><img src="figures/feature_{displacement_feature}_displacement_trend.png" alt="Displacement trend"></div><div><img src="figures/feature_{prs_feature}_prs_trend.png" alt="PRS trend"></div><div><img src="figures/feature_{displacement_feature}_displacement_permutation_null.png" alt="Displacement permutation null"></div><div><img src="figures/feature_{prs_feature}_prs_permutation_null.png" alt="PRS permutation null"></div></div>
<h2>Robustness and interpretation</h2><p>For the selected candidate set, {source_consistency["displacement"]:.1%} of source-specific displacement effects and {source_consistency["prs"]:.1%} of source-specific PRS effects are positive. Per-protein sign consistency and source-stratified effects are included in the downloadable tables.</p>
<p>Association does not establish that a feature causes a conformational transition. Candidate features should next be localized to structures and tested by controlled feature intervention.</p>
<h2>Artifacts</h2><ul><li><a href="sae_feature_associations.csv">Feature association table</a></li><li><a href="ranked_displacement_features.csv">Ranked displacement features</a></li><li><a href="ranked_prs_features.csv">Ranked PRS features</a></li><li><a href="significant_both_features.csv">Features significant for both targets</a></li><li><a href="source_robustness.csv">Source robustness</a></li><li><a href="sparse_activations.npz">Sparse SAE activations</a></li><li><a href="residue_index.parquet">Residue index</a></li></ul>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def consolidate_sparse_artifact(
    indices: np.ndarray, values: np.ndarray, residues: pd.DataFrame, output: Path
) -> None:
    np.savez_compressed(
        output / "sparse_activations.npz",
        feature_indices=indices.astype(np.uint16),
        activation_values=values.astype(np.float32),
        schema_version=np.asarray("topk_sae_sparse_v1"),
    )
    residue_columns = [
        "protein_id",
        "sequence_sha256",
        "canonical_residue_number",
        "source_dataset",
        *TARGETS.values(),
    ]
    residues[residue_columns].to_parquet(output / "residue_index.parquet", index=False)


def run(
    catalog_path: Path,
    displacement_path: Path,
    prs_path: Path,
    sae_root: Path,
    output: Path,
    config: AnalysisConfig,
) -> dict[str, object]:
    device = resolve_device(config.device)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    summary_path = output / "summary.json"
    if progress_path.is_file() and summary_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("status") == "complete":
            if progress.get("config_hash") != config.config_hash:
                raise ValueError("completed output uses a different configuration")
            required = {
                "index.html",
                "sae_feature_associations.csv",
                "sparse_activations.npz",
                "residue_index.parquet",
            }
            missing = sorted(name for name in required if not (output / name).is_file())
            if missing:
                raise FileNotFoundError(f"completed output is missing artifacts: {missing}")
            summary = json.loads(summary_path.read_text())
            status("already_complete", output=str(output))
            return summary
    proteins, residues, cohort = prepare_residue_index(
        catalog_path,
        displacement_path,
        prs_path,
        allow_incomplete=config.allow_incomplete,
    )
    model, center, manifest = load_frozen_sae(sae_root, device)
    if manifest.get("catalog_sha256") != sha256_file(catalog_path):
        raise ValueError("frozen SAE and Seed-42 catalog checksums differ")
    feature_count = int(model.config.latent_dim)
    indices, values = encode_sparse_activations(
        proteins, residues, model, center, device, output, config
    )
    if indices.shape != (len(residues), int(model.config.top_k or feature_count)):
        raise ValueError("sparse SAE activation shape is invalid")
    status("ranking_sparse_activations", residues=len(residues), features=feature_count)
    rank_deltas = sparse_feature_ranks(indices, values, feature_count, proteins)
    status("ranked_sparse_activations", residues=len(residues), features=feature_count)
    status("computing_observed_statistics", residues=len(residues), features=feature_count)
    table, target_ranks, observed = observed_statistics(
        proteins,
        residues,
        indices,
        values,
        rank_deltas,
        feature_count,
        config.minimum_active_residues_per_protein,
    )
    status("computed_observed_statistics", features=feature_count)
    nulls = run_permutations(
        proteins, indices, rank_deltas, target_ranks, observed, output, config, device
    )
    table = finalize_feature_table(table, observed, nulls)
    table.to_csv(output / "sae_feature_associations.csv", index=False)
    table.to_parquet(output / "sae_feature_associations.parquet", index=False)
    displacement_ranked = table.sort_values("displacement_balanced_spearman", ascending=False)
    prs_ranked = table.sort_values("prs_balanced_spearman", ascending=False)
    displacement_ranked.to_csv(output / "ranked_displacement_features.csv", index=False)
    prs_ranked.to_csv(output / "ranked_prs_features.csv", index=False)
    both = table.loc[
        (table.displacement_fdr <= config.fdr_alpha) & (table.prs_fdr <= config.fdr_alpha)
    ].sort_values(["displacement_fdr", "prs_fdr"])
    both.to_csv(output / "significant_both_features.csv", index=False)
    candidate_features = list(
        dict.fromkeys(
            displacement_ranked.feature_id.head(20).astype(int).tolist()
            + prs_ranked.feature_id.head(20).astype(int).tolist()
        )
    )
    source_table = source_robustness(candidate_features, residues, indices, values)
    source_table.to_csv(output / "source_robustness.csv", index=False)
    selected = save_figures(table, residues, indices, values, nulls, output, config.seed)
    consolidate_sparse_artifact(indices, values, residues, output)
    summary = {
        **cohort,
        "device": str(device),
        "configuration": asdict(config),
        "config_hash": config.config_hash,
        "sae_manifest_sha256": sha256_file(sae_root / "manifest.json"),
        "sae_name": manifest["name"],
        "features": feature_count,
        "displacement_fdr_significant": int((table.displacement_fdr <= config.fdr_alpha).sum()),
        "prs_fdr_significant": int((table.prs_fdr <= config.fdr_alpha).sum()),
        "both_fdr_significant": int(len(both)),
        "top_displacement_feature": int(selected["displacement"][0]),
        "top_prs_feature": int(selected["prs"][0]),
    }
    atomic_json(output / "summary.json", summary)
    write_html(table, selected, cohort, source_table, output, config)
    atomic_json(
        output / "progress.json",
        {
            "status": "complete",
            "config_hash": config.config_hash,
            "encoded_proteins": len(proteins),
            "total_proteins": len(proteins),
            "completed_permutations": config.n_permutations,
            "total_permutations": config.n_permutations,
        },
    )
    shutil.rmtree(output / ".work", ignore_errors=True)
    status("complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--displacement",
        type=Path,
        default=DATA_ROOT / "analysis/homology35_dynamic_transition_residue_ca_displacements.csv",
    )
    parser.add_argument(
        "--prs",
        type=Path,
        default=DATA_ROOT / "analysis/homology35_dynamic_transition_prs_scores.csv",
    )
    parser.add_argument(
        "--sae-root",
        type=Path,
        default=Path("ml/results/homology35_frozen_saes/esmfold_matrix_topk64_seed42"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze only proteins with paired QC-complete transition scores.",
    )
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--permutation-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.permutations < 1 or args.permutation_batch_size < 1:
        parser.error("permutation counts and batch size must be positive")
    config = AnalysisConfig(
        device=args.device,
        n_permutations=args.permutations,
        permutation_batch_size=args.permutation_batch_size,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        json.dumps(
            run(
                args.catalog,
                args.displacement,
                args.prs,
                args.sae_root,
                args.output,
                config,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
