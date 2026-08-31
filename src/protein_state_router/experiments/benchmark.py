"""Small, leakage-safe benchmark runners for the finalized router dataset."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.models.probes import FeatureMLP, SequenceCNN
from protein_state_router.pooling.pooling import pool_single
from protein_state_router.representations.bundle_io import load_embedding_bundle
from protein_state_router.training.trainer import resolve_device, train_feature_mlp

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_NAMES = ("log1p_sequence_length", *(f"fraction_{aa}" for aa in AMINO_ACIDS), "entropy")
# This is intentionally small: the CNN is a sequence baseline, not the structure model.
# A 256-residue N-terminal window keeps the CPU overnight suite practical.
CNN_MAX_SEQUENCE_LENGTH = 256
CNN_FEATURE_NAMES = tuple((*AMINO_ACIDS, "X"))
FORBIDDEN_COLUMNS = frozenset(
    {
        "dataset_label",
        "single_structure_insufficient",
        "source_dataset",
        "label_confidence_tier",
        "negative_evidence_tier",
        "evidence_type",
        "source_reference",
        "source_id",
        "provenance_json",
        "structure_paths_json",
        "structure_ids_json",
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Small, explicit search space for one reproducible benchmark run."""

    family: Literal[
        "linear",
        "tree",
        "random_forest",
        "svm",
        "knn",
        "naive_bayes",
        "mlp",
        "cnn",
        "embedding_cnn",
    ]
    random_seed: int = 42
    primary_metric: str = "auroc"
    device: str = "cpu"
    search: Literal["fast", "standard", "expanded", "exhaustive"] = "standard"
    save_model: bool = True
    cpu_threads: int = 2

    def __post_init__(self) -> None:
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")


def sequence_feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return fixed sequence-only features without inspecting catalog provenance."""
    if "sequence" not in frame or "sequence_length" not in frame:
        raise ValueError("frame must include sequence and sequence_length")
    rows: list[list[float]] = []
    for sequence, length in zip(frame["sequence"], frame["sequence_length"], strict=True):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("every sequence must be a non-empty string")
        normalized = sequence.upper()
        if int(length) != len(normalized):
            raise ValueError("sequence_length must match the canonical sequence")
        denominator = float(len(normalized))
        fractions = np.asarray([normalized.count(aa) / denominator for aa in AMINO_ACIDS])
        observed = fractions[fractions > 0]
        entropy = float(-(observed * np.log(observed)).sum())
        rows.append([float(np.log1p(length)), *fractions.tolist(), entropy])
    return np.asarray(rows, dtype=np.float32), FEATURE_NAMES


def sequence_cnn_tensor(
    frame: pd.DataFrame, max_length: int = CNN_MAX_SEQUENCE_LENGTH
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return N-terminal-capped one-hot sequences for the compact CNN.

    A 256-residue cap keeps the CPU experiment bounded; canonical sequence
    length remains available to the tabular baselines. Unknown residues use the
    final ``X`` channel and padding remains zero.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if "sequence" not in frame or "sequence_length" not in frame:
        raise ValueError("frame must include sequence and sequence_length")
    channels = {residue: index for index, residue in enumerate(CNN_FEATURE_NAMES)}
    tensor = np.zeros((len(frame), len(channels), max_length), dtype=np.float32)
    for row_index, (sequence, length) in enumerate(
        zip(frame["sequence"], frame["sequence_length"], strict=True)
    ):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("every sequence must be a non-empty string")
        normalized = sequence.upper()
        if int(length) != len(normalized):
            raise ValueError("sequence_length must match the canonical sequence")
        for residue_index, residue in enumerate(normalized[:max_length]):
            tensor[row_index, channels.get(residue, channels["X"]), residue_index] = 1.0
    return tensor, CNN_FEATURE_NAMES


def embedding_cnn_tensor(features: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Reshape pooled embeddings for an exploratory one-channel CNN probe."""
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("embedding features must be a finite 2-D matrix")
    return features[:, None, :].astype(np.float32), tuple(
        f"embedding_{index}" for index in range(features.shape[1])
    )


def run_benchmark(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: BenchmarkConfig,
    *,
    features: np.ndarray | None = None,
    feature_names: tuple[str, ...] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    dataset_reference: str | None = None,
) -> dict[str, float]:
    """Tune one family on validation data, then write one held-out test report."""
    frame = pd.read_parquet(dataset_path)
    frame = frame.assign(split=frame["split"].replace({"validation": "val"}))
    _validate_dataset(frame)
    if features is None:
        features, feature_names = (
            sequence_cnn_tensor(frame) if config.family == "cnn" else sequence_feature_matrix(frame)
        )
    if feature_names is None or features.ndim < 2 or features.shape[0] != len(frame):
        raise ValueError("feature matrix must align with the dataset and its feature names")
    if not np.isfinite(features).all():
        raise ValueError("feature matrix must contain only finite values")
    labels = frame["dataset_label"].to_numpy(dtype=np.int64)
    train = frame["split"].eq("train").to_numpy()
    validation = frame["split"].eq("val").to_numpy()
    test = frame["split"].eq("test").to_numpy()
    if config.device == "cpu":
        torch.set_num_threads(config.cpu_threads)
    with threadpool_limits(limits=config.cpu_threads):
        candidate_name, candidate, validation_trials = _select_candidate(
            config, features, labels, train, validation, progress_callback
        )
        fitted = _fit_candidate(
            candidate, features[train | validation], labels[train | validation], config
        )
        probabilities = _predict(fitted, features[test])
    metrics = classification_metrics(labels[test], probabilities)
    prediction_columns = ["protein_id", "dataset_label", "split"]
    if "negative_evidence_tier" in frame:
        prediction_columns.append("negative_evidence_tier")
    predictions = frame.loc[test, prediction_columns].copy()
    predictions["probability"] = probabilities
    tier = predictions.get("negative_evidence_tier", pd.Series("", index=predictions.index)).fillna(
        ""
    )
    clean_negative = tier.isin(("B", "clean"))
    contextual_negative = tier.isin(("C", "contextual"))
    metrics["clean_negative_sample_count"] = float(clean_negative.sum())
    clean_mask = ~contextual_negative.to_numpy()
    if np.unique(labels[test][clean_mask]).size == 2:
        metrics.update(
            {
                f"clean_sensitivity_{key}": value
                for key, value in classification_metrics(
                    labels[test][clean_mask], probabilities[clean_mask]
                ).items()
            }
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "test_predictions.parquet", index=False)
    if isinstance(fitted, (_MlpCandidate, _CnnCandidate)) and fitted.network is not None:
        fitted.network.to("cpu")
    if isinstance(fitted, (_MlpCandidate, _CnnCandidate)):
        # The suite's live callback is process-local and must not enter joblib.
        fitted.progress_callback = None
    if config.save_model:
        joblib.dump(fitted, destination / "model.joblib")
    if isinstance(fitted, (_MlpCandidate, _CnnCandidate)) and fitted.history:
        pd.DataFrame(fitted.history).to_csv(destination / "learning_curves.csv", index=False)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (destination / "validation_selection.json").write_text(
        json.dumps(
            {
                "primary_metric": config.primary_metric,
                "selected_candidate": candidate_name,
                "trials": validation_trials,
            },
            indent=2,
            sort_keys=True,
        )
    )
    dataset = Path(dataset_path)
    digest = hashlib.sha256()
    with dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "candidate": candidate_name,
                "features": list(feature_names),
                "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
                "dataset_path": dataset_reference or str(dataset_path),
                "dataset_sha256": digest.hexdigest(),
                "split_counts": frame["split"].value_counts().sort_index().to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return metrics


def run_grouped_cross_validation(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: BenchmarkConfig,
    *,
    n_splits: int = 5,
    features: np.ndarray | None = None,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Report grouped out-of-fold metrics for every fixed candidate.

    This is an exploratory comparison, not a model-selection estimate: it does
    not choose a winner from the same held-out folds used for reporting. Use
    validation for selection and the fixed test split for final performance.
    """
    frame = pd.read_parquet(dataset_path)
    frame = frame.assign(split=frame["split"].replace({"validation": "val"}))
    _validate_dataset(frame)
    if features is None:
        features, feature_names = (
            sequence_cnn_tensor(frame) if config.family == "cnn" else sequence_feature_matrix(frame)
        )
    if feature_names is None or features.ndim < 2 or features.shape[0] != len(frame):
        raise ValueError("feature matrix must align with the dataset and its feature names")
    labels = frame["dataset_label"].to_numpy(dtype=np.int64)
    if "homology_group_id" not in frame:
        raise ValueError("grouped cross-validation requires homology_group_id")
    groups = frame["homology_group_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_seed
    )
    if config.device == "cpu":
        torch.set_num_threads(config.cpu_threads)
    candidate_scores: dict[str, list[float]] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=config.cpu_threads):
        for name, _ in _candidates(config):
            predictions = np.full(len(frame), np.nan, dtype=float)
            scores: list[float] = []
            for train_index, test_index in splitter.split(features, labels, groups):
                candidate = dict(_candidates(config))[name]
                fitted = _fit_candidate(
                    candidate, features[train_index], labels[train_index], config
                )
                probabilities = _predict(fitted, features[test_index])
                predictions[test_index] = probabilities
                scores.append(
                    float(
                        classification_metrics(labels[test_index], probabilities)[
                            config.primary_metric
                        ]
                    )
                )
                del fitted
                gc.collect()
            candidate_scores[name] = scores
            candidate_predictions[name] = predictions
    reports = {
        name: {
            **classification_metrics(labels, values),
            "cv_mean_primary_metric": float(np.mean(candidate_scores[name])),
            "cv_std_primary_metric": float(np.std(candidate_scores[name])),
        }
        for name, values in candidate_predictions.items()
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_predictions = frame[["protein_id", "dataset_label"]].copy()
    output_predictions["group"] = groups
    for name, values in sorted(candidate_predictions.items()):
        output_predictions[f"probability_{name}"] = values
    output_predictions.to_parquet(destination / "cv_predictions.parquet", index=False)
    (destination / "cv_metrics.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
    (destination / "cv_candidates.json").write_text(
        json.dumps(
            {
                "primary_metric": config.primary_metric,
                "n_splits": n_splits,
                "note": "Exploratory fixed-candidate comparison; no candidate is selected here.",
                "trials": [
                    {"candidate": name, "fold_metrics": scores, "mean": float(np.mean(scores))}
                    for name, scores in sorted(candidate_scores.items())
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"candidate_count": len(reports), "candidates": reports}


def single_embedding_feature_matrix(
    catalog: pd.DataFrame, bundle_manifest: pd.DataFrame, base_directory: str | Path
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load and mean/std/max-pool verified single bundles in catalog row order."""
    if {"protein_id", "sequence"} - set(catalog):
        raise ValueError("catalog requires protein_id and sequence")
    if {"protein_id", "bundle_path"} - set(bundle_manifest):
        raise ValueError("bundle manifest requires protein_id and bundle_path")
    if bundle_manifest["protein_id"].duplicated().any():
        raise ValueError("bundle manifest must contain exactly one row per protein_id")
    paths = bundle_manifest.set_index("protein_id")["bundle_path"]
    rows: list[np.ndarray] = []
    root = Path(base_directory)
    for row in catalog.itertuples(index=False):
        if row.protein_id not in paths:
            raise ValueError(f"missing single embedding bundle for {row.protein_id}")
        path = Path(paths[row.protein_id])
        embedding = load_embedding_bundle(
            path if path.is_absolute() else root / path, modalities=("single",)
        )
        if embedding.protein_id != row.protein_id or embedding.sequence != row.sequence:
            raise ValueError(f"bundle identity does not match catalog: {row.protein_id}")
        assert embedding.single is not None
        rows.append(
            pool_single(
                embedding.single.values.unsqueeze(0), embedding.single.residue_mask.unsqueeze(0)
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
    matrix = np.stack(rows).astype(np.float32)
    names = tuple(
        f"esmfold_single_{stat}_{index}"
        for stat in ("mean", "std", "max")
        for index in range(matrix.shape[1] // 3)
    )
    return matrix, names


def _validate_dataset(frame: pd.DataFrame) -> None:
    required = {"protein_id", "sequence", "sequence_length", "dataset_label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if set(frame["split"]) != {"train", "val", "test"}:
        raise ValueError("dataset must use fixed train, val, and test splits")
    if not set(frame["dataset_label"]).issubset({0, 1}):
        raise ValueError("dataset_label must be binary")
    for split in ("train", "val", "test"):
        if frame.loc[frame["split"].eq(split), "dataset_label"].nunique() != 2:
            raise ValueError(f"{split} must contain both classes")
    if (
        "homology_group_id" in frame
        and (frame.groupby("homology_group_id")["split"].nunique() > 1).any()
    ):
        raise ValueError("homology_group_id crosses split boundaries")


def _select_candidate(
    config: BenchmarkConfig,
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[str, object, list[dict[str, float | str]]]:
    candidates = _candidates(config, progress_callback)
    if config.family == "linear" and config.search == "standard" and len(candidates) == 8:
        # Candidate fits are independent, but this runner must honor its CPU
        # budget.  Each worker gets one numerical-library thread, and there
        # can never be more workers than ``cpu_threads``.
        def fit_score(item: tuple[str, object]) -> tuple[str, object, float]:
            name, candidate = item
            fitted = _fit_candidate(candidate, features[train], labels[train], config)
            score = float(
                classification_metrics(
                    labels[validation], _predict(fitted, features[validation])
                )[config.primary_metric]
            )
            return name, fitted, score

        completed: list[tuple[str, object, float]] = []
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=config.cpu_threads) as executor:
                futures = {}
                for item in candidates:
                    if progress_callback is not None:
                        progress_callback({"event": "candidate_started", "candidate": item[0]})
                    futures[executor.submit(fit_score, item)] = item[0]
                for future in as_completed(futures):
                    result = future.result()
                    completed.append(result)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "candidate_completed",
                                "candidate": result[0],
                                "validation_metric": result[2],
                            }
                        )
        trials = [
            {"candidate": name, "validation_metric": score}
            for name, _, score in sorted(completed, key=lambda item: item[0])
        ]
        selected_name, selected_fitted, _ = max(completed, key=lambda item: (item[2], item[0]))
        for _name, fitted, _ in completed:
            if fitted is not selected_fitted:
                del fitted
        gc.collect()
        return selected_name, selected_fitted, trials

    trials: list[dict[str, float | str]] = []
    selected: tuple[float, str, object] | None = None
    for name, candidate in candidates:
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "candidate_started",
                    "candidate": name,
                    "completed_candidates": len(trials),
                    "total_candidates": len(candidates),
                }
            )
        if isinstance(candidate, _MlpCandidate):
            fitted = candidate.fit_with_validation(
                features[train], labels[train], features[validation], labels[validation]
            )
        else:
            fitted = _fit_candidate(candidate, features[train], labels[train], config)
        metric = classification_metrics(labels[validation], _predict(fitted, features[validation]))[
            config.primary_metric
        ]
        score = float(metric)
        trials.append({"candidate": name, "validation_metric": score})
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "candidate_completed",
                    "candidate": name,
                    "validation_metric": score,
                    "completed_candidates": len(trials),
                    "total_candidates": len(candidates),
                }
            )
        if selected is None or (score, name) > (selected[0], selected[1]):
            selected = (score, name, fitted)
        if selected[2] is not fitted:
            del fitted
        gc.collect()
    if selected is None:
        raise RuntimeError("candidate search produced no models")
    return selected[1], selected[2], sorted(trials, key=lambda item: str(item["candidate"]))


def _candidates(
    config: BenchmarkConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> list[tuple[str, Any]]:
    # Follow-up confounder probes use a fixed, predeclared representative
    # model.  This avoids repeating the full sweep for every residual view
    # while leaving the standard/expanded benchmark searches unchanged.
    if config.search == "fast":
        if config.family == "linear":
            return [("logistic_l2_C1.0", _logistic("l2", 1.0, config.random_seed))]
        if config.family == "tree":
            return [
                (
                    "extra_trees_trees50_leaf2",
                    ExtraTreesClassifier(
                        n_estimators=50,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=config.cpu_threads,
                        random_state=config.random_seed,
                    ),
                )
            ]
    if config.family == "linear":
        return [
            (f"logistic_l1_C{c}", _logistic("l1", c, config.random_seed))
            for c in (0.01, 0.1, 1.0, 10.0)
        ] + [
            (f"logistic_l2_C{c}", _logistic("l2", c, config.random_seed))
            for c in (0.01, 0.1, 1.0, 10.0)
        ]
    if config.family == "tree":
        candidates: list[tuple[str, object]] = [
            (
                f"hist_gradient_leaf{leaf}_lr{rate}",
                HistGradientBoostingClassifier(
                    max_leaf_nodes=leaf,
                    learning_rate=rate,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=config.random_seed,
                ),
            )
            for leaf in (7, 15, 31)
            for rate in (0.03, 0.08)
        ] + [
            (
                f"extra_trees_leaf{leaf}",
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=leaf,
                    class_weight="balanced",
                    n_jobs=config.cpu_threads,
                    random_state=config.random_seed,
                ),
            )
            for leaf in (2, 10)
        ]
        if config.search != "standard":
            candidates.extend(
                (
                    f"extra_trees_trees600_leaf{leaf}_features{features}",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        min_samples_leaf=leaf,
                        max_features=features,
                        class_weight="balanced",
                        n_jobs=config.cpu_threads,
                        random_state=config.random_seed,
                    ),
                )
                for leaf in (1, 5)
                for features in ("sqrt", 0.3)
            )
        return candidates
    if config.family == "random_forest":
        candidates = [
            (
                f"random_forest_trees{trees}_leaf{leaf}",
                RandomForestClassifier(
                    n_estimators=trees,
                    min_samples_leaf=leaf,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=config.cpu_threads,
                    random_state=config.random_seed,
                ),
            )
            for trees in (300, 600)
            for leaf in (1, 5)
        ]
        if config.search != "standard":
            candidates.extend(
                (
                    f"random_forest_trees600_leaf{leaf}_features{features}",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=leaf,
                        max_features=features,
                        class_weight="balanced",
                        n_jobs=config.cpu_threads,
                        random_state=config.random_seed,
                    ),
                )
                for leaf in (1, 5)
                for features in ("sqrt", 0.3)
            )
        return candidates
    if config.family == "svm":
        return [
            (
                f"svm_{kernel}_C{c}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            SVC(
                                C=c,
                                kernel=kernel,
                                class_weight="balanced",
                                probability=True,
                                random_state=config.random_seed,
                            ),
                        ),
                    ]
                ),
            )
            for kernel, c in (("linear", 0.1), ("linear", 1.0), ("rbf", 0.3), ("rbf", 1.0))
        ]
    if config.family == "knn":
        return [
            (
                f"knn_k{neighbors}_{weights}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            KNeighborsClassifier(
                                n_neighbors=neighbors,
                                weights=weights,
                                n_jobs=config.cpu_threads,
                            ),
                        ),
                    ]
                ),
            )
            for neighbors, weights in ((5, "distance"), (15, "distance"), (31, "uniform"))
        ]
    if config.family == "naive_bayes":
        return [
            (f"gaussian_nb_smoothing{smoothing}", GaussianNB(var_smoothing=smoothing))
            for smoothing in (1e-10, 1e-8, 1e-6, 1e-4)
        ]
    if config.family == "cnn":
        cnn_architectures: tuple[tuple[int, float, int, int], ...] = (
            (32, 0.1, 5, 2),
            (64, 0.3, 5, 2),
        )
        if config.search == "expanded":
            cnn_architectures += ((48, 0.2, 3, 3), (64, 0.2, 7, 3))
        return [
            (
                f"cnn_channels{channels}_dropout{dropout}_kernel{kernel}_depth{depth}_seed{seed}",
                _CnnCandidate(
                    channels,
                    dropout,
                    seed,
                    config.device,
                    f"cnn_channels{channels}_dropout{dropout}_kernel{kernel}_depth{depth}_seed{seed}",
                    progress_callback,
                    kernel,
                    depth,
                ),
            )
            for channels, dropout, kernel, depth in cnn_architectures
            for seed in (config.random_seed,)
        ]
    if config.family == "embedding_cnn":
        return [
            (
                f"embedding_cnn_channels{channels}_dropout{dropout}_kernel{kernel}_depth{depth}",
                _CnnCandidate(
                    channels,
                    dropout,
                    config.random_seed,
                    config.device,
                    f"embedding_cnn_channels{channels}_dropout{dropout}_kernel{kernel}_depth{depth}",
                    progress_callback,
                    kernel,
                    depth,
                    1,
                ),
            )
            for channels, dropout, kernel, depth in ((32, 0.1, 5, 2), (64, 0.2, 3, 3))
        ]
    mlp_architectures: tuple[tuple[int, float, str, tuple[int, ...]], ...] = (
        (32, 0.1, "relu", (256, 32)),
        (64, 0.1, "relu", (256, 64)),
        (32, 0.1, "gelu", (256, 32)),
        (64, 0.1, "gelu", (256, 64)),
        (32, 0.1, "silu", (256, 32)),
        (64, 0.1, "silu", (256, 64)),
        (32, 0.3, "relu", (256, 32)),
        (64, 0.3, "relu", (256, 64)),
        (32, 0.3, "gelu", (256, 32)),
        (64, 0.3, "gelu", (256, 64)),
        (32, 0.3, "silu", (256, 32)),
        (64, 0.3, "silu", (256, 64)),
    )
    if config.search != "standard":
        mlp_architectures += (
            (128, 0.1, "gelu", (512, 256, 128)),
            (256, 0.2, "silu", (512, 256)),
            (128, 0.3, "relu", (256, 128, 64)),
        )
    return [
        (
            f"mlp_layers{'-'.join(map(str, widths))}_dropout{dropout}_{activation}",
            _MlpCandidate(
                hidden,
                dropout,
                activation,
                config.random_seed,
                config.device,
                f"mlp_layers{'-'.join(map(str, widths))}_dropout{dropout}_{activation}",
                progress_callback,
                widths,
            ),
        )
        for hidden, dropout, activation, widths in mlp_architectures
    ]


def _logistic(penalty: str, c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    solver="saga",
                    l1_ratio=1.0 if penalty == "l1" else 0.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


class _MlpCandidate:
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        activation: str,
        seed: int,
        device: str,
        name: str = "mlp",
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        hidden_dims: tuple[int, ...] | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation
        self.seed = seed
        self.device = str(resolve_device(device))
        self.scaler = StandardScaler()
        self.network: FeatureMLP | None = None
        self.history: list[dict[str, float]] = []
        self.name = name
        self.progress_callback = progress_callback
        self.hidden_dims = hidden_dims

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _MlpCandidate:
        return self.fit_with_validation(features, labels, features, labels)

    def fit_with_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> _MlpCandidate:
        self.scaler.fit(features)
        torch.manual_seed(self.seed)
        self.network = FeatureMLP(
            features.shape[1],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,  # type: ignore[arg-type]
            hidden_dims=self.hidden_dims,
        )
        tensor = torch.from_numpy(self.scaler.transform(features).astype(np.float32))
        label_tensor = torch.from_numpy(labels.astype(np.float32))
        validation_tensor = torch.from_numpy(
            self.scaler.transform(validation_features).astype(np.float32)
        )
        validation_label_tensor = torch.from_numpy(validation_labels.astype(np.float32))
        result = train_feature_mlp(
            self.network,
            tensor,
            label_tensor,
            validation_tensor,
            validation_label_tensor,
            max_epochs=80,
            patience=8,
            learning_rate=1e-3,
            device=self.device,
            progress_callback=self._report_progress,
        )
        self.history = result.history
        return self

    def _report_progress(self, values: dict[str, float]) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"candidate": self.name, **values})

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("MLP must be fitted before prediction")
        self.network.eval()
        with torch.no_grad():
            logits = self.network(
                torch.from_numpy(self.scaler.transform(features).astype(np.float32)).to(self.device)
            )
        probability = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probability, probability])


class _CnnCandidate:
    """Sklearn-like facade so the sequence CNN uses the shared benchmark loop."""

    def __init__(
        self,
        channels: int,
        dropout: float,
        seed: int,
        device: str,
        name: str = "cnn",
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        kernel_size: int = 5,
        depth: int = 2,
        input_channels: int = 21,
    ):
        self.channels = channels
        self.dropout = dropout
        self.seed = seed
        self.device = str(resolve_device(device))
        self.network: SequenceCNN | None = None
        self.history: list[dict[str, float]] = []
        self.name = name
        self.progress_callback = progress_callback
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_channels = input_channels

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _CnnCandidate:
        return self.fit_with_validation(features, labels, features, labels)

    def fit_with_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> _CnnCandidate:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError("CNN features must have shape [proteins, channels, residues]")
        torch.manual_seed(self.seed)
        self.network = SequenceCNN(
            self.channels,
            self.dropout,
            self.kernel_size,
            self.depth,
            self.input_channels,
        )
        result = train_feature_mlp(
            self.network,
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
            torch.from_numpy(validation_features.astype(np.float32)),
            torch.from_numpy(validation_labels.astype(np.float32)),
            max_epochs=40,
            patience=5,
            learning_rate=1e-3,
            device=self.device,
            progress_callback=self._report_progress,
        )
        self.history = result.history
        return self

    def _report_progress(self, values: dict[str, float]) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"candidate": self.name, **values})

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("CNN must be fitted before prediction")
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.from_numpy(features.astype(np.float32)).to(self.device))
        probability = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probability, probability])


def _fit_candidate(
    candidate: Any, features: np.ndarray, labels: np.ndarray, config: BenchmarkConfig
) -> Any:
    del config
    return candidate.fit(features, labels)


def _predict(candidate: Any, features: np.ndarray) -> np.ndarray:
    return candidate.predict_proba(features)[:, 1]
