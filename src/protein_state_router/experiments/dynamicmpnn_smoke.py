"""Small, explicitly synthetic DynamicMPNN classification smoke-test helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from protein_state_router.models.baselines import ModelConfig, ModelExample, create_model

METADATA_FEATURES = (
    "sequence_length",
    "n_available_conformations",
    "min_pair_tm",
    "max_pair_tm",
    "mean_pair_tm",
    "max_ca_rmsd",
    "mean_ca_rmsd",
    "min_aligned_coverage",
)


def select_temporary_dataset(
    catalog: pd.DataFrame,
    sample_size: int = 20,
    seed: int = 42,
    max_sequence_length: int = 250,
) -> pd.DataFrame:
    """Select short valid candidates and attach balanced synthetic smoke-test labels."""
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    eligible = catalog.loc[
        catalog.sequence.notna()
        & ~catalog.load_failed
        & catalog.exclusion_reason.isna()
        & catalog.sequence_hash.notna()
        & catalog.sequence_length.le(max_sequence_length)
    ].drop_duplicates("sequence_hash")
    if len(eligible) < sample_size or sample_size % 2:
        raise ValueError(
            "need an even sample_size with enough unique valid candidate sequences "
            f"at or below {max_sequence_length} residues"
        )
    selected = eligible.sample(sample_size, random_state=seed).reset_index(drop=True).copy()
    labels = np.repeat([0, 1], sample_size // 2)
    selected["synthetic_label"] = np.random.default_rng(seed).permutation(labels)
    selected["label_note"] = "Synthetic smoke-test label; not a biological router label."
    return selected


def metadata_loocv(
    selected: pd.DataFrame, regularization_strength: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe L1/L2 LOOCV over catalog metadata only."""
    feature_names = available_metadata_features(selected)
    features = selected.loc[:, feature_names]
    labels = selected.synthetic_label.to_numpy(dtype=int)
    predictions, summary = _loocv(
        features,
        labels,
        selected.dynamicmpnn_cluster_id.astype(str).to_numpy(),
        lambda l1_ratio: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0 / regularization_strength,
                solver="saga",
                l1_ratio=l1_ratio,
                class_weight="balanced",
                random_state=42,
                max_iter=2000,
            ),
        ),
        {"lasso": 1.0, "ridge": 0.0},
    )
    summary["metadata_features"] = ",".join(feature_names)
    return predictions, summary


def embedding_loocv(
    examples: Sequence[ModelExample], regularization_strength: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the project pooled-embedding Lasso/Ridge models in each LOOCV fold."""
    if any(item.label is None for item in examples):
        raise ValueError("all embedding smoke-test examples need synthetic labels")
    labels = np.asarray([item.label for item in examples], dtype=int)
    protein_ids = np.asarray([item.protein_id for item in examples])
    predictions: list[dict[str, object]] = []
    for kind in ("lasso", "ridge"):
        config = ModelConfig(
            kind=kind,
            regularization_strength=regularization_strength,
            class_weight="balanced",
            random_state=42,
        )
        for fold, (train_index, test_index) in enumerate(LeaveOneOut().split(protein_ids), start=1):
            model = create_model(config).fit([examples[index] for index in train_index])
            held_out = [examples[test_index[0]]]
            predictions.append(
                _prediction_row(
                    kind,
                    fold,
                    protein_ids[test_index[0]],
                    labels[test_index[0]],
                    float(model.predict_proba(held_out)[0]),
                    len(train_index),
                )
            )
    frame = pd.DataFrame(predictions)
    return frame, _summary(frame, {"regularization_strength": regularization_strength})


def model_parameters(regularization_strength: float = 10.0) -> pd.DataFrame:
    """Return the shared Lasso/Ridge baseline settings shown by the notebook."""
    return pd.DataFrame(
        [
            asdict(
                ModelConfig(
                    kind=kind,
                    regularization_strength=regularization_strength,
                    class_weight="balanced",
                    random_state=42,
                )
            )
            for kind in ("lasso", "ridge")
        ]
    )


def _loocv(features, labels, identifiers, estimator_factory, penalties):
    predictions: list[dict[str, object]] = []
    for model_kind, penalty in penalties.items():
        for fold, (train_index, test_index) in enumerate(LeaveOneOut().split(features), start=1):
            estimator = estimator_factory(penalty).fit(
                features.iloc[train_index], labels[train_index]
            )
            probability = float(estimator.predict_proba(features.iloc[test_index])[0, 1])
            predictions.append(
                _prediction_row(
                    model_kind,
                    fold,
                    identifiers[test_index[0]],
                    labels[test_index[0]],
                    probability,
                    len(train_index),
                )
            )
    frame = pd.DataFrame(predictions)
    return frame, _summary(frame, {"feature_source": "dynamicmpnn_catalog_metadata"})


def available_metadata_features(frame: pd.DataFrame) -> tuple[str, ...]:
    """Keep only catalog metadata columns containing at least one observed value."""
    features = tuple(name for name in METADATA_FEATURES if frame[name].notna().any())
    if not features:
        raise ValueError("DynamicMPNN metadata baseline has no observed numeric features")
    return features


def _prediction_row(model_kind, fold, protein_id, label, probability, train_size):
    return {
        "model_kind": model_kind,
        "fold": fold,
        "protein_id": protein_id,
        "true_label": label,
        "probability": probability,
        "predicted_label": int(probability >= 0.5),
        "train_size": train_size,
    }


def _summary(predictions: pd.DataFrame, extra: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_kind": kind,
                "accuracy": accuracy_score(group.true_label, group.predicted_label),
                "balanced_accuracy": balanced_accuracy_score(
                    group.true_label, group.predicted_label
                ),
                "n_folds": len(group),
                **extra,
            }
            for kind, group in predictions.groupby("model_kind", sort=False)
        ]
    )
