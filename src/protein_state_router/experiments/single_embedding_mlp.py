"""Bounded validation sweep for pooled frozen single-residue embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.models.baselines import EmbeddingClassifier, ModelConfig, ModelExample
from protein_state_router.representations.embeddings import ProteinEmbeddings


@dataclass(frozen=True, slots=True)
class MLPTrial:
    """One deliberately small MLP setting for validation-only selection."""

    hidden_dim: int
    dropout: float
    learning_rate: float
    random_state: int
    activation: str = "gelu"


DEFAULT_MLP_TRIALS = (
    MLPTrial(32, 0.20, 1e-3, 11, "relu"),
    MLPTrial(32, 0.20, 1e-3, 23, "gelu"),
    MLPTrial(32, 0.20, 1e-3, 37, "silu"),
    MLPTrial(64, 0.40, 3e-4, 53, "relu"),
    MLPTrial(64, 0.40, 3e-4, 67, "gelu"),
    MLPTrial(64, 0.40, 3e-4, 79, "silu"),
)


@dataclass(slots=True)
class MLPSweepResult:
    """The selected model plus compact validation-only trial records."""

    model: EmbeddingClassifier
    trial: MLPTrial
    validation_probabilities: np.ndarray
    trial_metrics: list[dict[str, object]]
    learning_curves: list[dict[str, object]]


def mlp_sweep_settings(config: dict[str, Any]) -> tuple[tuple[MLPTrial, ...], int, int]:
    """Parse the compact public YAML configuration for this validation sweep."""
    values = config.get("mlp_sweep", config)
    if values.get("selection_metric", "auprc") != "auprc":
        raise ValueError("the MLP sweep selects models only by validation AUPRC")
    try:
        trials = tuple(MLPTrial(**trial) for trial in values["trials"])
    except (KeyError, TypeError) as error:
        raise ValueError("mlp_sweep requires a non-empty trials list") from error
    if not trials:
        raise ValueError("mlp_sweep requires at least one trial")
    return trials, int(values.get("max_epochs", 100)), int(values.get("patience", 10))


def single_only_examples(examples: Sequence[ModelExample]) -> list[ModelExample]:
    """Drop pair tensors while preserving single embeddings and numeric metadata."""
    result: list[ModelExample] = []
    for example in examples:
        embedding = example.embeddings
        if embedding.single is None:
            raise ValueError(f"{example.protein_id} has no single embedding")
        single_only = ProteinEmbeddings(
            embedding.protein_id,
            embedding.sequence,
            embedding.sequence_sha256,
            embedding.source,
            embedding.single,
            None,
            embedding.confidence_features,
            embedding.metadata,
        )
        result.append(
            ModelExample(example.protein_id, single_only, example.label, example.metadata)
        )
    return result


def run_single_embedding_mlp_sweep(
    train_examples: Sequence[ModelExample],
    validation_examples: Sequence[ModelExample],
    *,
    trials: Sequence[MLPTrial] = DEFAULT_MLP_TRIALS,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
) -> MLPSweepResult:
    """Fit a bounded MLP sweep and select strictly by validation AUPRC.

    All inputs are converted to pooled single-residue views, so this routine
    never pools pair embeddings or builds a CNN.  Ties are resolved by the
    declared trial order to make the selected artifact reproducible.
    """
    if not trials:
        raise ValueError("at least one MLP trial is required")
    train, validation = (
        single_only_examples(train_examples),
        single_only_examples(validation_examples),
    )
    labels = _labels(validation)
    records: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    best: tuple[float, EmbeddingClassifier, MLPTrial, np.ndarray] | None = None
    for index, trial in enumerate(trials):
        model = EmbeddingClassifier(
            ModelConfig(
                kind="mlp",
                hidden_dim=trial.hidden_dim,
                dropout=trial.dropout,
                activation=trial.activation,  # type: ignore[arg-type]
                learning_rate=trial.learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                random_state=trial.random_state,
                device=device,
            )
        ).fit(train, validation)
        probabilities = model.predict_proba(validation)
        metrics = classification_metrics(labels, probabilities)
        records.append({"trial_index": index, **asdict(trial), **metrics})
        assert model.train_result is not None
        curves.extend(
            {"trial_index": index, **asdict(trial), **point} for point in model.train_result.history
        )
        score = float(metrics["auprc"])
        if best is None or score > best[0]:
            best = score, model, trial, probabilities
    assert best is not None
    _, model, trial, probabilities = best
    return MLPSweepResult(model, trial, probabilities, records, curves)


def _labels(examples: Sequence[ModelExample]) -> np.ndarray:
    if not examples or any(example.label is None for example in examples):
        raise ValueError("validation examples require binary labels")
    labels: list[int] = []
    for example in examples:
        assert example.label is not None
        labels.append(example.label)
    return np.asarray(labels, dtype=np.int64)
