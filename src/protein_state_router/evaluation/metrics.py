"""Classification, calibration, and routing-oriented metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= low) & (
            (probabilities < high) if high < 1 else (probabilities <= high)
        )
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or len(labels) != len(probabilities)
        or not len(labels)
    ):
        raise ValueError("labels and probabilities must be non-empty, aligned 1D arrays")
    if not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(probabilities).all():
        raise ValueError("labels must be binary and probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    predicted = probabilities >= 0.5
    result = {
        "sample_count": float(len(labels)),
        "positive_prevalence": float(labels.mean()),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    )
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    qualifying = recall[precision >= 0.8]
    result["recall_at_precision_0_8"] = float(qualifying.max()) if len(qualifying) else 0.0
    return result
