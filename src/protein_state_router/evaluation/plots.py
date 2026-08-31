"""Reliability diagram helper."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def reliability_diagram(labels: np.ndarray, probabilities: np.ndarray):
    figure, axis = plt.subplots()
    bins = np.linspace(0, 1, 11)
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= low) & (probabilities <= high)
        if mask.any():
            axis.scatter(probabilities[mask].mean(), labels[mask].mean())
    axis.plot([0, 1], [0, 1], "--", color="black")
    axis.set(xlabel="Predicted probability", ylabel="Observed frequency")
    return figure


def classification_figures(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, plt.Figure]:
    """Return compact ROC, precision-recall, and confusion-matrix figures."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    figures: dict[str, plt.Figure] = {}
    figure, axis = plt.subplots()
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axis)
    axis.set_title("ROC curve")
    figures["roc"] = figure
    figure, axis = plt.subplots()
    PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=axis)
    axis.set_title("Precision-recall curve")
    figures["precision_recall"] = figure
    figure, axis = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(labels, probabilities >= 0.5, ax=axis)
    axis.set_title("Confusion matrix")
    figures["confusion_matrix"] = figure
    return figures
