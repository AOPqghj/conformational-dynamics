"""Bootstrap intervals for metrics when real datasets are large enough."""

from collections.abc import Callable

import numpy as np


def bootstrap_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    rng, scores = np.random.default_rng(seed), []
    for _ in range(n):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            scores.append(metric(labels[indices], probabilities[indices]))
    return tuple(np.quantile(scores, [0.025, 0.975]).tolist())
