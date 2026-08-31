"""Small paired-inference utilities with protein/run-level resampling units."""

from __future__ import annotations

import itertools

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be one finite vector in [0, 1]")
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_sign_flip_test(
    differences: np.ndarray,
    *,
    seed: int = 42,
    max_exact_pairs: int = 20,
    draws: int = 100_000,
) -> dict[str, float | int | str]:
    """Two-sided paired randomization test for a mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two finite paired differences are required")
    observed = float(values.mean())
    if len(values) <= max_exact_pairs:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
        null = (signs * values).mean(axis=1)
        p_value = float(np.mean(np.abs(null) >= abs(observed)))
        method = "exact_paired_sign_flip"
        permutations = len(null)
    else:
        rng = np.random.default_rng(seed)
        exceedances = 0
        completed = 0
        while completed < draws:
            count = min(10_000, draws - completed)
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=(count, len(values)))
            exceedances += int(
                np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= abs(observed))
            )
            completed += count
        p_value = (exceedances + 1) / (draws + 1)
        method = "monte_carlo_paired_sign_flip"
        permutations = draws
    return {
        "n_pairs": len(values),
        "mean_difference": observed,
        "median_difference": float(np.median(values)),
        "permutation_p_two_sided": p_value,
        "permutation_method": method,
        "permutations": permutations,
    }


def paired_bootstrap_interval(
    first: np.ndarray,
    second: np.ndarray,
    metric,
    *,
    draws: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a paired metric difference using the observation as the unit."""
    first_values = np.asarray(first)
    second_values = np.asarray(second)
    if len(first_values) != len(second_values) or len(first_values) < 2:
        raise ValueError("paired bootstrap inputs must have equal nontrivial length")
    rng = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.integers(0, len(first_values), size=len(first_values))
        differences[draw] = metric(first_values[sample]) - metric(second_values[sample])
    return tuple(np.quantile(differences, (0.025, 0.975)).astype(float))
