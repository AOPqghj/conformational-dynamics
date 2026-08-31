"""Small, composable methods shared by the three interpretability workstreams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LinearProbeConfig:
    """One fixed linear-probe candidate selected outside the held-out test set."""

    regularization_c: float = 1.0
    penalty: str = "l2"
    class_weight: str | dict[int, float] | None = "balanced"
    random_seed: int = 42
    max_iter: int = 5000

    def __post_init__(self) -> None:
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.penalty not in {"l1", "l2"}:
            raise ValueError("penalty must be l1 or l2")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True, slots=True)
class SparseAutoencoderConfig:
    """Architecture and loss weights for a tied-shape residue SAE."""

    input_dim: int = 1024
    latent_dim: int = 4096
    l1_coefficient: float = 1e-3
    top_k: int | None = None
    unit_decoder_norm: bool = True

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.latent_dim < 1:
            raise ValueError("SAE dimensions must be positive")
        if self.l1_coefficient < 0:
            raise ValueError("l1_coefficient must be non-negative")
        if self.top_k is not None and not 1 <= self.top_k <= self.latent_dim:
            raise ValueError("top_k must be between 1 and latent_dim")
        if self.top_k is not None and self.l1_coefficient != 0:
            raise ValueError("TopK SAE must use l1_coefficient=0")


@dataclass(frozen=True, slots=True)
class SparseAutoencoderLoss:
    """Named SAE loss components for stable logging."""

    total: Tensor
    reconstruction: Tensor
    sparsity: Tensor


class SparseAutoencoder(nn.Module):
    """A minimal ReLU sparse autoencoder for frozen residue activations."""

    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.config = config
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.encoder = nn.Linear(config.input_dim, config.latent_dim, bias=True)
        self.decoder = nn.Linear(config.latent_dim, config.input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder_()

    def encode(self, values: Tensor) -> Tensor:
        self._validate_values(values)
        latents = torch.relu(self.encoder(values - self.pre_bias))
        if self.config.top_k is None:
            return latents
        indices = latents.topk(self.config.top_k, dim=-1).indices
        return torch.zeros_like(latents).scatter_(1, indices, latents.gather(1, indices))

    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.config.latent_dim:
            raise ValueError("latents must have shape [items, latent_dim]")
        return self.decoder(latents) + self.pre_bias

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(values)
        return self.decode(latents), latents

    def loss(self, values: Tensor) -> SparseAutoencoderLoss:
        reconstruction, latents = self(values)
        reconstruction_loss = (reconstruction - values).square().mean()
        sparsity_loss = latents.abs().mean()
        total = reconstruction_loss + self.config.l1_coefficient * sparsity_loss
        return SparseAutoencoderLoss(total, reconstruction_loss, sparsity_loss)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project decoder feature vectors to unit norm after each optimizer step."""
        if self.config.unit_decoder_norm:
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def _validate_values(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.config.input_dim:
            raise ValueError("values must have shape [items, input_dim]")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[str],
    config: LinearProbeConfig,
) -> Pipeline:
    """Fit one fixed probe on train only and never consume validation or test rows."""
    matrix, targets, split_values = _validate_supervised_arrays(features, labels, splits)
    train = split_values == "train"
    if np.unique(targets[train]).size != 2:
        raise ValueError("training rows must contain both classes")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=config.regularization_c,
                    l1_ratio=1.0 if config.penalty == "l1" else 0.0,
                    solver="saga",
                    class_weight=config.class_weight,
                    random_state=config.random_seed,
                    max_iter=config.max_iter,
                ),
            ),
        ]
    )
    model.fit(matrix[train], targets[train])
    return model


def predict_partition(
    model: Pipeline,
    features: np.ndarray,
    splits: Sequence[str],
    partition: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices and positive-class probabilities for one named partition."""
    matrix = np.asarray(features)
    split_values = np.asarray(splits, dtype=str)
    if matrix.ndim != 2 or matrix.shape[0] != split_values.size:
        raise ValueError("features and splits must align")
    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    indices = np.flatnonzero(split_values == partition)
    if indices.size == 0:
        raise ValueError(f"partition is empty: {partition}")
    return indices, model.predict_proba(matrix[indices])[:, 1]


def balanced_residue_sample(
    matrices: Sequence[np.ndarray],
    *,
    residues_per_protein: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample at most an equal residue count per protein for SAE fitting.

    The returned second array stores the source protein index for every sampled row.
    """
    if residues_per_protein < 1:
        raise ValueError("residues_per_protein must be positive")
    if not matrices:
        raise ValueError("at least one matrix is required")
    rng = np.random.default_rng(random_seed)
    rows: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    width: int | None = None
    for protein_index, values in enumerate(matrices):
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
            raise ValueError("every residue matrix must be finite with shape [positive L, D]")
        if width is None:
            width = matrix.shape[1]
        elif matrix.shape[1] != width:
            raise ValueError("residue matrices must share one embedding width")
        count = min(residues_per_protein, matrix.shape[0])
        selected = rng.choice(matrix.shape[0], size=count, replace=False)
        rows.append(matrix[selected].astype(np.float32, copy=False))
        owners.append(np.full(count, protein_index, dtype=np.int64))
    return np.concatenate(rows), np.concatenate(owners)


def window_ablation_effects(
    values: np.ndarray,
    scorer: Callable[[np.ndarray], float],
    *,
    window_size: int,
    replacement: np.ndarray,
) -> np.ndarray:
    """Measure score drops after replacing each centered residue window.

    ``replacement`` must be a training-derived vector, such as the train-residue mean.
    Positive effects mean the original window supported the unablated score.
    """
    matrix = np.asarray(values, dtype=np.float32)
    baseline = np.asarray(replacement, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("values must be a finite [positive L, D] matrix")
    if baseline.shape != (matrix.shape[1],) or not np.isfinite(baseline).all():
        raise ValueError("replacement must be one finite vector matching the embedding width")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    original_score = float(scorer(matrix))
    if not np.isfinite(original_score):
        raise ValueError("scorer returned a non-finite baseline score")
    radius = window_size // 2
    effects = np.empty(matrix.shape[0], dtype=np.float64)
    for index in range(matrix.shape[0]):
        lower = max(0, index - radius)
        upper = min(matrix.shape[0], index + radius + 1)
        ablated = matrix.copy()
        ablated[lower:upper] = baseline
        score = float(scorer(ablated))
        if not np.isfinite(score):
            raise ValueError(f"scorer returned a non-finite ablation score at residue {index}")
        effects[index] = original_score - score
    return effects


def top_fraction_enrichment(
    scores: np.ndarray,
    annotations: np.ndarray,
    *,
    fraction: float = 0.1,
) -> float:
    """Compare annotation prevalence in top-scoring residues with the full protein."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(annotations)
    if values.ndim != 1 or labels.shape != values.shape or values.size == 0:
        raise ValueError("scores and annotations must be aligned non-empty vectors")
    if not np.isfinite(values).all() or not np.isin(labels, (0, 1, False, True)).all():
        raise ValueError("scores must be finite and annotations must be binary")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    prevalence = float(labels.mean())
    if prevalence == 0:
        raise ValueError("enrichment is undefined when no residues are annotated")
    count = max(1, int(np.ceil(values.size * fraction)))
    selected = np.argpartition(values, -count)[-count:]
    return float(labels[selected].mean() / prevalence)


def paired_sign_flip_pvalue(
    differences: np.ndarray, *, random_seed: int, draws: int = 10000
) -> float:
    """Two-sided paired randomization p-value across proteins or saved splits."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite vector with at least two pairs")
    if draws < 1:
        raise ValueError("draws must be positive")
    rng = np.random.default_rng(random_seed)
    observed = abs(float(values.mean()))
    extreme = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=values.size)
        extreme += abs(float((values * signs).mean())) >= observed
    return float((extreme + 1) / (draws + 1))


def _validate_supervised_arrays(
    features: np.ndarray, labels: np.ndarray, splits: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels)
    split_values = np.asarray(splits, dtype=str)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("features must be a finite two-dimensional matrix")
    if targets.shape != (matrix.shape[0],) or split_values.shape != (matrix.shape[0],):
        raise ValueError("features, labels, and splits must align")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("labels must be binary")
    if set(split_values) != {"train", "val", "test"}:
        raise ValueError("splits must contain exactly train, val, and test")
    return matrix, targets.astype(np.int64), split_values
