"""Compact full-batch trainer for small MVP feature models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from protein_state_router.training.losses import binary_loss
from sklearn.metrics import average_precision_score
from torch import Tensor, nn


@dataclass
class TrainResult:
    best_epoch: int
    validation_auprc: float
    history: list[dict[str, float]] = field(default_factory=list)


def resolve_device(requested: str | torch.device = "cpu") -> torch.device:
    """Resolve a requested Torch device, including Apple Silicon MPS."""
    value = str(requested)
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch installation")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch installation")
    if value not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    return torch.device(value)


def train_feature_mlp(
    model: nn.Module,
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    max_epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> TrainResult:
    """Train models accepting a single feature tensor and producing logits."""
    target = resolve_device(device)
    model.to(target)
    train_features, train_labels = train_features.to(target), train_labels.to(target)
    validation_features, validation_labels = (
        validation_features.to(target),
        validation_labels.to(target),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state, best_score, stale, best_epoch = None, -float("inf"), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = binary_loss(model(train_features).squeeze(-1), train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(validation_features).squeeze(-1)).cpu().numpy()
        score = average_precision_score(validation_labels.cpu().numpy(), probability)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss.detach().cpu()),
                "validation_auprc": float(score),
            }
        )
        if progress_callback is not None:
            progress_callback(history[-1])
        if score > best_score:
            best_score, best_state, stale, best_epoch = (
                score,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                0,
                epoch,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(best_epoch, float(best_score), history)
