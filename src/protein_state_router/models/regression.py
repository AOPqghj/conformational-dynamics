"""Compact scalar and residue-field regressors over frozen residue embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def masked_mean_max(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError(
            "values and mask must be [batch, residues, features] and [batch, residues]"
        )
    expanded = mask[:, :, None]
    mean = (values * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
    maximum = values.masked_fill(~expanded, float("-inf")).amax(dim=1)
    return torch.cat((mean, maximum), dim=1)


class AttentionScalarRegressor(nn.Module):
    """Attention-pool a residue matrix into one scalar RMSD prediction."""

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int = 128,
        heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, heads),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim * heads),
            nn.Linear(embedding_dim * heads, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        scores = self.score(values).transpose(1, 2).masked_fill(~mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, values).flatten(1)
        return self.head(pooled).squeeze(-1)


class CnnScalarRegressor(nn.Module):
    """Masked depthwise CNN with scalar regression head."""

    def __init__(
        self,
        embedding_dim: int,
        channels: int = 128,
        depth: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd")
        layers: list[nn.Module] = [nn.Conv1d(embedding_dim, channels, 1), nn.GELU()]
        for _ in range(depth):
            layers.extend(
                (
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        groups=channels,
                    ),
                    nn.Conv1d(channels, channels, 1),
                    nn.GELU(),
                )
            )
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        expanded = mask[:, None, :]
        encoded = (values * mask[:, :, None]).transpose(1, 2)
        for layer in self.features:
            encoded = layer(encoded) * expanded
        return masked_mean_max(encoded.transpose(1, 2), mask)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.head(self.encode(values, mask)).squeeze(-1)


class SinusoidalPositions(nn.Module):
    def __init__(self, width: int, maximum_length: int = 2048) -> None:
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32)[:, None]
        scale = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
        )
        values = torch.zeros(maximum_length, width)
        values[:, 0::2] = torch.sin(position * scale)
        values[:, 1::2] = torch.cos(position * scale[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[1] > self.values.shape[0]:
            raise ValueError("sequence exceeds positional encoding limit")
        return values + self.values[: values.shape[1]]


class AttentionVectorRegressor(nn.Module):
    """Predict a local-frame displacement vector at each residue."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        depth: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("attention hidden dimension must be divisible by heads")
        self.input = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim)
        )
        self.positions = SinusoidalPositions(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        encoded = self.positions(self.input(values))
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        return self.output(encoded) * mask[:, :, None]


class CnnVectorRegressor(nn.Module):
    """Predict a local-frame displacement vector with a masked residue CNN."""

    def __init__(
        self,
        embedding_dim: int,
        channels: int = 128,
        depth: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel size must be odd")
        self.input = nn.Conv1d(embedding_dim, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    groups=channels,
                ),
                nn.Conv1d(channels, channels, 1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(depth)
        )
        self.output = nn.Conv1d(channels, 3, 1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        expanded = mask[:, None, :]
        encoded = self.input(values.transpose(1, 2)) * expanded
        for block in self.blocks:
            encoded = (encoded + block(encoded)) * expanded
        return self.output(encoded).transpose(1, 2) * mask[:, :, None]


def bidirectional_vector_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Protein-weighted vector MSE, invariant to one global sign per protein."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must share shape [batch, residues, 3]")
    if mask.shape != prediction.shape[:2] or not torch.all(mask.any(dim=1)):
        raise ValueError("each protein requires at least one valid vector target")
    denominator = mask.sum(dim=1).clamp_min(1) * 3
    positive = (((prediction - target) ** 2) * mask[:, :, None]).sum(dim=(1, 2)) / denominator
    negative = (((prediction + target) ** 2) * mask[:, :, None]).sum(dim=(1, 2)) / denominator
    return torch.minimum(positive, negative).mean()


def align_vector_field_sign(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Choose the target-compatible global sign independently for each protein."""
    positive = (((prediction - target) ** 2) * mask[:, :, None]).sum(dim=(1, 2))
    negative = (((prediction + target) ** 2) * mask[:, :, None]).sum(dim=(1, 2))
    sign = torch.where(positive <= negative, 1.0, -1.0)
    return prediction * sign[:, None, None]
