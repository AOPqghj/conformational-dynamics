"""Small neural probes for pooled features and raw protein sequences."""

from typing import Literal

import torch
from torch import Tensor, nn

ActivationName = Literal["relu", "gelu", "silu"]


def _activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


class FeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
        activation: ActivationName = "gelu",
        hidden_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()
        widths = hidden_dims or (256, hidden_dim)
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("hidden_dims must contain positive widths")
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for index, width in enumerate(widths):
            layers.extend((nn.Linear(previous, width), _activation(activation)))
            layers.append(nn.Dropout(dropout if index == 0 else dropout / 2))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class SequenceCNN(nn.Module):
    """Compact masked 1D CNN for fixed-width one-hot protein sequences."""

    def __init__(
        self,
        channels: int = 32,
        dropout: float = 0.15,
        kernel_size: int = 5,
        depth: int = 2,
        input_channels: int = 21,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0 or depth <= 0 or input_channels <= 0:
            raise ValueError("kernel_size must be odd; depth and input_channels must be positive")
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend(
                (
                    nn.Conv1d(
                        input_channels if index == 0 else channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
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

    def forward(self, sequences: Tensor) -> Tensor:
        """Return logits for ``[batch, 21, residues]`` one-hot inputs.

        Padding is all zeros; masked mean/max pooling excludes padded positions
        from the protein-level representation.
        """
        mask = sequences.sum(dim=1, keepdim=True).gt(0)
        encoded = self.features(sequences)
        mean = (encoded * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        maximum = encoded.masked_fill(~mask, float("-inf")).amax(dim=-1)
        return self.head(torch.cat((mean, maximum), dim=1)).squeeze(-1)


class AttentionPoolClassifier(nn.Module):
    """Classify a protein from masked per-residue frozen embeddings."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        attention_dim: int = 128,
        heads: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        include_global_stats: bool = False,
    ):
        super().__init__()
        if min(embedding_dim, attention_dim, heads, hidden_dim) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.score = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, heads),
        )
        self.include_global_stats = include_global_stats
        encoded_dim = embedding_dim * (heads + (2 if include_global_stats else 0))
        self.head = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.head(self.encode(values, mask)).squeeze(-1)

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError(
                "values and mask must have shapes [batch, residues, features] and [batch, residues]"
            )
        scores = self.score(values).transpose(1, 2).masked_fill(~mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, values).flatten(1)
        if not self.include_global_stats:
            return pooled
        expanded = mask[:, :, None]
        mean = (values * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
        maximum = values.masked_fill(~expanded, float("-inf")).amax(dim=1)
        return torch.cat((pooled, mean, maximum), dim=1)


class SegmentPoolClassifier(nn.Module):
    """Classify residue embeddings using fixed relative sequence segments."""

    def __init__(
        self,
        segments: int,
        embedding_dim: int = 1024,
        include_segment_std: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if min(segments, embedding_dim, hidden_dim) <= 0:
            raise ValueError("segment dimensions must be positive")
        self.segments = segments
        self.include_segment_std = include_segment_std
        statistics = segments * (2 if include_segment_std else 1) + 2
        self.head = nn.Sequential(
            nn.LayerNorm(statistics * embedding_dim),
            nn.Linear(statistics * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.head(self.encode(values, mask)).squeeze(-1)

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError(
                "values and mask must have shapes [batch, residues, features] and [batch, residues]"
            )
        length = mask.sum(dim=1, keepdim=True).clamp_min(1)
        position = torch.arange(values.shape[1], device=values.device)[None, :]
        parts = []
        for index in range(self.segments):
            lower = (length * index) // self.segments
            upper = (length * (index + 1)) // self.segments
            part_mask = mask & (position >= lower) & (position < upper)
            count = part_mask.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (values * part_mask[:, :, None]).sum(dim=1) / count
            parts.append(mean)
            if self.include_segment_std:
                centered = values - mean[:, None, :]
                variance = (centered.square() * part_mask[:, :, None]).sum(dim=1) / count
                parts.append(variance.sqrt())
        count = length
        global_mean = (values * mask[:, :, None]).sum(dim=1) / count
        global_variance = ((values - global_mean[:, None, :]).square() * mask[:, :, None]).sum(
            dim=1
        ) / count
        parts.extend((global_mean, global_variance.sqrt()))
        return torch.cat(parts, dim=1)


class ResidueEmbeddingCNN(nn.Module):
    """Compact masked 1D CNN over the full frozen residue-embedding matrix."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        channels: int = 128,
        depth: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if min(embedding_dim, channels, depth, kernel_size) <= 0 or kernel_size % 2 == 0:
            raise ValueError("CNN dimensions must be positive and kernel_size must be odd")
        layers: list[nn.Module] = [nn.Conv1d(embedding_dim, channels, 1), nn.GELU()]
        for _ in range(depth):
            layers.extend(
                (
                    nn.Conv1d(
                        channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
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

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.head(self.encode(values, mask)).squeeze(-1)

    def encode(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError(
                "values and mask must have shapes [batch, residues, features] and [batch, residues]"
            )
        expanded = mask[:, None, :]
        encoded = (values * mask[:, :, None]).transpose(1, 2)
        for layer in self.features:
            encoded = layer(encoded) * expanded
        mean = (encoded * expanded).sum(dim=-1) / expanded.sum(dim=-1).clamp_min(1)
        maximum = encoded.masked_fill(~expanded, float("-inf")).amax(dim=-1)
        return torch.cat((mean, maximum), dim=1)
