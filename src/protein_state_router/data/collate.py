"""Padding-aware batching helpers."""

from __future__ import annotations

import numpy as np
import torch
from protein_state_router.representations.embeddings import ProteinEmbeddings
from protein_state_router.representations.types import ProteinRepresentation
from torch import Tensor


def pad_residue_matrices(matrices: tuple[np.ndarray, ...]) -> tuple[Tensor, Tensor]:
    """Pad finite ``[L, D]`` residue matrices and return their residue mask."""
    if not matrices:
        raise ValueError("cannot pad an empty residue-matrix batch")
    width = matrices[0].shape[1]
    if any(matrix.ndim != 2 or matrix.shape[1] != width for matrix in matrices):
        raise ValueError("residue matrices must share one embedding width")
    longest = max(matrix.shape[0] for matrix in matrices)
    values = torch.zeros((len(matrices), longest, width), dtype=torch.float32)
    mask = torch.zeros((len(matrices), longest), dtype=torch.bool)
    for index, matrix in enumerate(matrices):
        values[index, : matrix.shape[0]] = torch.from_numpy(matrix)
        mask[index, : matrix.shape[0]] = True
    return values, mask


def collate_residue_matrices(
    batch: list[tuple[str, int, np.ndarray]],
) -> tuple[list[str], Tensor, Tensor, Tensor]:
    """Collate the canonical residue-matrix dataset contract."""
    identifiers, labels, matrices = zip(*batch, strict=True)
    values, mask = pad_residue_matrices(matrices)
    return list(identifiers), torch.tensor(labels, dtype=torch.float32), values, mask


def collate_representations(items: list[tuple[ProteinRepresentation, int]]) -> dict[str, object]:
    """Pad variable-length representations and retain explicit availability flags."""
    reps, labels = zip(*items, strict=True)
    max_length = max(rep.residue_mask.shape[0] for rep in reps)
    single_dim = next((rep.single.shape[-1] for rep in reps if rep.single is not None), 0)
    pair_dim = next((rep.pair.shape[-1] for rep in reps if rep.pair is not None), 0)
    single = torch.zeros(len(reps), max_length, single_dim)
    pair = torch.zeros(len(reps), max_length, max_length, pair_dim) if pair_dim else None
    residue_mask = torch.zeros(len(reps), max_length, dtype=torch.bool)
    pair_mask = (
        torch.zeros(len(reps), max_length, max_length, dtype=torch.bool) if pair_dim else None
    )
    single_available, pair_available = [], []
    for index, rep in enumerate(reps):
        length = rep.residue_mask.shape[0]
        residue_mask[index, :length] = rep.residue_mask
        single_available.append(rep.single is not None)
        pair_available.append(rep.pair is not None)
        if rep.single is not None:
            single[index, :length] = rep.single
        if pair is not None and pair_mask is not None and rep.pair is not None:
            pair[index, :length, :length] = rep.pair
            pair_mask[index, :length, :length] = rep.pair_mask
    return {
        "single": single,
        "pair": pair,
        "residue_mask": residue_mask,
        "pair_mask": pair_mask,
        "single_available": torch.tensor(single_available),
        "pair_available": torch.tensor(pair_available),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "protein_ids": [rep.protein_id for rep in reps],
    }


def collate_embeddings(items: list[tuple[ProteinEmbeddings, int]]) -> dict[str, object]:
    """Convert normalized records to the existing batch contract with provenance retained."""
    representations = []
    for embedding, label in items:
        single = embedding.single.values if embedding.single is not None else None
        pair = embedding.pair.values if embedding.pair is not None else None
        pair_mask = embedding.pair.pair_mask if embedding.pair is not None else None
        representations.append(
            (
                ProteinRepresentation(
                    embedding.protein_id,
                    single,
                    pair,
                    embedding.residue_mask,
                    pair_mask,
                    embedding.confidence_features,
                    {
                        "sequence_sha256": embedding.sequence_sha256,
                        "backend": embedding.source.backend,
                        "backend_version": embedding.source.backend_version,
                    },
                ),
                label,
            )
        )
    return collate_representations(representations)
