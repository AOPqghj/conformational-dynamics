"""Torch datasets for provenance-checked protein representations."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from protein_state_router.representations.bundle_io import load_embedding_bundle
from protein_state_router.representations.embeddings import ProteinEmbeddings
from protein_state_router.representations.types import ProteinRepresentation
from torch.utils.data import DataLoader, Dataset


class RepresentationDataset(Dataset[tuple[ProteinRepresentation, int]]):
    def __init__(self, items: list[tuple[ProteinRepresentation, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[ProteinRepresentation, int]:
        return self.items[index]


class EmbeddingDataset(Dataset[tuple[ProteinEmbeddings, int]]):
    """Load normalized bundles only after checking exact catalog provenance."""

    def __init__(self, items: list[tuple[str | Path, int]]):
        self.items = [(Path(path), label) for path, label in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[ProteinEmbeddings, int]:
        path, label = self.items[index]
        return load_embedding_bundle(path), label


class ResidueMatrixDataset(Dataset[tuple[str, int, np.ndarray]]):
    """Load one validated residue embedding matrix at a time."""

    def __init__(self, catalog: pd.DataFrame, paths: pd.Series, width: int = 1024):
        self.catalog = catalog.reset_index(drop=True)
        self.paths = paths.reindex(self.catalog["protein_id"])
        self.width = width
        if self.paths.isna().any():
            raise ValueError("every catalog protein requires one embedding path")

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, index: int) -> tuple[str, int, np.ndarray]:
        row = self.catalog.iloc[index]
        path = Path(str(self.paths.iloc[index]))
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                values = archive["single"].astype(np.float32, copy=False)
        else:
            bundle = load_embedding_bundle(path, modalities=("single",))
            if bundle.protein_id != row.protein_id or bundle.sequence != row.sequence:
                raise ValueError(f"embedding identity mismatch for {row.protein_id}")
            if bundle.single is None:
                raise ValueError(f"single embedding is missing for {row.protein_id}")
            values = bundle.single.values.detach().cpu().numpy().astype(np.float32, copy=False)
        expected = (int(row.sequence_length), self.width)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"invalid residue embedding for {row.protein_id}: {values.shape}")
        return str(row.protein_id), int(row.dataset_label), values


def residue_matrix_loader(
    dataset: Dataset[tuple[str, int, np.ndarray]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build the single-process loader used by matrix-aware experiments."""
    from protein_state_router.data.collate import collate_residue_matrices

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_residue_matrices,
        num_workers=0,
    )
