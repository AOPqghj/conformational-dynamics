"""Per-protein safetensors cache with JSON metadata sidecars."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from protein_state_router.representations.types import ProteinRepresentation
from safetensors.torch import load_file, save_file


def cache_representation(rep: ProteinRepresentation, root: str | Path) -> Path:
    directory = Path(root) / rep.protein_id
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {"residue_mask": rep.residue_mask}
    for name in ("single", "pair", "pair_mask", "confidence_features"):
        value = getattr(rep, name)
        if value is not None:
            tensors[name] = value.contiguous()
    save_file(tensors, directory / "tensors.safetensors")
    metadata = {
        **rep.metadata,
        "protein_id": rep.protein_id,
        "sequence_length": int(rep.residue_mask.numel()),
        "single_shape": list(rep.single.shape) if rep.single is not None else None,
        "pair_shape": list(rep.pair.shape) if rep.pair is not None else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return directory


def load_cached_representation(directory: str | Path) -> ProteinRepresentation:
    directory = Path(directory)
    tensors = load_file(directory / "tensors.safetensors")
    metadata = json.loads((directory / "metadata.json").read_text())
    return ProteinRepresentation(
        metadata["protein_id"],
        tensors.get("single"),
        tensors.get("pair"),
        tensors["residue_mask"].bool(),
        tensors.get("pair_mask", None),
        tensors.get("confidence_features", None),
        metadata,
    )
