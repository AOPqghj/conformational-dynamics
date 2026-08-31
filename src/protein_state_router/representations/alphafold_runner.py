"""AlphaFold 3 runner and documented embedding NPZ parser.

This module deliberately does not import AlphaFold. AF3 installation, licensed
weights, and the Colab command are external prerequisites.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from protein_state_router.representations.bundle_io import EmbeddingRequest
from protein_state_router.representations.embeddings import (
    EmbeddingSource,
    PairEmbedding,
    ProteinEmbeddings,
    SingleEmbedding,
)
from protein_state_router.representations.errors import BackendCapabilityError


def write_alphafold3_input(request: EmbeddingRequest, path: str | Path) -> Path:
    """Write a minimal AF3 monomer input JSON from an immutable request."""
    path = Path(path)
    payload = {
        "name": request.protein_id,
        "modelSeeds": [1],
        "sequences": [{"protein": {"id": "A", "sequence": request.sequence}}],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def parse_alphafold3_embeddings(
    npz_path: str | Path, request: EmbeddingRequest
) -> ProteinEmbeddings:
    """Parse AF3's documented `single_embeddings` and `pair_embeddings` arrays."""
    with np.load(npz_path) as archive:
        single_array = archive.get("single_embeddings")
        pair_array = archive.get("pair_embeddings")
    if single_array is None and pair_array is None:
        raise BackendCapabilityError("AF3 output lacks single_embeddings and pair_embeddings")
    length = len(request.sequence)
    if single_array is not None and single_array.shape != (length, 384):
        raise BackendCapabilityError(
            f"unexpected AF3 single shape {single_array.shape}; expected ({length}, 384)"
        )
    if pair_array is not None and pair_array.shape != (length, length, 128):
        raise BackendCapabilityError(
            f"unexpected AF3 pair shape {pair_array.shape}; expected ({length}, {length}, 128)"
        )
    source = EmbeddingSource(
        "alphafold3",
        request.backend_version,
        request.model_name,
        request.extraction_config_hash,
        request.sequence_sha256,
        "final_pairformer",
    )
    mask = torch.ones(length, dtype=torch.bool)
    single = (
        SingleEmbedding(torch.from_numpy(single_array), mask, source)
        if single_array is not None
        else None
    )
    pair = (
        PairEmbedding(
            torch.from_numpy(pair_array), torch.ones(length, length, dtype=torch.bool), source
        )
        if pair_array is not None
        else None
    )
    return ProteinEmbeddings(
        request.protein_id,
        request.sequence,
        request.sequence_sha256,
        source,
        single,
        pair,
        metadata={"raw_embedding_file": str(npz_path)},
    )


def run_alphafold_embedding_job(
    request: EmbeddingRequest, input_path: str | Path, output_dir: str | Path, command: list[str]
) -> Path:
    """Invoke a configured AF3 command with saving enabled; return its output directory."""
    if not command:
        raise BackendCapabilityError(
            "AlphaFold command is required; install AF3 and authorized weights in Colab/local runtime"
        )
    write_alphafold3_input(request, input_path)
    invocation = [
        *command,
        f"--json_path={input_path}",
        f"--output_dir={output_dir}",
        "--save_embeddings=true",
    ]
    try:
        subprocess.run(invocation, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BackendCapabilityError(f"AlphaFold inference failed: {error}") from error
    return Path(output_dir)
