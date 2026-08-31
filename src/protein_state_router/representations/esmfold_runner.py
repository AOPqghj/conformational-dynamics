"""Frozen ESMFold v1 folding-trunk embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from protein_state_router.representations.embeddings import (
    EmbeddingSource,
    ProteinEmbeddings,
    SingleEmbedding,
)
from protein_state_router.representations.errors import BackendCapabilityError
from protein_state_router.representations.query import sequence_sha256

ESMFOLD_MODEL_ID = "facebook/esmfold_v1"
ESMFOLD_MODEL_REVISION = "75a3841ee059df2bf4d56688166c8fb459ddd97a"
ESMFOLD_OUTPUT_WIDTH = 1024
ESMFOLD_EXTRACTION_CONFIG: dict[str, object] = {
    "representation": "folding_trunk_s_s",
    "num_recycles": 0,
    "chunk_size": 128,
    "max_sequence_length": 1022,
    "nonstandard_residue_policy": "U_to_X",
    "output_dtype": "float32",
}


def _configuration_hash(configuration: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]


def normalize_esmfold_sequence(sequence: str) -> str:
    """Map selenocysteine to ESMFold's explicit unknown-residue token."""
    return sequence.upper().replace("U", "X")


@dataclass(frozen=True, slots=True)
class ESMFoldConfig:
    """Pinned settings used by both extraction and bundle provenance."""

    model_id: str = ESMFOLD_MODEL_ID
    model_revision: str = ESMFOLD_MODEL_REVISION
    num_recycles: int = 0
    chunk_size: int = 128
    max_sequence_length: int = 1022

    def extraction_config(self) -> dict[str, object]:
        return {
            **ESMFOLD_EXTRACTION_CONFIG,
            "num_recycles": self.num_recycles,
            "chunk_size": self.chunk_size,
            "max_sequence_length": self.max_sequence_length,
        }


class ESMFoldTrunkExtractor:
    """Extract final per-residue folding-trunk states from ESMFold v1."""

    backend = "esmfold_v1"

    def __init__(
        self,
        config: ESMFoldConfig | None = None,
        *,
        device: str | torch.device | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config or ESMFoldConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be supplied together")
        if model is None:
            try:
                from transformers import AutoTokenizer, EsmForProteinFolding
            except ImportError as error:  # pragma: no cover - optional dependency
                raise BackendCapabilityError(
                    "ESMFold requires the 'esmfold' extra; run `uv sync --extra esmfold`."
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, revision=self.config.model_revision
            )
            model = EsmForProteinFolding.from_pretrained(
                self.config.model_id,
                revision=self.config.model_revision,
                low_cpu_mem_usage=True,
            )
            if self.device.type == "cuda":
                model.esm = model.esm.half()
        self.model = model.to(self.device).eval()
        if tokenizer is None:
            raise BackendCapabilityError("ESMFold tokenizer initialization failed")
        self.tokenizer = tokenizer
        self.model.trunk.set_chunk_size(self.config.chunk_size)

    def extract(self, protein_id: str, sequence: str) -> ProteinEmbeddings:
        """Return one finite float32 `[L,1024]` trunk representation."""
        original = sequence.upper()
        if not original or any(character.isspace() for character in original):
            raise ValueError("sequence must be non-empty and whitespace-free")
        if len(original) > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {len(original)} exceeds ESMFold limit "
                f"{self.config.max_sequence_length}; refusing silent truncation"
            )
        model_sequence = normalize_esmfold_sequence(original)
        tokens = self.tokenizer(model_sequence, return_tensors="pt", add_special_tokens=False)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        token_count = int(tokens["attention_mask"].sum().item())
        if token_count != len(original):
            raise BackendCapabilityError(
                f"ESMFold tokenization changed sequence length for {protein_id}: "
                f"expected {len(original)}, got {token_count}"
            )
        with torch.inference_mode():
            output = self.model(**tokens, num_recycles=self.config.num_recycles)
        raw = output.s_s if hasattr(output, "s_s") else output["s_s"]
        values = raw[0].detach().float().cpu().contiguous()
        expected = (len(original), ESMFOLD_OUTPUT_WIDTH)
        if tuple(values.shape) != expected or not torch.isfinite(values).all():
            raise BackendCapabilityError(
                f"invalid ESMFold trunk tensor for {protein_id}: {tuple(values.shape)}"
            )
        original_hash = sequence_sha256(original)
        extraction_config = self.config.extraction_config()
        source = EmbeddingSource(
            self.backend,
            self.config.model_revision,
            self.config.model_id,
            _configuration_hash(extraction_config),
            original_hash,
            "folding_trunk_s_s",
            self.config.num_recycles,
        )
        return ProteinEmbeddings(
            protein_id,
            original,
            original_hash,
            source,
            SingleEmbedding(values, torch.ones(len(original), dtype=torch.bool), source),
            None,
            metadata={
                "model_sequence_sha256": sequence_sha256(model_sequence),
                "normalization": "U_to_X" if model_sequence != original else "none",
                "token_count": token_count,
                "extraction_config": extraction_config,
            },
        )
