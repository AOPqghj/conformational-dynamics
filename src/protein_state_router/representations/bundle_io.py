"""Portable normalized embedding bundles, independent from backend raw artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from protein_state_router.representations.embeddings import (
    EmbeddingSource,
    PairEmbedding,
    ProteinEmbeddings,
    SingleEmbedding,
)
from protein_state_router.representations.errors import EmbeddingBundleError
from protein_state_router.representations.query import sequence_sha256
from safetensors import safe_open
from safetensors.torch import save_file

EMBEDDING_BUNDLE_SCHEMA = "protein_state_router_embedding_bundle_v1"


def _path_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in "/\\"):
        raise EmbeddingBundleError(f"invalid {label} path component")
    return value


def build_embedding_output_path(
    root: str | Path,
    backend: str,
    backend_version: str,
    protein_id: str,
    sequence_sha256: str,
    extraction_config_hash: str,
) -> Path:
    """Return a deterministic cache path for one embedding request."""
    return (
        Path(root)
        / _path_component(backend, "backend")
        / _path_component(backend_version, "backend version")
        / _path_component(protein_id, "protein id")
        / _path_component(sequence_sha256, "sequence hash")
        / _path_component(extraction_config_hash, "configuration hash")
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    protein_id: str
    sequence: str
    sequence_sha256: str
    backend: str
    backend_version: str
    model_name: str
    extraction_config: dict[str, Any]
    requested_modalities: tuple[str, ...] = ("single", "pair")

    def __post_init__(self) -> None:
        if self.sequence_sha256 != sequence_sha256(self.sequence):
            raise EmbeddingBundleError("request sequence hash does not match sequence")
        if not set(self.requested_modalities).issubset({"single", "pair"}):
            raise EmbeddingBundleError("modalities must be single and/or pair")

    @property
    def extraction_config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.extraction_config, sort_keys=True).encode()
        ).hexdigest()[:16]


def embedding_request_dict(request: EmbeddingRequest) -> dict[str, Any]:
    """Return the stable JSON representation used in every manifest."""
    value = asdict(request)
    value["requested_modalities"] = list(request.requested_modalities)
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingBundleManifest:
    schema_version: str
    request: dict[str, Any]
    actual_modalities: tuple[str, ...]
    tensor_shapes: dict[str, list[int]]
    tensor_dtype: str
    files: dict[str, str]
    status: str = "complete"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_embedding_bundle(
    embeddings: ProteinEmbeddings,
    request: EmbeddingRequest,
    directory: str | Path,
    raw_artifact_paths: list[str | Path] | None = None,
) -> Path:
    """Write normalized tensors/manifest and copy no backend parsing logic downstream."""
    if (
        embeddings.sequence_sha256 != request.sequence_sha256
        or embeddings.sequence != request.sequence
    ):
        raise EmbeddingBundleError("embedding sequence does not match request")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {"residue_mask": embeddings.residue_mask.bool()}
    shapes: dict[str, list[int]] = {}
    modalities = []
    if embeddings.single is not None:
        tensors["single"] = embeddings.single.values.contiguous()
        shapes["single"] = list(embeddings.single.values.shape)
        modalities.append("single")
    if embeddings.pair is not None:
        tensors["pair"] = embeddings.pair.values.contiguous()
        tensors["pair_mask"] = embeddings.pair.pair_mask.bool().contiguous()
        shapes["pair"] = list(embeddings.pair.values.shape)
        modalities.append("pair")
    if embeddings.confidence_features is not None:
        tensors["confidence_features"] = embeddings.confidence_features.contiguous()
    tensor_path = destination / "tensors.safetensors"
    save_file(tensors, tensor_path)
    files = {"tensors.safetensors": sha256_file(tensor_path)}
    raw_directory = destination / "raw"
    for raw in raw_artifact_paths or []:
        raw_path = Path(raw)
        if raw_path.is_file():
            raw_directory.mkdir(exist_ok=True)
            copied_path = raw_directory / raw_path.name
            copied_path.write_bytes(raw_path.read_bytes())
            files[f"raw/{raw_path.name}"] = sha256_file(copied_path)
    manifest = EmbeddingBundleManifest(
        EMBEDDING_BUNDLE_SCHEMA,
        embedding_request_dict(request),
        tuple(modalities),
        shapes,
        str(next(iter(tensors.values())).dtype),
        files,
        metadata=embeddings.metadata,
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )
    return destination


def load_embedding_bundle(
    directory: str | Path, *, modalities: tuple[str, ...] | None = None
) -> ProteinEmbeddings:
    """Validate a normalized bundle and return selected backend-neutral modalities.

    ``modalities=("single",)`` avoids materializing a potentially very large
    quadratic pair tensor for pooled-single experiments.  The bundle checksum is
    still validated in full before any tensor is read.
    """
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingBundleError(f"missing or invalid manifest: {error}") from error
    if (
        manifest.get("schema_version") != EMBEDDING_BUNDLE_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise EmbeddingBundleError("unsupported or incomplete embedding bundle")
    tensor_path = directory / "tensors.safetensors"
    if manifest.get("files", {}).get("tensors.safetensors") != sha256_file(tensor_path):
        raise EmbeddingBundleError("tensor checksum mismatch")
    requested = (
        set(modalities)
        if modalities is not None
        else set(manifest.get("actual_modalities", ("single", "pair")))
    )
    if not requested or not requested.issubset({"single", "pair"}):
        raise EmbeddingBundleError("modalities must be a non-empty subset of single and pair")
    for relative_path, checksum in manifest.get("files", {}).items():
        artifact = directory / relative_path
        if not artifact.is_file() or sha256_file(artifact) != checksum:
            raise EmbeddingBundleError(f"artifact checksum mismatch: {relative_path}")
    request = manifest["request"]
    source = EmbeddingSource(
        request["backend"],
        request["backend_version"],
        request["model_name"],
        hashlib.sha256(
            json.dumps(request["extraction_config"], sort_keys=True).encode()
        ).hexdigest()[:16],
        request["sequence_sha256"],
    )
    tensor_names = {"residue_mask", *requested}
    if "pair" in requested:
        tensor_names.add("pair_mask")
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = tensor_names - available
        if missing:
            raise EmbeddingBundleError(f"bundle is missing requested tensors: {sorted(missing)}")
        # Read only the requested representation tensors.  In particular, a
        # single-only training run does not materialize the quadratic pair array.
        tensors = {name: handle.get_tensor(name) for name in tensor_names}
        if "confidence_features" in available:
            tensors["confidence_features"] = handle.get_tensor("confidence_features")
    single = (
        SingleEmbedding(tensors["single"], tensors["residue_mask"].bool(), source)
        if "single" in requested
        else None
    )
    pair = (
        PairEmbedding(tensors["pair"], tensors["pair_mask"].bool(), source)
        if "pair" in requested
        else None
    )
    return ProteinEmbeddings(
        request["protein_id"],
        request["sequence"],
        request["sequence_sha256"],
        source,
        single,
        pair,
        tensors.get("confidence_features"),
        manifest.get("metadata", {}),
    )


def validate_single_embedding_bundle(
    directory: str | Path,
    request: EmbeddingRequest,
    *,
    width: int = 384,
) -> ProteinEmbeddings:
    """Strictly validate one normalized single-representation bundle."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_request = embedding_request_dict(request)
    expected_shape = [len(request.sequence), width]
    if tuple(request.requested_modalities) != ("single",):
        raise EmbeddingBundleError("embedding request must ask for exactly the single modality")
    if manifest.get("request") != expected_request:
        raise EmbeddingBundleError("bundle request does not match the expected protein")
    if manifest.get("actual_modalities") != ["single"]:
        raise EmbeddingBundleError("bundle must contain exactly the single modality")
    if manifest.get("tensor_shapes") != {"single": expected_shape}:
        raise EmbeddingBundleError(f"single tensor shape must be {expected_shape}")

    with safe_open(directory / "tensors.safetensors", framework="pt", device="cpu") as handle:
        single_tensor = handle.get_tensor("single")
        residue_mask = handle.get_tensor("residue_mask")
    if single_tensor.dtype != torch.float32 or not torch.isfinite(single_tensor).all():
        raise EmbeddingBundleError("single tensor must contain finite float32 values")
    if (
        residue_mask.dtype != torch.bool
        or tuple(residue_mask.shape) != (len(request.sequence),)
        or not residue_mask.all()
    ):
        raise EmbeddingBundleError("residue mask must be an all-true bool vector")

    embeddings = load_embedding_bundle(directory, modalities=("single",))
    if embeddings.single is None or tuple(embeddings.single.values.shape) != tuple(expected_shape):
        raise EmbeddingBundleError(f"single tensor shape must be {expected_shape}")
    if tuple(embeddings.residue_mask.shape) != (len(request.sequence),):
        raise EmbeddingBundleError("residue mask length does not match the sequence")
    source = embeddings.source
    if (
        embeddings.protein_id != request.protein_id
        or embeddings.sequence != request.sequence
        or embeddings.sequence_sha256 != request.sequence_sha256
        or source.backend != request.backend
        or source.backend_version != request.backend_version
        or source.model_name != request.model_name
        or source.extraction_config_hash != request.extraction_config_hash
        or source.sequence_sha256 != request.sequence_sha256
    ):
        raise EmbeddingBundleError("bundle provenance does not match the request")
    return embeddings
