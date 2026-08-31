"""Strict read-only contracts for interpretability inputs.

The helpers in this module deliberately do not fit models or write derived datasets.
They make identity, provenance, split, representation, and frozen-artifact checks a
required precondition for every interpretability workstream.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.lib import format as npy_format
from safetensors import safe_open

VALID_SPLITS = ("train", "val", "test")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
REQUIRED_CATALOG_COLUMNS = frozenset(
    {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
        "dataset_label",
        "source_dataset",
        "split",
    }
)
FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "dataset_label",
        "single_structure_insufficient",
        "source_dataset",
        "label_class",
        "label_confidence",
        "label_confidence_tier",
        "negative_evidence_tier",
        "evidence_type",
        "source_reference",
        "source_id",
        "provenance_json",
        "source_metadata_json",
        "structure_paths_json",
        "structure_ids_json",
        "split",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEQUENCE = re.compile(r"^[A-Z]+$")


@dataclass(frozen=True, slots=True)
class EmbeddingAudit:
    """Summary of one exact catalog-to-embedding join."""

    count: int
    expected_width: int
    verification: str
    model_id: str | None
    model_revision: str | None
    extraction_config_hash: str | None
    dtype: str | None


@dataclass(frozen=True, slots=True)
class FrozenModelAudit:
    """Summary of a checksum-verified frozen model registry."""

    count: int
    artifact_names: tuple[str, ...]
    pooled_model_count: int


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Serializable input audit record for a future run manifest."""

    catalog_rows: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    embedding: EmbeddingAudit
    frozen_models: FrozenModelAudit

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet table without guessing another format."""
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"table does not exist: {location}")
    if location.suffix == ".csv":
        return pd.read_csv(location)
    if location.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(location)
    raise ValueError(f"table must be CSV or parquet: {location}")


def validate_catalog(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (),
    provenance_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Validate the immutable row identities, labels, splits, and provenance.

    ``group_columns`` should name every available family or sequence-cluster field.
    A group is never allowed to cross a split boundary.
    """
    if frame.empty:
        raise ValueError("catalog cannot be empty")
    required = REQUIRED_CATALOG_COLUMNS | set(provenance_columns)
    if missing := required - set(frame.columns):
        raise ValueError(f"catalog is missing columns: {sorted(missing)}")
    if unknown := set(group_columns) - set(frame.columns):
        raise ValueError(f"requested leakage group columns are missing: {sorted(unknown)}")

    catalog = frame.copy().reset_index(drop=True)
    _validate_identifiers(catalog["protein_id"])
    if catalog["protein_id"].duplicated().any():
        raise ValueError("protein_id values must be unique")

    sequences = catalog["sequence"]
    if (
        sequences.isna().any()
        or not sequences.map(
            lambda value: isinstance(value, str) and bool(_SEQUENCE.fullmatch(value))
        ).all()
    ):
        raise ValueError("sequences must be non-empty uppercase ASCII letters without whitespace")
    lengths = pd.to_numeric(catalog["sequence_length"], errors="coerce")
    actual_lengths = sequences.str.len().astype(np.int64)
    if lengths.isna().any() or not np.array_equal(lengths.to_numpy(), actual_lengths.to_numpy()):
        raise ValueError("sequence_length must equal the exact sequence length")

    sequence_hashes = catalog["sequence_sha256"]
    expected_hashes = sequences.map(_sequence_sha256)
    if (
        sequence_hashes.isna().any()
        or not sequence_hashes.map(
            lambda value: isinstance(value, str) and bool(_SHA256.fullmatch(value))
        ).all()
    ):
        raise ValueError("sequence_sha256 values must be lowercase SHA-256 strings")
    if not sequence_hashes.equals(expected_hashes):
        mismatches = catalog.loc[sequence_hashes.ne(expected_hashes), "protein_id"].tolist()
        raise ValueError(f"sequence hash mismatch for protein IDs: {mismatches[:5]}")
    if sequence_hashes.duplicated().any():
        raise ValueError("exact sequences must not be duplicated within or across splits")

    labels = pd.to_numeric(catalog["dataset_label"], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("dataset_label must be binary")
    splits = catalog["split"]
    if set(splits) != set(VALID_SPLITS):
        raise ValueError("split must contain exactly train, val, and test")
    for split in VALID_SPLITS:
        if labels.loc[splits.eq(split)].nunique() != 2:
            raise ValueError(f"{split} must contain both binary classes")

    _validate_nonempty_strings(catalog, ("source_dataset", *provenance_columns))
    for column in group_columns:
        populated = catalog.dropna(subset=[column])
        populated = populated.loc[populated[column].astype(str).str.len().gt(0)]
        if (populated.groupby(column, dropna=False)["split"].nunique() > 1).any():
            raise ValueError(f"{column} crosses split boundaries")
    return catalog


def validate_split_assignments(catalog: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Require an external split table to reproduce the catalog split exactly."""
    required = {"protein_id", "split"}
    if missing := required - set(assignments.columns):
        raise ValueError(f"split assignments are missing columns: {sorted(missing)}")
    if assignments["protein_id"].duplicated().any():
        raise ValueError("split assignments contain duplicate protein IDs")
    expected = catalog.set_index("protein_id")["split"].sort_index()
    actual = assignments.set_index("protein_id")["split"].sort_index()
    if not actual.index.equals(expected.index):
        missing = expected.index.difference(actual.index).tolist()
        extra = actual.index.difference(expected.index).tolist()
        raise ValueError(f"split assignment ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if not actual.equals(expected):
        changed = expected.index[actual.ne(expected)].tolist()
        raise ValueError(f"split assignments changed for protein IDs: {changed[:5]}")


def apply_reference_split(catalog: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Return a catalog with one complete, externally declared split assignment."""
    required = {"protein_id", "split"}
    missing = required - set(assignments)
    if missing or assignments.protein_id.duplicated().any():
        raise ValueError(
            "reference split assignments must have unique protein_id and split columns"
        )
    result = catalog.drop(columns="split", errors="ignore").merge(
        assignments.loc[:, ["protein_id", "split"]], on="protein_id", validate="one_to_one"
    )
    if len(result) != len(catalog) or set(result.split) != set(VALID_SPLITS):
        raise ValueError("reference split assignments must exactly cover the catalog")
    return validate_catalog(result)


def validate_feature_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Reject label, split, evidence, and source fields from a feature view."""
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("feature columns must be non-empty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("feature columns must be unique")
    if forbidden := set(columns) & FORBIDDEN_FEATURE_COLUMNS:
        raise ValueError(f"feature view contains forbidden columns: {sorted(forbidden)}")


def validate_embedding_manifest(
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    root: str | Path = ".",
    expected_width: int = 1024,
    verify: Literal["none", "metadata", "full"] = "metadata",
) -> EmbeddingAudit:
    """Validate an exact one-to-one NPZ manifest and optional tensor contents.

    Metadata verification reads NPZ headers and the small metadata member without
    materializing the residue matrix.
    Full verification additionally loads every matrix and checks finite values.
    """
    if expected_width < 1:
        raise ValueError("expected_width must be positive")
    if verify not in {"none", "metadata", "full"}:
        raise ValueError("verify must be none, metadata, or full")
    required = {"protein_id", "embedding_path"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"embedding manifest is missing columns: {sorted(missing)}")
    if manifest.empty or manifest["protein_id"].duplicated().any():
        raise ValueError("embedding manifest must contain one row per protein ID")
    if manifest["embedding_path"].isna().any() or manifest["embedding_path"].duplicated().any():
        raise ValueError("embedding paths must be non-null and unique")

    expected = catalog.set_index("protein_id", verify_integrity=True)
    actual = manifest.set_index("protein_id", verify_integrity=True)
    if set(actual.index) != set(expected.index):
        missing = sorted(set(expected.index) - set(actual.index))
        extra = sorted(set(actual.index) - set(expected.index))
        raise ValueError(
            f"embedding manifest ID mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    if "embedding_path" in expected:
        aligned = actual.loc[expected.index, "embedding_path"].astype(str)
        catalog_paths = expected["embedding_path"].astype(str)
        if not aligned.equals(catalog_paths):
            changed = aligned.index[aligned.ne(catalog_paths)].tolist()
            raise ValueError(f"embedding paths differ from catalog for: {changed[:5]}")

    base = Path(root)
    signatures: set[tuple[str, str, str, str]] = set()
    for protein_id, row in expected.iterrows():
        path = _resolve_path(actual.at[protein_id, "embedding_path"], base)
        if not path.exists():
            raise FileNotFoundError(f"embedding does not exist for {protein_id}: {path}")
        if not path.is_dir() and path.suffix != ".npz":
            raise ValueError(f"embedding must be an NPZ file or normalized bundle: {path}")
        if verify == "none":
            continue
        metadata, shape, dtype = _read_embedding_contract(path)
        _validate_embedding_metadata(
            metadata,
            protein_id=str(protein_id),
            sequence=str(row["sequence"]),
            sequence_sha256=str(row["sequence_sha256"]),
            expected_shape=(int(row["sequence_length"]), expected_width),
            actual_shape=shape,
            actual_dtype=dtype,
        )
        config = metadata["extraction_config"]
        signature = (
            str(metadata["model_id"]),
            str(metadata["model_revision"]),
            _json_sha256(config),
            dtype,
        )
        signatures.add(signature)
        if verify == "full":
            values = _load_embedding_values(path)
            if not np.isfinite(values).all():
                raise ValueError(f"embedding contains non-finite values for {protein_id}")
    if len(signatures) > 1:
        raise ValueError("embedding manifest mixes model revisions, extraction configs, or dtypes")
    signature = next(iter(signatures), None)
    return EmbeddingAudit(
        count=len(actual),
        expected_width=expected_width,
        verification=verify,
        model_id=signature[0] if signature else None,
        model_revision=signature[1] if signature else None,
        extraction_config_hash=signature[2] if signature else None,
        dtype=signature[3] if signature else None,
    )


def load_residue_matrix(
    path: str | Path,
    *,
    protein_id: str,
    sequence: str,
    sequence_sha256: str,
    sequence_length: int,
    expected_width: int | None = None,
) -> np.ndarray:
    """Load one matrix only after checking its identity and provenance metadata."""
    location = Path(path)
    metadata, shape, dtype = _read_embedding_contract(location)
    width = int(shape[1]) if expected_width is None and len(shape) == 2 else expected_width
    if width is None:
        raise ValueError(f"cannot infer embedding width for {protein_id}")
    _validate_embedding_metadata(
        metadata,
        protein_id=protein_id,
        sequence=sequence,
        sequence_sha256=sequence_sha256,
        expected_shape=(sequence_length, width),
        actual_shape=shape,
        actual_dtype=dtype,
    )
    values = _load_embedding_values(location)
    if not np.isfinite(values).all():
        raise ValueError(f"embedding contains non-finite values for {protein_id}")
    return values


def read_embedding_contract(path: str | Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    """Return validated lightweight provenance without materializing the residue matrix."""
    return _read_embedding_contract(Path(path))


def pool_residue_matrix(values: np.ndarray) -> np.ndarray:
    """Reproduce the frozen classifier's ordered mean, std, and max pooling."""
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("residue matrix must have shape [positive L, positive D]")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError("residue matrix must contain finite floating-point values")
    return np.concatenate((matrix.mean(0), matrix.std(0), matrix.max(0))).astype(np.float32)


def validate_frozen_models(
    models_root: str | Path,
    registry_path: str | Path,
    *,
    expected_embedding_width: int = 1024,
) -> FrozenModelAudit:
    """Verify every registered artifact checksum and adjacent feature manifest.

    This function never deserializes joblib files because joblib is pickle-based.
    Experiment code should deserialize only after this check and only from this trusted registry.
    """
    root = Path(models_root)
    registry_location = Path(registry_path)
    try:
        registry = json.loads(registry_location.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen model registry: {error}") from error
    if not isinstance(registry, dict) or not registry:
        raise ValueError("frozen model registry must be a non-empty object")
    names: list[str] = []
    pooled = 0
    for relative, expected_hash in sorted(registry.items()):
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("registry keys and hashes must be strings")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe or unsupported registry path: {relative}")
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"invalid artifact SHA-256 for {relative}")
        artifact = root.joinpath(*pure.parts)
        if not artifact.is_file() or _sha256_file(artifact) != expected_hash:
            raise ValueError(f"frozen artifact checksum mismatch: {relative}")
        # The trainer freezes checksums for auxiliary provenance and exported
        # parameter artifacts alongside model.joblib.  Verify those files, but
        # apply feature-view contracts only to actual model artifacts.
        if pure.name != "model.joblib":
            continue
        manifest_path = artifact.parent / "manifest.json"
        try:
            model_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"missing or invalid model manifest for {relative}: {error}"
            ) from error
        features = model_manifest.get("features")
        if not isinstance(features, list):
            raise ValueError(f"model manifest has no feature list: {relative}")
        validate_feature_columns(features)
        forbidden = model_manifest.get("forbidden_columns")
        if (
            not isinstance(forbidden, list)
            or any(not isinstance(column, str) or not column for column in forbidden)
            or len(forbidden) != len(set(forbidden))
        ):
            raise ValueError(f"invalid forbidden_columns in model manifest: {relative}")
        if set(features) & set(forbidden):
            raise ValueError(f"model feature list intersects forbidden columns: {relative}")
        counts = model_manifest.get("split_counts")
        if not isinstance(counts, dict) or set(counts) != set(VALID_SPLITS):
            raise ValueError(f"model manifest must record fixed split counts: {relative}")
        if any(not isinstance(counts[split], int) or counts[split] < 1 for split in VALID_SPLITS):
            raise ValueError(f"model manifest contains invalid split counts: {relative}")
        model_name = pure.parts[-2]
        pooled_match = re.fullmatch(r"([a-z0-9_]+)_single_(linear|tree|embedding_cnn)", model_name)
        if pooled_match:
            representation_name = pooled_match.group(1)
            expected_features = (
                [f"embedding_{index}" for index in range(expected_embedding_width * 3)]
                if pooled_match.group(2) == "embedding_cnn"
                else _pooled_feature_names(expected_embedding_width, representation_name)
            )
            if features != expected_features:
                raise ValueError(f"pooled feature contract mismatch: {relative}")
            pooled += 1
        elif model_name.startswith("sequence_plus_esmfold_"):
            expected_features = [
                *_sequence_feature_names(),
                *_pooled_feature_names(expected_embedding_width),
            ]
            if features != expected_features:
                raise ValueError(f"combined feature contract mismatch: {relative}")
            pooled += 1
        elif model_name == "sequence_cnn":
            if features != [*AMINO_ACIDS, "X"]:
                raise ValueError(f"sequence CNN feature contract mismatch: {relative}")
        elif model_name.startswith("sequence_"):
            if features != _sequence_feature_names():
                raise ValueError(f"sequence feature contract mismatch: {relative}")
        else:
            raise ValueError(f"unrecognized frozen model feature view: {relative}")
        names.append(model_name)
    return FrozenModelAudit(len(names), tuple(names), pooled)


def audit_inputs(
    catalog_path: str | Path,
    embedding_manifest_path: str | Path,
    models_root: str | Path,
    registry_path: str | Path,
    *,
    repository_root: str | Path = ".",
    expected_width: int = 1024,
    verify_embeddings: Literal["none", "metadata", "full"] = "metadata",
    group_columns: tuple[str, ...] = (),
    reference_split_path: str | Path | None = None,
) -> AuditSummary:
    """Run the complete read-only preflight used by future experiment commands."""
    catalog = read_table(catalog_path)
    if reference_split_path is not None:
        catalog = apply_reference_split(catalog, read_table(reference_split_path))
    else:
        catalog = validate_catalog(catalog, group_columns=group_columns)
    embedding = validate_embedding_manifest(
        read_table(embedding_manifest_path),
        catalog,
        root=repository_root,
        expected_width=expected_width,
        verify=verify_embeddings,
    )
    frozen = validate_frozen_models(
        models_root, registry_path, expected_embedding_width=expected_width
    )
    split_counts = {
        str(key): int(value) for key, value in catalog["split"].value_counts().sort_index().items()
    }
    label_counts = {
        str(int(key)): int(value)
        for key, value in catalog["dataset_label"].value_counts().sort_index().items()
    }
    return AuditSummary(len(catalog), split_counts, label_counts, embedding, frozen)


def _validate_identifiers(values: pd.Series) -> None:
    if (
        values.isna().any()
        or not values.map(
            lambda value: (
                isinstance(value, str)
                and value == value.strip()
                and bool(value)
                and not any(ord(character) < 32 for character in value)
            )
        ).all()
    ):
        raise ValueError("protein_id values must be non-empty, trimmed strings without controls")


def _validate_nonempty_strings(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if (
            frame[column].isna().any()
            or not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"{column} must contain non-empty provenance strings")


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("embedding paths must be non-empty strings")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_embedding_contract(path: Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    return _read_bundle_contract(path) if path.is_dir() else _read_npz_contract(path)


def _read_npz_contract(path: Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            if not {"single.npy", "metadata.npy"}.issubset(members):
                raise ValueError("NPZ must contain exactly addressable single and metadata arrays")
            with archive.open("single.npy") as handle:
                version = npy_format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = npy_format.read_array_header_1_0(handle)
                elif version in {(2, 0), (3, 0)}:
                    shape, _, dtype = npy_format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"unsupported NPY header version: {version}")
        with np.load(path, allow_pickle=False) as data:
            raw_metadata = data["metadata"]
            if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
                raise ValueError("NPZ metadata must be one scalar JSON string")
            metadata = json.loads(raw_metadata.item())
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid embedding NPZ {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"embedding metadata must be a JSON object: {path}")
    return metadata, tuple(int(value) for value in shape), str(dtype)


def _read_bundle_contract(path: Path) -> tuple[dict[str, Any], tuple[int, ...], str]:
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid embedding bundle manifest {manifest_path}: {error}") from error
    if (
        manifest.get("schema_version") != "protein_state_router_embedding_bundle_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"unsupported or incomplete embedding bundle: {path}")
    if "single" not in manifest.get("actual_modalities", ()):
        raise ValueError(f"embedding bundle has no single representation: {path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or "tensors.safetensors" not in files:
        raise ValueError(f"embedding bundle has no tensor checksum: {path}")
    for relative, expected_hash in files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not _SHA256.fullmatch(str(expected_hash)):
            raise ValueError(f"unsafe bundle artifact entry: {relative}")
        artifact = path.joinpath(*pure.parts)
        if not artifact.is_file() or _sha256_file(artifact) != expected_hash:
            raise ValueError(f"embedding bundle checksum mismatch: {artifact}")
    request = manifest.get("request")
    metadata = manifest.get("metadata")
    if not isinstance(request, dict) or not isinstance(metadata, dict):
        raise ValueError(f"embedding bundle request and metadata must be objects: {path}")
    combined = dict(metadata)
    combined.setdefault("protein_id", request.get("protein_id"))
    combined.setdefault("sequence_sha256", request.get("sequence_sha256"))
    combined.setdefault("model_sequence_sha256", request.get("sequence_sha256"))
    combined.setdefault("model_id", request.get("model_name"))
    combined.setdefault("model_revision", request.get("backend_version"))
    combined.setdefault("extraction_config", request.get("extraction_config"))
    shape = tuple(manifest.get("tensor_shapes", {}).get("single", ()))
    combined.setdefault("shape", list(shape))
    combined.setdefault("dtype", "float32")
    tensor_path = path / "tensors.safetensors"
    try:
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if "single" not in handle.keys():
                raise ValueError(f"embedding bundle tensor file has no single array: {path}")
            single = handle.get_slice("single")
            tensor_shape = tuple(single.get_shape())
            tensor_dtype = single.get_dtype()
    except OSError as error:
        raise ValueError(f"invalid safetensors bundle {path}: {error}") from error
    if tensor_shape != shape:
        raise ValueError(f"bundle manifest and tensor shapes differ: {path}")
    if tensor_dtype != "F32":
        raise ValueError(f"bundle single tensor must be float32: {path}")
    return combined, tuple(int(value) for value in shape), "float32"


def _load_embedding_values(path: Path) -> np.ndarray:
    if path.is_dir():
        from protein_state_router.representations.bundle_io import load_embedding_bundle

        bundle = load_embedding_bundle(path, modalities=("single",))
        if bundle.single is None:
            raise ValueError(f"embedding bundle has no single representation: {path}")
        return bundle.single.values.detach().cpu().numpy().astype(np.float32, copy=False)
    with np.load(path, allow_pickle=False) as archive:
        return archive["single"].astype(np.float32, copy=False)


def _validate_embedding_metadata(
    metadata: dict[str, Any],
    *,
    protein_id: str,
    sequence: str,
    sequence_sha256: str,
    expected_shape: tuple[int, int],
    actual_shape: tuple[int, ...],
    actual_dtype: str,
) -> None:
    required = {
        "protein_id",
        "sequence_sha256",
        "model_id",
        "model_revision",
        "extraction_config",
    }
    if missing := required - set(metadata):
        raise ValueError(
            f"embedding metadata is missing fields for {protein_id}: {sorted(missing)}"
        )
    if metadata["protein_id"] != protein_id:
        raise ValueError(f"embedding protein ID mismatch for {protein_id}")
    if metadata["sequence_sha256"] != sequence_sha256:
        raise ValueError(f"embedding sequence hash mismatch for {protein_id}")
    declared_shape = tuple(metadata.get("shape", actual_shape))
    if actual_shape != expected_shape or declared_shape != expected_shape:
        raise ValueError(
            f"embedding shape mismatch for {protein_id}: expected {expected_shape}, got {actual_shape}"
        )
    if actual_dtype != "float32" or metadata.get("dtype", actual_dtype) != "float32":
        raise ValueError(f"embedding dtype must be float32 for {protein_id}")
    if not isinstance(metadata["model_id"], str) or not metadata["model_id"]:
        raise ValueError(f"embedding model_id is missing for {protein_id}")
    if not isinstance(metadata["model_revision"], str) or not metadata["model_revision"]:
        raise ValueError(f"embedding model_revision is missing for {protein_id}")
    config = metadata["extraction_config"]
    if not isinstance(config, dict) or not config:
        raise ValueError(f"embedding extraction_config is missing for {protein_id}")
    representation = config.get("representation")
    if representation not in {
        "folding_trunk_s_s",
        "alphafold2_evoformer_single",
        "esm2_final_hidden_state",
    }:
        raise ValueError(f"unsupported residue representation for {protein_id}: {representation}")
    if config.get("output_dtype") != "float32":
        raise ValueError(f"unexpected extraction dtype for {protein_id}")
    policy = config.get("nonstandard_residue_policy", "none")
    if policy == "U_to_X":
        model_sequence = sequence.replace("U", "X")
    elif policy in {None, "none"}:
        model_sequence = sequence
    else:
        raise ValueError(f"unsupported nonstandard-residue policy for {protein_id}: {policy}")
    model_sequence_hash = metadata.get("model_sequence_sha256", sequence_sha256)
    if model_sequence_hash != _sequence_sha256(model_sequence):
        raise ValueError(f"model input sequence hash mismatch for {protein_id}")


def _pooled_feature_names(width: int, representation_name: str = "esmfold") -> list[str]:
    return [
        f"{representation_name}_single_{stat}_{index}"
        for stat in ("mean", "std", "max")
        for index in range(width)
    ]


def _sequence_feature_names() -> list[str]:
    return [
        "log1p_sequence_length",
        *(f"fraction_{residue}" for residue in AMINO_ACIDS),
        "entropy",
    ]
