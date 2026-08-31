"""Verified adapters for frozen pooled-ESMFold reference models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from interpretability.contracts import (
    load_residue_matrix,
    pool_residue_matrix,
    validate_frozen_models,
)


@dataclass(frozen=True, slots=True)
class ModelScore:
    """One frozen-model score, preserving both direction and probability."""

    margin: float
    probability: float


class FrozenPooledModel:
    """Checksum-verified pooled model with explicit per-protein scoring."""

    def __init__(self, root: Path, name: str):
        registry = root / "frozen_model_registry.json"
        manifest = _read_manifest(root / name)
        features = manifest.get("features")
        if not isinstance(features, list) or len(features) % 3:
            raise ValueError(f"pooled model has an invalid feature contract: {name}")
        self.embedding_width = len(features) // 3
        validate_frozen_models(root, registry, expected_embedding_width=self.embedding_width)
        relative = f"{name}/model.joblib"
        hashes = json.loads(registry.read_text())
        if relative not in hashes:
            raise ValueError(f"model is not registered: {name}")
        path = root / relative
        if _sha256(path) != hashes[relative]:
            raise ValueError(f"model checksum changed after validation: {name}")
        self.name = name
        self.path = path
        self.model = joblib.load(path)

    def score_matrix(self, values: np.ndarray) -> ModelScore:
        features = pool_residue_matrix(values).reshape(1, -1)
        probability = float(self.model.predict_proba(features)[0, 1])
        if hasattr(self.model, "decision_function"):
            margin = float(self.model.decision_function(features)[0])
        else:
            probability = float(np.clip(probability, 1e-7, 1 - 1e-7))
            margin = float(np.log(probability / (1 - probability)))
        if not np.isfinite([margin, probability]).all():
            raise ValueError("frozen model returned a non-finite score")
        return ModelScore(margin, probability)

    def score_protein(self, row: object) -> ModelScore:
        values = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=self.embedding_width,
        )
        return self.score_matrix(values)

    def linear_weights(self) -> dict[str, np.ndarray | float]:
        """Return exported linear parameters, rejecting non-linear model names."""
        path = self.path.with_name("linear_weights.npz")
        if not path.is_file():
            raise ValueError(f"model has no exported linear weights: {self.name}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "feature_names",
                "standardized_coefficients",
                "raw_coefficients",
                "raw_intercept",
            }
            if required - set(archive.files):
                raise ValueError("linear weight artifact is incomplete")
            return {
                "feature_names": archive["feature_names"].copy(),
                "standardized_coefficients": archive["standardized_coefficients"].copy(),
                "raw_coefficients": archive["raw_coefficients"].copy(),
                "raw_intercept": float(archive["raw_intercept"].item()),
            }


class FrozenResidueCNNModel:
    """Checksum-verified CNN that consumes the complete residue matrix."""

    def __init__(self, root: Path, name: str, *, device: str = "cpu"):
        registry_path = root / "frozen_residue_model_registry.json"
        relative = f"{name}/model.pt"
        expected_hash = _registry_hash(registry_path, relative)
        path = root / relative
        if _sha256(path) != expected_hash:
            raise ValueError(f"model checksum changed after validation: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("model") not in {"residue_cnn", "residue_cnn_expanded"}:
            raise ValueError("frozen residue model is not a supported full-matrix CNN")
        manifest = _read_manifest(path.parent)
        if manifest.get("representation") != "full_matrix_cnn_encode":
            raise ValueError("frozen residue model does not declare full-matrix encoding")
        from ml.train_residue_embedding_models import Candidate, _model

        self.name = name
        self.path = path
        self.device = torch.device(device)
        self.embedding_width = int(manifest.get("embedding_width", 1024))
        self.model = _model(
            str(payload["model"]), Candidate(**payload["candidate"]), self.embedding_width
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    def encode_matrix(self, values: np.ndarray) -> np.ndarray:
        matrix = _validate_matrix(values, self.embedding_width)
        tensor = torch.from_numpy(matrix).unsqueeze(0).to(self.device)
        mask = torch.ones((1, matrix.shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            encoded = self.model.encode(tensor, mask).squeeze(0).cpu().numpy()
        if not np.isfinite(encoded).all():
            raise ValueError("frozen CNN returned non-finite features")
        return encoded.astype(np.float32)

    def score_matrix(self, values: np.ndarray) -> ModelScore:
        matrix = _validate_matrix(values, self.embedding_width)
        tensor = torch.from_numpy(matrix).unsqueeze(0).to(self.device)
        mask = torch.ones((1, matrix.shape[0]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logit = float(self.model(tensor, mask).squeeze(0).cpu())
        probability = float(torch.sigmoid(torch.tensor(logit)))
        return _score(logit, probability)

    def score_protein(self, row: object) -> ModelScore:
        values = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=self.embedding_width,
        )
        return self.score_matrix(values)


class FrozenResidueStackedModel:
    """Frozen full-matrix CNN encoder followed by a classical classifier head."""

    def __init__(
        self,
        root: Path,
        name: str,
        extractor_root: Path,
        *,
        device: str = "cpu",
    ):
        registry_path = root / "frozen_residue_stacker_registry.json"
        relative = f"{name}/classifier.joblib"
        expected_hash = _registry_hash(registry_path, relative)
        path = root / relative
        if _sha256(path) != expected_hash:
            raise ValueError(f"stacked classifier checksum changed: {name}")
        manifest = _read_manifest(path.parent)
        if manifest.get("input") != "full residue matrix through frozen CNN encoder":
            raise ValueError("stacked classifier does not declare full-matrix input")
        extractor_name = manifest.get("extractor")
        if not isinstance(extractor_name, str) or not extractor_name.startswith("residue_cnn"):
            raise ValueError("stacked classifier has an invalid CNN extractor")
        self.name = name
        self.path = path
        self.extractor = FrozenResidueCNNModel(extractor_root, extractor_name, device=device)
        self.model = joblib.load(path)
        expected_dimension = int(manifest.get("feature_dimension", -1))
        if expected_dimension <= 0:
            raise ValueError("stacked classifier is missing feature dimension")
        self.feature_dimension = expected_dimension

    def score_matrix(self, values: np.ndarray) -> ModelScore:
        features = self.extractor.encode_matrix(values).reshape(1, -1)
        if features.shape[1] != self.feature_dimension:
            raise ValueError("CNN feature dimension differs from stacked head contract")
        probability = float(self.model.predict_proba(features)[0, 1])
        if hasattr(self.model, "decision_function"):
            margin = float(self.model.decision_function(features)[0])
        else:
            clipped = float(np.clip(probability, 1e-7, 1 - 1e-7))
            margin = float(np.log(clipped / (1 - clipped)))
        return _score(margin, probability)

    def score_protein(self, row: object) -> ModelScore:
        values = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
        )
        return self.score_matrix(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen model manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("frozen model manifest must be an object")
    return value


def _registry_hash(registry_path: Path, relative: str) -> str:
    try:
        registry = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen model registry: {error}") from error
    expected = registry.get(relative) if isinstance(registry, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"model is not registered: {relative}")
    return expected


def _validate_matrix(values: np.ndarray, expected_width: int = 1024) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != expected_width:
        raise ValueError(f"full residue matrix must have shape [positive L, {expected_width}]")
    if not np.isfinite(matrix).all():
        raise ValueError("residue matrix must contain finite values")
    return matrix


def _score(margin: float, probability: float) -> ModelScore:
    if not np.isfinite([margin, probability]).all() or not 0.0 <= probability <= 1.0:
        raise ValueError("frozen model returned an invalid score")
    return ModelScore(float(margin), float(probability))
