"""Dataset-agnostic binary baselines over pooled frozen protein embeddings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
from protein_state_router.models.probes import ActivationName, FeatureMLP
from protein_state_router.pooling.pooling import pool_pair, pool_single
from protein_state_router.representations.embeddings import ProteinEmbeddings
from protein_state_router.training.checkpoints import load_checkpoint, save_checkpoint
from protein_state_router.training.trainer import TrainResult, resolve_device, train_feature_mlp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelKind = Literal["lasso", "ridge", "mlp"]


@dataclass(frozen=True, slots=True)
class ModelExample:
    """One protein's frozen embeddings, optional binary label, and numeric covariates."""

    protein_id: str
    embeddings: ProteinEmbeddings
    label: int | None = None
    metadata: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.label not in (None, 0, 1):
            raise ValueError("label must be 0, 1, or None")
        if self.protein_id != self.embeddings.protein_id:
            raise ValueError("example protein_id must match its embeddings")
        if self.metadata is not None and not all(
            isinstance(value, (int, float)) for value in self.metadata.values()
        ):
            raise ValueError("metadata values must be numeric")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration shared by L1/L2 logistic and Torch MLP baseline models."""

    kind: ModelKind
    regularization_strength: float = 1.0  # Larger values apply stronger L1/L2 shrinkage.
    class_weight: str | dict[int, float] | None = "balanced"
    hidden_dim: int = 64
    dropout: float = 0.15
    activation: ActivationName = "gelu"
    device: str = "cpu"
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.kind not in {"lasso", "ridge", "mlp"}:
            raise ValueError(f"unsupported model kind {self.kind!r}; supported: lasso, ridge, mlp")
        if self.regularization_strength <= 0:
            raise ValueError("regularization_strength must be positive")
        if self.hidden_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("hidden_dim must be positive and dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu", "silu"}:
            raise ValueError("activation must be relu, gelu, or silu")
        resolve_device(self.device)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Fixed pooled-vector layout learned from the training examples."""

    single_dim: int
    pair_dim: int
    confidence_dim: int
    metadata_keys: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return (
            self.single_dim * 3
            + self.pair_dim * 12
            + self.confidence_dim
            + 2
            + len(self.metadata_keys)
        )


class EmbeddingFeatureBuilder:
    """Turn variable-length embedding records into schema-checked fixed vectors."""

    def __init__(self, schema: FeatureSchema | None = None):
        self.schema = schema

    def fit(self, examples: Sequence[ModelExample]) -> EmbeddingFeatureBuilder:
        if not examples:
            raise ValueError("at least one training example is required")
        single_dims = {
            item.embeddings.single.values.shape[-1] for item in examples if item.embeddings.single
        }
        pair_dims = {
            item.embeddings.pair.values.shape[-1] for item in examples if item.embeddings.pair
        }
        confidence_dims = {
            item.embeddings.confidence_features.numel()
            for item in examples
            if item.embeddings.confidence_features is not None
        }
        metadata_keys = {tuple(sorted((item.metadata or {}).keys())) for item in examples}
        self.schema = FeatureSchema(
            _one_dimension(single_dims, "single"),
            _one_dimension(pair_dims, "pair"),
            _one_dimension(confidence_dims, "confidence"),
            _one_metadata_schema(metadata_keys),
        )
        return self

    def transform(self, examples: Sequence[ModelExample]) -> np.ndarray:
        if self.schema is None:
            raise RuntimeError("fit the feature builder before transforming examples")
        return np.stack([self._transform_one(item) for item in examples]).astype(np.float32)

    def _transform_one(self, item: ModelExample) -> np.ndarray:
        assert self.schema is not None
        embedding, schema = item.embeddings, self.schema
        chunks = [
            _single_features(embedding, schema.single_dim),
            _pair_features(embedding, schema.pair_dim),
            _confidence_features(embedding, schema.confidence_dim),
            np.asarray(
                [embedding.single is not None, embedding.pair is not None], dtype=np.float32
            ),
        ]
        actual_keys = tuple(sorted((item.metadata or {}).keys()))
        if actual_keys != schema.metadata_keys:
            raise ValueError(
                f"metadata keys for {item.protein_id} differ from fitted schema: "
                f"expected {schema.metadata_keys}, got {actual_keys}"
            )
        if schema.metadata_keys:
            metadata = item.metadata
            if metadata is None:
                raise ValueError(f"metadata is missing for {item.protein_id}")
            chunks.append(
                np.asarray([metadata[key] for key in schema.metadata_keys], dtype=np.float32)
            )
        return np.concatenate(chunks)


class EmbeddingClassifier:
    """Shared fit/predict/save facade for pooled frozen-embedding baselines."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_builder = EmbeddingFeatureBuilder()
        self.estimator: Pipeline | None = None
        self.scaler: StandardScaler | None = None
        self.network: FeatureMLP | None = None
        self.train_result: TrainResult | None = None

    def fit(
        self,
        train_examples: Sequence[ModelExample],
        validation_examples: Sequence[ModelExample] | None = None,
    ) -> EmbeddingClassifier:
        labels = _labels(train_examples)
        features = self.feature_builder.fit(train_examples).transform(train_examples)
        validation_examples = validation_examples or train_examples
        validation_labels = _labels(validation_examples)
        validation_features = self.feature_builder.transform(validation_examples)
        if self.config.kind in {"lasso", "ridge"}:
            self.estimator = _linear_pipeline(self.config)
            self.estimator.fit(features, labels)
            return self
        self.scaler = StandardScaler().fit(features)
        train_tensor = torch.from_numpy(self.scaler.transform(features).astype(np.float32))
        validation_tensor = torch.from_numpy(
            self.scaler.transform(validation_features).astype(np.float32)
        )
        torch.manual_seed(self.config.random_state)
        self.network = FeatureMLP(
            features.shape[1], self.config.hidden_dim, self.config.dropout, self.config.activation
        )
        self.train_result = train_feature_mlp(
            self.network,
            train_tensor,
            torch.from_numpy(labels.astype(np.float32)),
            validation_tensor,
            torch.from_numpy(validation_labels.astype(np.float32)),
            max_epochs=self.config.max_epochs,
            patience=self.config.patience,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
        )
        return self

    def predict_proba(self, examples: Sequence[ModelExample]) -> np.ndarray:
        features = self.feature_builder.transform(examples)
        if self.estimator is not None:
            return self.estimator.predict_proba(features)[:, 1]
        if self.network is not None and self.scaler is not None:
            self.network.eval()
            with torch.no_grad():
                logits = self.network(
                    torch.from_numpy(self.scaler.transform(features).astype(np.float32)).to(
                        resolve_device(self.config.device)
                    )
                )
            return torch.sigmoid(logits).cpu().numpy()
        raise RuntimeError("fit or load the model before prediction")

    def predict_logit(self, examples: Sequence[ModelExample]) -> np.ndarray:
        probability = np.clip(self.predict_proba(examples), 1e-6, 1 - 1e-6)
        return np.log(probability / (1 - probability))

    def predict(self, examples: Sequence[ModelExample], threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        return (self.predict_proba(examples) >= threshold).astype(np.int64)

    def save(self, directory: str | Path) -> Path:
        if self.feature_builder.schema is None:
            raise RuntimeError("cannot save an unfitted model")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "config": asdict(self.config),
            "feature_schema": asdict(self.feature_builder.schema),
        }
        if self.estimator is not None:
            joblib.dump(self.estimator, destination / "sklearn.joblib")
            manifest["artifact"] = "sklearn.joblib"
        elif self.network is not None and self.scaler is not None:
            save_checkpoint(
                destination / "model.pt",
                self.network,
                input_dim=self.feature_builder.schema.input_dim,
            )
            joblib.dump(self.scaler, destination / "scaler.joblib")
            manifest["artifact"] = "model.pt"
        else:
            raise RuntimeError("cannot save an unfitted model")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingClassifier:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        config_values = dict(manifest["config"])
        if isinstance(config_values.get("class_weight"), dict):
            config_values["class_weight"] = {
                int(key): value for key, value in config_values["class_weight"].items()
            }
        model = cls(ModelConfig(**config_values))
        schema_values = dict(manifest["feature_schema"])
        schema_values["metadata_keys"] = tuple(schema_values["metadata_keys"])
        model.feature_builder = EmbeddingFeatureBuilder(FeatureSchema(**schema_values))
        if manifest["artifact"] == "sklearn.joblib":
            model.estimator = joblib.load(directory / "sklearn.joblib")
        elif manifest["artifact"] == "model.pt":
            schema = model.feature_builder.schema
            assert schema is not None
            model.network = FeatureMLP(
                schema.input_dim,
                model.config.hidden_dim,
                model.config.dropout,
                model.config.activation,
            )
            load_checkpoint(directory / "model.pt", model.network)
            model.scaler = joblib.load(directory / "scaler.joblib")
        else:
            raise ValueError("unsupported model artifact")
        return model


def create_model(config: ModelConfig) -> EmbeddingClassifier:
    """Create a supported baseline model; CNNs are reserved for a later family."""
    return EmbeddingClassifier(config)


def load_model(directory: str | Path) -> EmbeddingClassifier:
    """Load a saved :class:`EmbeddingClassifier` artifact."""
    return EmbeddingClassifier.load(directory)


def _one_dimension(values: set[int], name: str) -> int:
    if len(values) > 1:
        raise ValueError(f"inconsistent {name} embedding dimensions: {sorted(values)}")
    return next(iter(values), 0)


def _one_metadata_schema(values: set[tuple[str, ...]]) -> tuple[str, ...]:
    if len(values) != 1:
        raise ValueError("all training examples must use the same metadata keys")
    return next(iter(values))


def _single_features(embedding: ProteinEmbeddings, expected_dim: int) -> np.ndarray:
    if embedding.single is None:
        return np.zeros(expected_dim * 3, dtype=np.float32)
    if embedding.single.values.shape[-1] != expected_dim:
        raise ValueError("single embedding dimension differs from fitted schema")
    values = pool_single(
        embedding.single.values.unsqueeze(0), embedding.single.residue_mask.unsqueeze(0)
    )[0]
    return values.detach().cpu().numpy()


def _pair_features(embedding: ProteinEmbeddings, expected_dim: int) -> np.ndarray:
    if embedding.pair is None:
        return np.zeros(expected_dim * 12, dtype=np.float32)
    if embedding.pair.values.shape[-1] != expected_dim:
        raise ValueError("pair embedding dimension differs from fitted schema")
    values = pool_pair(embedding.pair.values.unsqueeze(0), embedding.pair.pair_mask.unsqueeze(0))[0]
    return values.detach().cpu().numpy()


def _confidence_features(embedding: ProteinEmbeddings, expected_dim: int) -> np.ndarray:
    if embedding.confidence_features is None:
        return np.zeros(expected_dim, dtype=np.float32)
    if embedding.confidence_features.numel() != expected_dim:
        raise ValueError("confidence feature dimension differs from fitted schema")
    return embedding.confidence_features.detach().cpu().numpy().astype(np.float32)


def _labels(examples: Sequence[ModelExample]) -> np.ndarray:
    if not examples or any(item.label is None for item in examples):
        raise ValueError("all training and validation examples require binary labels")
    labels = np.asarray([item.label for item in examples], dtype=np.int64)
    if np.unique(labels).size < 2:
        raise ValueError("training labels must contain both classes")
    return labels


def _linear_pipeline(config: ModelConfig) -> Pipeline:
    options: dict[str, object] = {
        "C": 1.0 / config.regularization_strength,
        "class_weight": config.class_weight,
        "max_iter": 2000,
        "random_state": config.random_state,
    }
    options.update({"solver": "saga", "l1_ratio": 1.0 if config.kind == "lasso" else 0.0})
    return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(**options))])
