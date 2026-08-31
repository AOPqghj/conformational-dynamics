# Models module

Models are deliberately small heads over frozen features.
The baseline question is whether representations help beyond simple sequence features.

## General pooled-embedding baselines

`baselines.py` is the dataset-agnostic public API for the first trainable
models. Create one `ModelExample` per protein with its `ProteinEmbeddings`, an
optional binary label, and optional numeric metadata. A `ModelConfig` selects
`lasso` (L1 logistic regression), `ridge` (L2 logistic regression), or `mlp`.
`create_model(config)` returns an `EmbeddingClassifier` with `fit`,
`predict_proba`, `predict_logit`, `predict`, and `save` methods; `load_model`
restores the portable artifact for inference.

The feature builder learns its layout from training examples and pools single
and pair tensors with the existing mask-aware functions. It appends explicit
availability flags, confidence features, and consistently named metadata. This
lets a future finalized catalog supply labels without changing the embedding
model interface. CNNs and residue-level architectures are deliberately future
model families; the first API rejects unsupported kinds rather than silently
changing input semantics.

## Calibration

`PlattCalibrator` fits a logistic calibration model to validation logits and
returns calibrated positive probabilities. `TemperatureScaler` is currently an
alias for the same implementation. Never fit it using held-out test labels.
