# Training module

Training helpers are small on purpose: the MVP trains heads on already pooled,
frozen features rather than fine-tuning a protein foundation model.

- `trainer.train_feature_mlp(...)` is a full-batch binary trainer for a model
  that accepts one feature tensor and returns logits. It uses AdamW, gradient
  clipping, validation AUPRC early stopping, and restores the best state. Its
  return value gives the best epoch and validation AUPRC. Pass `device="mps"`
  on Apple Silicon, or `device="auto"` to select CUDA, MPS, then CPU.
- `losses.binary_loss(logits, labels, positive_weight=None)` wraps numerically
  stable binary cross-entropy with logits and optionally applies a positive-class
  weight.
- `checkpoints.save_checkpoint(path, model, **metadata)` writes a portable
  PyTorch state dictionary plus metadata; `load_checkpoint` restores it on CPU
  by default.

The ML runners use this trainer for bounded neural baselines and persist selected weights with their validation metadata.
