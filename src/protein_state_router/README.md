# `protein_state_router` package

This is the importable implementation. Its central design choice is a
backbone-neutral `ProteinRepresentation`: extractors translate model-specific
outputs into optional per-residue `single` tensors, optional residue-pair `pair`
tensors, boolean masks, confidence features, and provenance metadata. Everything
downstream works from that contract rather than importing AlphaFold, Boltz, or
Chai directly.

## End-to-end flow

`data` validates labeled protein records and assigns a leakage-aware split.
`representations` produces or loads a `ProteinRepresentation` and caches it.
`pooling` reduces `[L, d]` and `[L, L, d]` tensors into fixed feature vectors.
`models` trains pooled-vector baselines and residue-matrix encoders.
`training` supplies optimization and checkpoint utilities.
`evaluation` calibrates and reports probabilities.

## Main public starting points

- `ProteinRepresentation` — data contract shared by extractors, batching, and model inputs.
- `data.schema.validate_catalog(frame)` — validates and canonicalizes a pandas catalog before it is saved or split.
- `data.splitting.make_splits(catalog, mode=...)` — makes deterministic shared split assignments.
- `representations.PrecomputedRepresentationExtractor(root)` — loads a cache created by this package or an external extractor that follows the documented format.

Subpackage READMEs explain their input/output shapes and boundaries.
`esmfold_runner.py` provides the active embedding backend; AlphaFold and Chai are optional extension boundaries.
