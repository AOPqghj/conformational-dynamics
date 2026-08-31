# Protein Structure Embeddings and Conformational Dynamics

Anonymous reproducibility repository for the accompanying NeurIPS workshop submission.
It contains the complete 8,598-protein manifest, frozen homology-aware partitions, experiment implementations, compact result tables, and figure builders.
It excludes embeddings, checkpoints, structure caches, raw predictions, internal reviews, development notes, and machine-specific state.

## Quick start

```bash
uv sync --frozen
uv run python scripts/reproduce.py preflight
uv run python scripts/reproduce.py verify
uv run python scripts/reproduce.py list
```

See [REPRODUCING.md](REPRODUCING.md) for embedding generation and experiment commands, [EXPERIMENTS.md](EXPERIMENTS.md) for the paper-to-command map, and [DATASET_CARD.md](DATASET_CARD.md) for scope and limitations.

## Release boundaries

The labels are operational benchmark labels assembled from heterogeneous public sources; “static” does not mean physically immobile. The frozen split prevents MMseqs2 clusters formed at 35% sequence identity and 80% bidirectional coverage from crossing partitions, but it is not an absolute remote-homology guarantee. Source-held-out and PATHpre-only controls should be used when assessing generalization.

Code is MIT licensed.
Original annotations and manifest organization are CC BY 4.0; upstream records remain subject to their respective terms described in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
