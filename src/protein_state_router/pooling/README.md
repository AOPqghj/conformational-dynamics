# Pooling module

Pooling turns variable-length backbone tensors into fixed-width vectors that
simple probes and MLPs can consume. All functions require boolean masks, so
padding is excluded rather than becoming artificial signal.

## Single features

`pool_single(single, residue_mask)` accepts `[B, L, d_single]` plus `[B, L]` and
returns `[B, 3*d_single]`: masked mean, population standard deviation, and
maximum over residues. The underlying `masked_mean_std_max` raises an error for
an entirely masked example, which prevents silent invalid features.

## Pair features

`pool_pair(pair, pair_mask)` accepts `[B, L, L, d_pair]` plus `[B, L, L]` and
returns `[B, 12*d_pair]`. It computes the same three statistics over four masks:
global (with the diagonal excluded by default), local separations 1–8, medium
separations 9–32, and long separations above 32 residues. Short proteins may
have an empty band; that band gets a zero feature block rather than failing.

`separation_band_masks` exposes those masks for inspection, and
`pair_per_residue_mean` provides an optional `[B, L, d_pair]` summary for future
residue-level or convolutional models. It is not used by the MVP classifiers.

Pooled baselines use these functions before fitting feature-level models.
Matrix-aware models instead consume the residue-level ESMFold matrix directly.
