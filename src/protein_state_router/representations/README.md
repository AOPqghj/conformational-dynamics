# Representations

This package defines model-independent protein representation types and local
bundle I/O. Training code consumes `ProteinEmbeddings`; it does not import a
structure-model runtime.

- `embeddings.py` defines single and optional pair tensors with provenance.
- `bundle_io.py` reads and writes validated local safetensor bundles.
- `precomputed.py` loads existing cached representations.
- `esmfold_runner.py` creates frozen ESMFold v1 folding-trunk single embeddings.
- AlphaFold and Chai modules are extension boundaries, not live backends.

The sole supported live GPU path is `notebooks/09_esmfold_trunk_colab.ipynb`.
It consumes sequences staged through Google Drive and writes finite `[L,1024]`
single embeddings. `scripts/esmfold_dataset.py` prepares and imports them.
