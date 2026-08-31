# Data module

This package validates catalogs, checks leakage, loads the canonical dataset,
and prepares batches for future PyTorch training.

- `schema.py` defines catalog records and validation rules.
- `catalog.py` reads and normalizes CSV or Parquet catalogs.
- `assembly.py` combines labeled tables when constructing a derived dataset.
- `quality_checks.py` validates labels, hashes, source counts, and bias reports.
- `splitting.py` creates deterministic train, validation, and test assignments.
- `dataset.py` and `collate.py` provide variable-length PyTorch data loading.

The active biological dataset is already assembled at
`data/lifecycle/final/initial_8598_dataset/`.
