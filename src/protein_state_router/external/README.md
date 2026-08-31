# External data adapters

This package contains narrow, cache-friendly adapters for public protein-data
sources and their local evidence formats. Modules return plain tables or
typed records; they do not train models or silently create labels.

- `dynamicmpnn.py` and `zenodo.py` read the trusted DynamicMPNN source data.
- `dynamicmpnn_positive.py` applies the positive-label evidence policy and
  provides cached UniProt/RCSB lookups used by dataset builders.
- `negative_api.py` discovers and enriches RCSB/SIFTS negative candidates.
- `structure_geometry.py` parses mmCIF C-alpha coordinates and compares
  aligned structures.
- `negative_labeling.py` contains the older, evidence-table-based negative
  policy.
- `promise.py` normalizes the official ProMiSE conformational-pair tables and
  retrieves only canonical RCSB chain sequences needed for the positive cohort.
- `pathpre.py` reads PATHpre's compact SS/MS chain tables and retains unresolved
  or cross-class records for audit rather than silently making them trainable.
- `atlas.py` parses official ATLAS metadata into conservative MD-supported
  low-flexibility candidates, explicitly marked non-training-ready.
- `evaluate.py` verifies frozen initial-2k model checksums before external
  scoring; it never fits, tunes, or recalibrates an artifact.

The assembled dataset retains source identifiers and evidence in its final
catalog, so these adapters are reusable import utilities rather than active
local data stores.
