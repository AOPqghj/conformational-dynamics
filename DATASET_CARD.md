# Dataset card

## Contents

`data/dataset_manifest.csv.gz` contains 8,598 unique protein sequences: 4,289 operationally dynamic and 4,309 operationally static. Frozen partitions contain 6,020 train, 1,289 validation, and 1,289 test proteins. Each row records sequence, SHA-256, source provenance, operational label, MMseqs2 homology-group ID, split, selected identifiers/transition metadata, pLDDT where available, and BioEMU availability.

## Construction and intended use

Dynamic examples derive from DynamicMPNN, PATHpre, and ProMiSE; static examples derive from PATHpre, ATLAS, and an RCSB/PDB screen. Labels reflect source-specific screening criteria summarized in the paper and are intended for research on representation-level multistate signal, not clinical or safety-critical decisions. The manifest supports exact split reconstruction and source-aware analyses.

## Known limitations

Sources are heterogeneous and partially label-associated. A classifier can recover some source signal, and held-out-source performance drops. “Static” means no qualifying large transition was identified under the curation procedure, not absence of molecular motion. pLDDT is missing for some proteins. BioEMU embeddings cover 8,572 proteins. The sequence-cluster split reduces close-homology leakage but cannot exclude every remote evolutionary relationship.

## Data access and licensing

No embeddings, model weights, cached structures, or bulk upstream archives are included. Original manifest organization and annotations are CC BY 4.0; upstream sequence and structure records retain their original terms. See `THIRD_PARTY_LICENSES.md` and verify upstream terms before redistribution beyond this manifest.
