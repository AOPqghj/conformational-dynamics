# Reproducing the experiments

## Environment

Install [uv](https://docs.astral.sh/uv/), clone this repository, and run:

```bash
uv sync --frozen
uv run python scripts/reproduce.py preflight
```

CPU model fitting is constrained with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `OPENBLAS_NUM_THREADS=2` in the registry commands. Neural models require a CUDA, MPS, or CPU PyTorch device; the reported ESMFold SAE and classifier experiments can be reproduced from regenerated residue embeddings. See [COMPUTE.md](COMPUTE.md).

## Regenerate embeddings

- ESMFold: open `notebooks/09_esmfold_trunk_colab.ipynb` in a CUDA Colab runtime and point it to `data/dataset_manifest.csv.gz`.
- BioEMU (MSA-aware): use `notebooks/28_bioemu_tpu_drive_pipeline.ipynb`; use notebook 29 to resume checkpointed runs.
- BioEMU without MSA: use `notebooks/30_bioemu_no_msa_colab.ipynb`.
- ESM2-3B: use `notebooks/31_esm2_3b_final_layer_colab.ipynb`.

The notebooks write one residue matrix per protein. Embeddings and upstream model weights are not redistributed. After generation, build a local catalog with embedding paths using the included catalog utilities, then pass its path through the environment variables printed by `scripts/reproduce.py list`.

## Run and verify

```bash
uv run python scripts/reproduce.py list
uv run python scripts/reproduce.py run frozen-esmfold
uv run python scripts/reproduce.py run sae-esmfold
uv run python scripts/reproduce.py verify
```

Set `ESMFOLD_CATALOG` or `BIOEMU_CATALOG` to a regenerated catalog before the corresponding frozen-model or SAE command.
For confounder experiments, update only the paths in the included configuration and create an immutable plan before launching:

```bash
uv run python scripts/run_experiment.py plan --config configs/homology35_confounder_rerun.yaml --output experiment_plan.json
uv run python scripts/run_experiment.py run --plan experiment_plan.json --confirm PLAN_SHA256_PRINTED_ABOVE
```

Commands fail before computation if required catalogs, embeddings, or checkpoints are absent.
Compact results under `results/` allow paper-value verification without regenerating large tensors.
Structural-role reproduction additionally downloads public structures from RCSB/PDB and therefore requires network access.
