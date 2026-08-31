# Experiment architecture

This package owns reusable experiment behavior.
Executable commands belong in `scripts/`; raw outputs belong in ignored run storage.

## Canonical control flow

Long-running confounder experiments use `scripts/run_experiment.py` and have four operations:

1. `plan` validates every input and creates an immutable execution plan.
2. `run` accepts only that plan and its exact SHA-256 confirmation.
3. `status` reconciles progress JSON with the recorded live process.
4. `compare` refuses runs with different cohorts, splits, seeds, or pooled scope.

The plan fingerprints the eligible protein cohort and regenerates every homology-aware split in memory before computation.
The pooled cohort is all embedding-covered proteins with observed pLDDT.
Configured pLDDT bins apply only to stratified follow-up evaluation and never silently change the pooled cohort.

```bash
uv run python scripts/run_experiment.py plan \
  --config configs/homology35_confounder_rerun.yaml \
  --output /tmp/esmfold_confounder_plan.json

uv run python scripts/run_experiment.py run \
  --plan /tmp/esmfold_confounder_plan.json \
  --confirm PLAN_SHA256_PRINTED_BY_PREFLIGHT
```

Do not invoke `ml/run_homology35_confounder_rerun.py` directly.
It is retained temporarily as the implementation backend while reusable logic is moved into this package.

## Ownership map

- `control.py` owns immutable plans, fingerprints, comparison gates, process truth, and resource propagation.
- `benchmark.py` owns leakage-aware pooled baseline comparisons.
- `single_embedding_mlp.py` owns bounded single-embedding MLP selection.
- `dynamicmpnn_smoke.py` contains temporary-label smoke-test plumbing, not paper evidence.
- `scripts/run_experiment.py` is the only supported confounder launch interface.

## Result contract

Every completed canonical run writes `run_contract.json` beside its outputs.
Cross-representation statistics require identical cohort hashes, per-seed split hashes, seed sets, and metric scope.
Same-numbered seeds generated over different protein sets are not comparable.
Partial embedding cohorts must be declared exploratory and cannot enter inferential comparisons.
