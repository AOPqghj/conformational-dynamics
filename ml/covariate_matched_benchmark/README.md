# Covariate-matched ESMFold benchmark

This isolated experiment tests whether ESMFold embeddings retain predictive value after dynamic and static proteins are explicitly matched on measured covariates.
It does not modify the shared training or confounder pipelines.

For each saved homology-aware split, a regularized propensity model is fit on the training partition using log sequence length, amino-acid composition, sequence entropy, and mean pLDDT.
Dynamic and static proteins are matched separately within train, validation, and test partitions using training-derived propensity-score bins and a caliper.
This preserves the original split boundaries and prevents homology groups from crossing partitions.

The benchmark compares covariates, pooled ESMFold embeddings, and their concatenation using the repository's existing linear and tree model-selection pipeline.
The full pLDDT-observed cohort runs first over seeds 10-19.
A PathPre-only replication runs second over seeds 10-12.

Prepare an immutable plan, then run it:

```bash
uv run python ml/covariate_matched_benchmark/run.py prepare
uv run python ml/covariate_matched_benchmark/run.py run --plan ml/results/covariate_matched_benchmark/run_plan.json
```

Live state is written to `ml/results/covariate_matched_benchmark/progress.json`.
Completed model directories are resumable and are only moved into place after all artifacts for that model have been written.
