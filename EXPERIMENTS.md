# Experiment registry

| Paper result | Registry name | Primary implementation | Inputs not shipped |
|---|---|---|---|
| Frozen ESMFold classifiers | `frozen-esmfold` | `ml/train_frozen_8598_models.py` | ESMFold residue embeddings |
| Frozen BioEMU classifiers | `frozen-bioemu` | `ml/train_frozen_8598_models.py` | BioEMU residue embeddings |
| Frozen ESM2 classifiers | standard frozen-model CLI | `ml/train_frozen_8598_models.py` | ESM2 residue embeddings |
| Repeated splits, residualization, pLDDT strata | signed experiment plan | `scripts/run_experiment.py` | corresponding embeddings |
| pLDDT-only and pLDDT-distribution baselines | direct CLI | `ml/run_plddt_only_benchmark.py`, `ml/run_plddt_distribution_benchmark.py` | AlphaFold pLDDT vectors |
| Single-structure geometry baseline | direct CLI | `ml/run_single_structure_geometry_baseline.py` | cached AlphaFold structures |
| Covariate-matched controls | direct CLI | `ml/covariate_matched_benchmark/run.py` | pooled ESMFold embeddings |
| Source prediction | `source-prediction` | `ml/run_dataset_source_prediction.py` | pooled ESMFold embeddings |
| Source-held-out tests | `source-heldout` | `ml/run_source_heldout_benchmark.py` | pooled ESMFold embeddings |
| PATHpre-only controls | `pathpre-esmfold`, `pathpre-bioemu` | `scripts/build_pathpre_only_controls.py` plus confounder runner | corresponding embeddings |
| ESMFold/BioEMU TopK SAE | `sae-esmfold`, `sae-bioemu` | `ml/train_seed42_test_sae.py` | residue embeddings |
| Transition/PRS associations | `transition-esmfold`, `transition-bioemu` | `interpretability/analyze_sae_transition_residue_associations.py` | frozen SAE and transition arrays |
| Protein-label enrichment and ablation | `router-features-*` | `interpretability/analyze_sae_router_feature_tests.py` | frozen SAE and embeddings |
| Independent validation structural roles | `structural-validation` | `interpretability/analyze_sae_feature_structural_roles.py` | SAE, mappings, public structures |
| Three-protein hinge case study | `hinge-case-study` | `interpretability/test_hinge_atlas_sae.py` | SAE, embeddings, public structures |
| Paper panels | `figures` | files under `AA-upgraded-neurips-workshop/figures/` | compact result tables |

The random-trunk and contact-chemistry experiments were unfinished or omitted from the paper and are intentionally outside this release.
Compact outputs are under `results/classification/` and `results/interpretability/`.
