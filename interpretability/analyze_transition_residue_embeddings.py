"""Compare transition-important residues with other residues in the same protein."""

# ruff: noqa: E402 - direct execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from protein_state_router.evaluation.inference import benjamini_hochberg, paired_sign_flip_test

from interpretability.contracts import load_residue_matrix

ROOT = Path("data/lifecycle/final/initial_8598_dataset")
ANALYSIS = ROOT / "analysis"
DEFAULT_OUTPUT = Path("interpretability/results/homology35_transition_residue_embedding_analysis")
TOP_N = (1, 3, 6, 10)


def _bootstrap_ci(values: np.ndarray, seed: int = 42, repeats: int = 2_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _paired_row(
    metric: str,
    ranker: str,
    top_n: int,
    per_protein: list[tuple[float, float]],
) -> dict[str, object]:
    top = np.asarray([pair[0] for pair in per_protein], dtype=np.float64)
    rest = np.asarray([pair[1] for pair in per_protein], dtype=np.float64)
    difference = top - rest
    lower, upper = _bootstrap_ci(difference, seed=42 + top_n)
    permutation = paired_sign_flip_test(difference, seed=42 + top_n)
    return {
        "ranker": ranker,
        "top_n": top_n,
        "metric": metric,
        "proteins": len(difference),
        "top_residue_mean": float(top.mean()),
        "other_residue_mean": float(rest.mean()),
        "paired_mean_difference": float(difference.mean()),
        "paired_difference_ci95_low": lower,
        "paired_difference_ci95_high": upper,
        "paired_cohen_dz": float(difference.mean() / difference.std(ddof=1)),
        "paired_permutation_p": permutation["permutation_p_two_sided"],
        "proteins_top_gt_other": int(np.count_nonzero(difference > 0)),
        "proteins_top_lt_other": int(np.count_nonzero(difference < 0)),
    }


def analyze(catalog: pd.DataFrame, residues: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    catalog = catalog.set_index("protein_id", verify_integrity=True)
    grouped = residues.groupby("protein_id", sort=False)
    rankers = {
        "ca_displacement_after_global_kabsch_angstrom": "displacement",
        "prs_max_overlap": "prs",
    }
    comparisons: dict[tuple[str, int, str], list[tuple[float, float]]] = {}
    missing_embedding: list[str] = []
    for index, (protein_id, group) in enumerate(grouped, start=1):
        row = catalog.loc[protein_id]
        path = Path(row.embedding_path)
        if not path.exists():
            missing_embedding.append(protein_id)
            continue
        values = load_residue_matrix(
            path,
            protein_id=protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
        ).astype(np.float64, copy=False)
        positions = group.canonical_residue_number.to_numpy(dtype=int) - 1
        if positions.min() < 0 or positions.max() >= len(values):
            raise ValueError(f"canonical residue position outside embedding for {protein_id}")
        selected = values[positions]
        mean = values.mean(axis=0)
        norms = np.linalg.norm(selected, axis=1)
        distances = np.linalg.norm(selected - mean, axis=1)
        cosine = selected @ mean / np.maximum(norms * np.linalg.norm(mean), 1e-12)
        contribution = selected @ weights
        group_metrics = {
            "embedding_l2_norm": norms,
            "distance_to_protein_mean": distances,
            "cosine_to_protein_mean": cosine,
            "frozen_linear_mean_block_contribution": contribution,
        }
        for rank_column, ranker in rankers.items():
            order = np.argsort(-group[rank_column].to_numpy(dtype=float), kind="stable")
            for top_n in TOP_N:
                if len(order) <= top_n:
                    continue
                top_mask = np.zeros(len(order), dtype=bool)
                top_mask[order[:top_n]] = True
                for metric, values_for_metric in group_metrics.items():
                    comparisons.setdefault((ranker, top_n, metric), []).append(
                        (
                            float(values_for_metric[top_mask].mean()),
                            float(values_for_metric[~top_mask].mean()),
                        )
                    )
        if index == 1 or index % 100 == 0 or index == len(grouped):
            print(f"TRANSITION_EMBEDDING_ANALYSIS {index}/{len(grouped)}", flush=True)
    if missing_embedding:
        raise FileNotFoundError(f"missing embeddings for {len(missing_embedding)} proteins")
    for (ranker, top_n, metric), per_protein in comparisons.items():
        rows.append(_paired_row(metric, ranker, top_n, per_protein))
    result = pd.DataFrame(rows).sort_values(["ranker", "top_n", "metric"]).reset_index(drop=True)
    result["paired_permutation_fdr"] = benjamini_hochberg(
        result.paired_permutation_p.to_numpy(dtype=float)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "homology35_seed42/catalog.parquet")
    parser.add_argument(
        "--displacement",
        type=Path,
        default=ANALYSIS / "homology35_dynamic_transition_residue_ca_displacements.csv",
    )
    parser.add_argument(
        "--prs", type=Path, default=ANALYSIS / "homology35_dynamic_transition_prs_scores.csv"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "ml/results/homology35_frozen_models/esmfold_single_linear/linear_weights.npz"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = pd.read_parquet(args.catalog)
    dynamic_test = catalog.loc[catalog.split.eq("test") & catalog.dataset_label.eq(1)].copy()
    displacement = pd.read_csv(args.displacement)
    prs = pd.read_csv(args.prs)
    key = ["protein_id", "sequence_sha256", "canonical_residue_number"]
    residues = displacement[key + ["ca_displacement_after_global_kabsch_angstrom"]].merge(
        prs[key + ["prs_max_overlap"]], on=key, validate="one_to_one"
    )
    if dynamic_test.empty or dynamic_test.protein_id.duplicated().any():
        raise ValueError("homology-grouped dynamic test partition is empty or duplicated")
    if not set(residues.protein_id).issubset(set(dynamic_test.protein_id)):
        raise ValueError("transition residue tables contain rows outside the dynamic test split")
    if residues.protein_id.nunique() < 1:
        raise ValueError("transition residue tables contain no completed proteins")
    with np.load(args.weights, allow_pickle=False) as archive:
        coefficients = archive["raw_coefficients"]
        if coefficients.ndim != 1 or len(coefficients) % 3:
            raise ValueError("pooled linear coefficients must contain mean, std, and max blocks")
        weights = coefficients[: len(coefficients) // 3]
    result = analyze(dynamic_test, residues, weights)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "within_protein_top_residue_embedding_contrasts.csv", index=False)
    summary = {
        "population": "completed dynamic proteins in the homology-grouped seed-42 test set",
        "eligible_dynamic_test_proteins": len(dynamic_test),
        "completed_structure_prs_proteins": int(residues.protein_id.nunique()),
        "excluded_structural_qc_proteins": int(len(dynamic_test) - residues.protein_id.nunique()),
        "aligned_residues": int(len(residues)),
        "top_n": list(TOP_N),
        "metrics": {
            "embedding_l2_norm": "L2 norm of the selected residue embedding",
            "distance_to_protein_mean": "L2 distance from that protein's residue-embedding mean",
            "cosine_to_protein_mean": "cosine similarity to that protein's residue-embedding mean",
            "frozen_linear_mean_block_contribution": (
                "raw projection onto the mean-pooling block of the frozen seed-42 linear classifier"
            ),
        },
        "comparison": "top N residues by annotation score versus remaining aligned residues within protein",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
