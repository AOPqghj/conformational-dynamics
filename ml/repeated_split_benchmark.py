"""Run paired ten-split benchmarks and compare embedding versus sequence models."""

# ruff: noqa: E402 - direct execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
from protein_state_router.evaluation.inference import benjamini_hochberg, paired_sign_flip_test
from scripts.datasets.make_router_dataset_splits import make_splits
from train_suite import run_suite

ROOT = Path("data/lifecycle/final/initial_8598_dataset")
BASE = ROOT / "homology35_seed42/catalog.parquet"
EMBEDDING_MANIFEST = ROOT / "embedding_manifest.csv"
OUTPUT = Path("ml/results/homology35_repeated_splits")
SEEDS = tuple(range(10, 20))


def split_catalog(frame: pd.DataFrame, seed: int, path: Path) -> None:
    assignments, _ = make_splits(frame, seed, group_column="homology_group_id")
    result = frame.drop(columns="split", errors="ignore").merge(
        assignments[["protein_id", "split"]], on="protein_id", validate="one_to_one"
    )
    result.to_parquet(path, index=False)


def main() -> None:
    frame = pd.read_parquet(BASE)
    records = []
    for seed in SEEDS:
        dataset = OUTPUT / f"split_{seed}.parquet"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        split_catalog(frame, seed, dataset)
        result = run_suite(
            OUTPUT / f"seed_{seed}",
            dataset,
            dataset,
            EMBEDDING_MANIFEST,
            device="cpu",
            seed=seed,
            search="standard",
            include_cnn=False,
            save_models=False,
        )
        for model in result["models"]:
            if model.get("status") == "completed":
                records.append({"seed": seed, **model})
        (OUTPUT / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_seeds": seed - SEEDS[0] + 1,
                    "total_seeds": len(SEEDS),
                },
                indent=2,
            )
            + "\n"
        )
        partial = pd.DataFrame(records)
        partial.to_csv(OUTPUT / "live_metrics.csv", index=False)
        (OUTPUT / "live_report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Live repeated split benchmark</title>"
            "<style>body{font:15px system-ui;max-width:1100px;margin:32px auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px}</style>"
            f"<h1>Live repeated split benchmark</h1><p>Completed splits: {seed - 9}/{len(SEEDS)}</p>"
            + partial.to_html(index=False, float_format=lambda x: f"{x:.5f}")
        )

    results = pd.DataFrame(records)
    results.to_csv(OUTPUT / "all_metrics.csv", index=False)
    model_summary = results.groupby(["name", "family", "feature_view"], as_index=False).agg(
        splits=("seed", "count"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_best=("test_accuracy", "max"),
        accuracy_worst=("test_accuracy", "min"),
        auroc_mean=("test_auroc", "mean"),
        auroc_best=("test_auroc", "max"),
        auroc_worst=("test_auroc", "min"),
        auprc_mean=("test_auprc", "mean"),
        auprc_best=("test_auprc", "max"),
        auprc_worst=("test_auprc", "min"),
    )
    model_summary.to_csv(OUTPUT / "model_summary.csv", index=False)
    reports = []
    for family in sorted(results.family.unique()):
        seq = results[(results.family == family) & (results.feature_view == "sequence")]
        emb = results[(results.family == family) & (results.feature_view == "esmfold_single")]
        paired = seq.merge(emb, on="seed", suffixes=("_sequence", "_embedding"))
        if len(paired) == len(SEEDS):
            report = {"family": family, "n_splits": len(paired)}
            for metric in ("test_accuracy", "test_auroc", "test_auprc"):
                sequence = paired[f"{metric}_sequence"]
                embedding = paired[f"{metric}_embedding"]
                test = paired_sign_flip_test((embedding - sequence).to_numpy(), seed=seed)
                report[f"mean_sequence_{metric.removeprefix('test_')}"] = sequence.mean()
                report[f"mean_embedding_{metric.removeprefix('test_')}"] = embedding.mean()
                report[f"paired_{metric.removeprefix('test_')}_p_value"] = test[
                    "permutation_p_two_sided"
                ]
                report[f"paired_{metric.removeprefix('test_')}_method"] = test["permutation_method"]
            reports.append(report)
    summary = pd.DataFrame(reports)
    p_columns = [column for column in summary if column.endswith("_p_value")]
    if p_columns:
        adjusted = benjamini_hochberg(summary[p_columns].to_numpy().ravel())
        for index, column in enumerate(p_columns):
            summary[column.replace("_p_value", "_fdr")] = adjusted.reshape(
                len(summary), len(p_columns)
            )[:, index]
    summary.to_csv(OUTPUT / "paired_permutation_tests.csv", index=False)
    ranking = model_summary.sort_values("accuracy_best", ascending=False)
    html = (
        """<!doctype html><meta charset='utf-8'><title>Ensemble repeated split benchmark</title>
<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;color:#222}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}h1{color:#17324d}</style>
<h1>Ensemble initial 8,598 repeated-split benchmark</h1>
<p>Ten paired MMseqs2 homology-grouped train/validation/test splits. Hyperparameters are selected on validation AUROC and all final metrics are reported on the held-out test partition.</p>
<h2>Best test accuracy by model</h2>
"""
        + ranking[["name", "family", "feature_view", "splits", "accuracy_best"]].to_html(
            index=False, float_format=lambda x: f"{x:.5f}"
        )
        + "<h2>Best, worst, and mean statistics by model</h2>"
        + model_summary.to_html(index=False, float_format=lambda x: f"{x:.5f}")
        + "<h2>All model metrics</h2>"
        + results.to_html(index=False, float_format=lambda x: f"{x:.5f}")
        + "<h2>Paired sign-flip permutation statistics</h2>"
        + summary.to_html(index=False, float_format=lambda x: f"{x:.5f}")
    )
    (OUTPUT / "repeated_split_report.html").write_text(html)
    (OUTPUT / "progress.json").write_text(
        json.dumps(
            {"status": "complete", "completed_seeds": len(SEEDS), "total_seeds": len(SEEDS)},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "splits": len(SEEDS),
                "metrics": str(OUTPUT / "all_metrics.csv"),
                "tests": str(OUTPUT / "paired_permutation_tests.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
