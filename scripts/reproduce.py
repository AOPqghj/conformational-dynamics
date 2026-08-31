#!/usr/bin/env python3
"""List, preflight, run, or verify the paper experiment registry."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
REGISTRY = {
    "frozen-esmfold": ("ESMFOLD_CATALOG", "ml/train_frozen_8598_models.py"),
    "frozen-bioemu": ("BIOEMU_CATALOG", "ml/train_frozen_8598_models.py"),
    "source-prediction": ("ESMFOLD_CATALOG", "ml/run_dataset_source_prediction.py"),
    "source-heldout": ("ESMFOLD_CATALOG", "ml/run_source_heldout_benchmark.py"),
    "sae-esmfold": ("ESMFOLD_CATALOG", "ml/train_seed42_test_sae.py"),
    "sae-bioemu": ("BIOEMU_CATALOG", "ml/train_seed42_test_sae.py"),
    "transition-esmfold": ("ESMFOLD_SAE", "interpretability/analyze_sae_transition_residue_associations.py"),
    "transition-bioemu": ("BIOEMU_SAE", "interpretability/analyze_sae_transition_residue_associations.py"),
    "structural-validation": ("ESMFOLD_SAE", "interpretability/analyze_sae_feature_structural_roles.py"),
    "hinge-case-study": ("ESMFOLD_SAE", "interpretability/test_hinge_atlas_sae.py"),
}
DEFAULT_ARGS = {
    "frozen-esmfold": lambda p: ["--catalog", p, "--output-root", "outputs/frozen-esmfold", "--representation-name", "esmfold", "--embedding-width", "1024", "--preserve-catalog-split"],
    "frozen-bioemu": lambda p: ["--catalog", p, "--output-root", "outputs/frozen-bioemu", "--representation-name", "bioemu", "--embedding-width", "384", "--preserve-catalog-split"],
    "sae-esmfold": lambda p: ["--catalog", p, "--output-root", "outputs/sae-esmfold", "--representation-name", "esmfold", "--input-dim", "1024", "--latent-dim", "4096", "--top-k", "64", "--fit-partition", "train"],
    "sae-bioemu": lambda p: ["--catalog", p, "--output-root", "outputs/sae-bioemu", "--representation-name", "bioemu", "--input-dim", "384", "--latent-dim", "4096", "--top-k", "64", "--fit-partition", "train"],
    "source-heldout": lambda p: ["--esmfold-catalog", p, "--output", "outputs/source-heldout", "--representations", "esmfold", "--suite", "all", "--cpu-threads", "2"],
}


def verify() -> None:
    manifest = pd.read_csv(ROOT / "data/dataset_manifest.csv.gz")
    assert len(manifest) == manifest.protein_id.nunique() == manifest.sequence_sha256.nunique() == 8598
    assert manifest.dataset_label.value_counts().to_dict() == {0: 4309, 1: 4289}
    assert manifest.split.value_counts().to_dict() == {"train": 6020, "val": 1289, "test": 1289}
    assert manifest.groupby("homology_group_id").split.nunique().max() == 1
    assert int(manifest.bioemu_available.sum()) == 8572
    for line in (ROOT / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative == "checksums.sha256":
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    print("verified: 8,598 unique proteins, frozen homology groups, results, and file hashes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "preflight", "run", "verify"))
    parser.add_argument("experiment", nargs="?", choices=sorted(REGISTRY))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    if ns.action == "verify":
        verify()
        return
    if ns.action == "list":
        for name, (variable, script) in REGISTRY.items():
            print(f"{name:24} {variable:18} {script}")
        return
    if ns.action == "preflight":
        verify()
        missing = [p for p in (ROOT / "src", ROOT / "ml", ROOT / "interpretability") if not p.exists()]
        if missing:
            raise SystemExit(f"missing source trees: {missing}")
        print("preflight passed; set the catalog/checkpoint variables shown by `list` before full runs")
        return
    if not ns.experiment:
        parser.error("run requires an experiment name")
    variable, script = REGISTRY[ns.experiment]
    required = os.environ.get(variable)
    if not required or not Path(required).exists():
        raise SystemExit(f"set {variable} to an existing local artifact before running {ns.experiment}")
    env = os.environ.copy()
    env.update(THREAD_ENV)
    defaults = DEFAULT_ARGS.get(ns.experiment, lambda _path: [])(required)
    if not defaults and not ns.args:
        raise SystemExit(
            f"{ns.experiment} needs experiment-specific CLI arguments; see EXPERIMENTS.md "
            "and pass them after the experiment name"
        )
    subprocess.run([sys.executable, str(ROOT / script), *defaults, *ns.args], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
