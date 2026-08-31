# ruff: noqa: E402 - executable script needs the repository root before local imports.
"""Build the matched, homology-grouped PathPre-only control cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.datasets.make_router_dataset_splits import make_splits

EXPECTED_ROWS = 4395
EXPECTED_LABELS = {0: 3065, 1: 1330}
EXPECTED_PLDDT_ROWS = 3811


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(
    catalog_path: Path,
    esmfold_manifest_path: Path,
    bioemu_manifest_path: Path,
    output_root: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    catalog = pd.read_parquet(catalog_path)
    required = {"protein_id", "dataset_label", "homology_group_id", "split"}
    if missing := required - set(catalog):
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    if catalog.protein_id.duplicated().any():
        raise ValueError("catalog protein IDs must be unique")
    esmfold = pd.read_csv(esmfold_manifest_path)
    bioemu = pd.read_csv(bioemu_manifest_path)
    for name, manifest in (("esmfold", esmfold), ("bioemu", bioemu)):
        if manifest.protein_id.duplicated().any() or not {
            "protein_id",
            "embedding_path",
        }.issubset(manifest):
            raise ValueError(f"{name} manifest has an invalid identity contract")
    pathpre = catalog.loc[catalog.protein_id.astype(str).str.startswith("pathpre:")].copy()
    shared = set(esmfold.protein_id) & set(bioemu.protein_id)
    cohort = pathpre.loc[pathpre.protein_id.isin(shared)].copy()
    if len(cohort) != EXPECTED_ROWS:
        raise ValueError(f"matched PathPre coverage is {len(cohort)}, expected {EXPECTED_ROWS}")
    labels = cohort.dataset_label.value_counts().sort_index().to_dict()
    if labels != EXPECTED_LABELS:
        raise ValueError(f"unexpected matched PathPre labels: {labels}")
    assignments, split_report = make_splits(cohort, seed=seed, group_column="homology_group_id")
    cohort = cohort.drop(columns="split").merge(
        assignments[["protein_id", "split"]], on="protein_id", validate="one_to_one"
    )
    if cohort.groupby("homology_group_id").split.nunique().max() != 1:
        raise ValueError("PathPre homology group crosses the frozen split")
    plddt_rows = int(cohort.alphafold_mean_plddt.notna().sum())
    if plddt_rows != EXPECTED_PLDDT_ROWS:
        raise ValueError(f"PathPre pLDDT coverage is {plddt_rows}, expected {EXPECTED_PLDDT_ROWS}")
    cohort = cohort.sort_values("protein_id").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_output = output_root / "pathpre_matched_4395_catalog.parquet"
    split_output = output_root / "seed42_split.parquet"
    cohort.to_parquet(catalog_output, index=False)
    cohort[["protein_id", "homology_group_id", "dataset_label", "split"]].to_parquet(
        split_output, index=False
    )
    manifest_outputs: dict[str, Path] = {}
    for name, manifest in (("esmfold", esmfold), ("bioemu", bioemu)):
        selected = cohort[["protein_id"]].merge(
            manifest[["protein_id", "embedding_path"]], on="protein_id", validate="one_to_one"
        )
        if (
            len(selected) != EXPECTED_ROWS
            or not selected.embedding_path.map(lambda value: Path(str(value)).is_file()).all()
        ):
            raise FileNotFoundError(f"{name} manifest does not resolve all matched embeddings")
        destination = output_root / f"{name}_embedding_manifest.csv"
        selected.to_csv(destination, index=False)
        manifest_outputs[name] = destination
    report = {
        "status": "complete",
        "source_catalog": str(catalog_path),
        "catalog": str(catalog_output),
        "rows": len(cohort),
        "label_counts": {str(key): int(value) for key, value in labels.items()},
        "homology_groups": int(cohort.homology_group_id.nunique()),
        "mixed_label_groups": int(
            cohort.groupby("homology_group_id").dataset_label.nunique().gt(1).sum()
        ),
        "plddt_rows": plddt_rows,
        "split_report": split_report,
        "manifests": {name: str(path) for name, path in manifest_outputs.items()},
        "sha256": {
            "catalog": sha256(catalog_output),
            "split": sha256(split_output),
            **{name: sha256(path) for name, path in manifest_outputs.items()},
        },
    }
    atomic_json(output_root / "audit.json", report)
    return report


def main() -> None:
    dataset = Path("data/lifecycle/final/initial_8598_dataset")
    frozen = dataset / "homology35_seed42"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=frozen / "catalog.parquet")
    parser.add_argument("--esmfold-manifest", type=Path, default=dataset / "embedding_manifest.csv")
    parser.add_argument(
        "--bioemu-manifest", type=Path, default=frozen / "bioemu_8572_embedding_manifest.csv"
    )
    parser.add_argument("--output-root", type=Path, default=frozen / "pathpre_only_controls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.catalog,
                args.esmfold_manifest,
                args.bioemu_manifest,
                args.output_root,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
