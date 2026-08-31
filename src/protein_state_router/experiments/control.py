"""Immutable planning and safety gates for long-running experiments."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from protein_state_router.data.splitting import make_grouped_splits

SCHEMA_VERSION = 1
REQUIRED_CATALOG_COLUMNS = {
    "protein_id",
    "dataset_label",
    "homology_group_id",
    "alphafold_mean_plddt",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame.loc[:, columns].sort_values(columns[0], kind="stable")
    return hashlib.sha256(values.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _load_config(config_path: Path) -> dict[str, Any]:
    values = yaml.safe_load(config_path.read_text())
    if not isinstance(values, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    for key in (
        "paths",
        "expected_split_count",
        "split_seeds",
        "followup_seeds",
        "cpu_threads",
        "stages",
    ):
        if key not in values:
            raise ValueError(f"experiment configuration is missing {key!r}")
    seeds = [int(seed) for seed in values["split_seeds"]]
    expected = int(values["expected_split_count"])
    if len(seeds) != expected or len(set(seeds)) != expected:
        raise ValueError(
            f"expected exactly {expected} unique split seeds, received {len(seeds)}"
        )
    if int(values["cpu_threads"]) < 1:
        raise ValueError("cpu_threads must be positive")
    return values


def build_plan(root: Path, config_path: Path, *, verify_embedding_files: bool = True) -> dict[str, Any]:
    """Resolve and fingerprint the exact cohort and splits before computation."""
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    paths = config["paths"]
    catalog_path = _resolve(root, str(paths["catalog"]))
    manifest_path = _resolve(root, str(paths["embedding_manifest"]))
    output_root = _resolve(root, str(paths["output_root"]))
    catalog = pd.read_parquet(catalog_path)
    missing_columns = REQUIRED_CATALOG_COLUMNS - set(catalog)
    if missing_columns:
        raise ValueError(f"catalog missing columns: {sorted(missing_columns)}")
    if catalog.protein_id.duplicated().any():
        raise ValueError("catalog contains duplicate protein IDs")
    manifest = pd.read_csv(manifest_path)
    if set(manifest.columns) != {"protein_id", "embedding_path"}:
        raise ValueError("embedding manifest must contain protein_id and embedding_path only")
    if manifest.protein_id.duplicated().any():
        raise ValueError("embedding manifest contains duplicate protein IDs")
    unknown = set(manifest.protein_id) - set(catalog.protein_id)
    if unknown:
        raise ValueError(f"embedding manifest contains {len(unknown)} unknown proteins")
    if verify_embedding_files:
        missing_files = [path for path in manifest.embedding_path.astype(str) if not Path(path).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"embedding manifest contains {len(missing_files)} missing files; first: {missing_files[0]}"
            )

    eligible = catalog.loc[
        catalog.alphafold_mean_plddt.notna() & catalog.protein_id.isin(manifest.protein_id)
    ].copy()
    expected_catalog = int(config.get("cohort", {}).get("expected_catalog_rows", len(catalog)))
    expected_eligible = int(
        config.get("representation", {}).get("expected_plddt_rows", len(eligible))
    )
    if len(catalog) != expected_catalog:
        raise ValueError(f"catalog has {len(catalog)} rows, expected {expected_catalog}")
    if len(eligible) != expected_eligible:
        raise ValueError(f"eligible cohort has {len(eligible)} rows, expected {expected_eligible}")
    if eligible.dataset_label.nunique() != 2:
        raise ValueError("eligible cohort must contain both labels")

    identity_columns = ["protein_id", "dataset_label", "homology_group_id"]
    for candidate in ("sequence_sha256", "sequence_hash", "sequence"):
        if candidate in eligible:
            identity_columns.append(candidate)
            break
    split_hashes: dict[str, str] = {}
    for seed in config["split_seeds"]:
        assignment, _ = make_grouped_splits(
            eligible, int(seed), group_column="homology_group_id"
        )
        split_hashes[str(seed)] = _frame_hash(
            assignment, ["protein_id", "split"]
        )

    labels = eligible.dataset_label.value_counts().sort_index()
    representation = str(config.get("representation", {}).get("name", "esmfold"))
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "path": str(config_path),
            "sha256": _file_hash(config_path),
        },
        "inputs": {
            "catalog": str(catalog_path),
            "catalog_sha256": _file_hash(catalog_path),
            "embedding_manifest": str(manifest_path),
            "embedding_manifest_sha256": _file_hash(manifest_path),
            "embedding_rows": int(len(manifest)),
        },
        "cohort": {
            "definition": "embedding_covered_and_plddt_observed",
            "rows": int(len(eligible)),
            "groups": int(eligible.homology_group_id.nunique()),
            "labels": {str(key): int(value) for key, value in labels.items()},
            "sha256": _frame_hash(eligible, identity_columns),
            "identity_columns": identity_columns,
        },
        "splits": {
            "seeds": [int(seed) for seed in config["split_seeds"]],
            "sha256_by_seed": split_hashes,
        },
        "scope": {
            "pooled": "entire eligible cohort",
            "stratification_bins": list(config.get("stratification_bins", [])),
            "stratification_only": True,
            "comparable": not bool(config.get("exploratory_noncomparable", False)),
        },
        "execution": {
            "representation": representation,
            "stages": [str(stage) for stage in config["stages"]],
            "pooled_views": list(config.get("pooled_views", [])),
            "stratification_views": list(config.get("stratification_views", [])),
            "cpu_threads": int(config["cpu_threads"]),
            "parallel": bool(config.get("parallel", False)),
        },
        "output_root": str(output_root),
    }
    plan["plan_sha256"] = _json_hash(plan)
    return plan


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_plan(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    expected_sha = str(plan.get("plan_sha256", ""))
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _json_hash(unsigned) != expected_sha:
        raise ValueError("execution plan content does not match its plan_sha256")
    current = build_plan(root, Path(plan["config"]["path"]))
    if current != plan:
        raise RuntimeError("inputs or configuration changed after the execution plan was created")
    return current


def assert_comparable(contracts: list[dict[str, Any]]) -> None:
    if len(contracts) < 2:
        raise ValueError("at least two contracts are required")
    if any(not contract["scope"]["comparable"] for contract in contracts):
        raise ValueError("an exploratory or partial run cannot be used in a comparison")
    reference = contracts[0]
    required = {
        "cohort": reference["cohort"]["sha256"],
        "splits": reference["splits"]["sha256_by_seed"],
        "seeds": reference["splits"]["seeds"],
        "pooled_scope": reference["scope"]["pooled"],
    }
    for contract in contracts[1:]:
        observed = {
            "cohort": contract["cohort"]["sha256"],
            "splits": contract["splits"]["sha256_by_seed"],
            "seeds": contract["splits"]["seeds"],
            "pooled_scope": contract["scope"]["pooled"],
        }
        if observed != required:
            differing = [key for key in required if required[key] != observed[key]]
            raise ValueError(f"runs are not comparable; mismatched: {', '.join(differing)}")


def process_is_live(lock: dict[str, Any]) -> bool:
    if lock.get("host") not in {None, socket.gethostname()}:
        return False
    try:
        os.kill(int(lock["pid"]), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def run_verified_plan(root: Path, plan_path: Path, confirmation: str) -> None:
    plan = json.loads(plan_path.read_text())
    if confirmation != plan.get("plan_sha256"):
        raise ValueError("--confirm must exactly match the plan_sha256")
    verify_plan(root, plan)
    environment = {
        **os.environ,
        "PROTEIN_EXPERIMENT_PLAN_SHA": confirmation,
        "PYTHONUNBUFFERED": "1",
    }
    threads = str(plan["execution"]["cpu_threads"])
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        environment[name] = threads
    subprocess.run(
        [
            sys.executable,
            "ml/run_homology35_confounder_rerun.py",
            "--config",
            plan["config"]["path"],
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    contract = {**plan, "status": "completed"}
    write_plan(Path(plan["output_root"]) / "run_contract.json", contract)
