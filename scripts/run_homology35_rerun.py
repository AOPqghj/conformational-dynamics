"""Run the homology-grouped model and interpretability rerun from YAML.

This orchestrator only launches subprocesses after validating the catalog and
embedding manifest. It writes one runner progress file and forwards every child
stdout line to the terminal so long jobs remain observable and resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    marker: Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_inputs(config: dict[str, Any], root: Path) -> tuple[Path, pd.DataFrame]:
    paths = config["paths"]
    representation = config.get("representation", {"name": "esmfold", "width": 1024})
    catalog_path = _path(paths["catalog"], root)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"catalog is missing: {catalog_path}")
    catalog = pd.read_parquet(catalog_path)
    required = {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
        "dataset_label",
        "split",
        "homology_group_id",
        "embedding_path",
    }
    if missing := required - set(catalog):
        raise ValueError(f"homology catalog is missing columns: {sorted(missing)}")
    expected_catalog_rows = int(config.get("cohort", {}).get("expected_catalog_rows", 8598))
    if len(catalog) != expected_catalog_rows or catalog.protein_id.duplicated().any():
        raise ValueError(f"expected the unique {expected_catalog_rows:,}-row homology catalog")
    if catalog.groupby("homology_group_id").split.nunique().max() != 1:
        raise ValueError("homology group crosses a split boundary")
    if "embedding_root" in representation:
        embedding_root = _path(str(representation["embedding_root"]), root)
        manifest = catalog[["protein_id", "sequence_sha256"]].copy()
        manifest["embedding_path"] = manifest.sequence_sha256.map(
            lambda value: str(embedding_root / f"{value}.npz")
        )
        manifest = manifest.drop(columns="sequence_sha256")
    else:
        manifest_path = _path(paths["embedding_manifest"], root)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"embedding manifest is missing: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
    if manifest.protein_id.duplicated().any() or not set(manifest.protein_id).issubset(
        set(catalog.protein_id)
    ):
        raise ValueError("embedding manifest contains duplicate or unknown proteins")
    exists = manifest.embedding_path.map(lambda path: _path(str(path), root).exists())
    missing_embeddings = manifest.loc[~exists, "embedding_path"].astype(str).tolist()
    allowed_missing = int(representation.get("allowed_missing", 0))
    if len(missing_embeddings) > allowed_missing:
        raise FileNotFoundError(f"missing embedding files, first entries: {missing_embeddings[:3]}")
    manifest = manifest.loc[exists].copy()
    expected_rows = int(representation.get("expected_rows", len(catalog)))
    if len(manifest) != expected_rows:
        raise ValueError(
            f"{representation['name']} embedding coverage must contain exactly "
            f"{expected_rows} rows, found {len(manifest)}"
        )
    manifest["embedding_path"] = manifest.embedding_path.map(
        lambda value: str(_path(str(value), root))
    )
    return catalog_path, manifest


def _completed(marker: Path) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    status = payload.get("status")
    # Immutable artifact manifests and checksum registries are written only when
    # their producing stage has completed, while resumable progress files carry
    # an explicit terminal status.
    return status in {"complete", "completed"} if status is not None else bool(payload)


def _stage_list(config: dict[str, Any], root: Path, catalog: Path, manifest: Path) -> list[Stage]:
    seed = str(config.get("seed", 42))
    representation = config.get("representation", {"name": "esmfold", "width": 1024})
    representation_name = str(representation["name"])
    embedding_width = str(representation["width"])
    training = config["training"]
    interpretation = config["interpretability"]
    paths = config["paths"]
    stages: list[Stage] = []

    pooled = training["pooled"]
    pooled_root = _path(pooled["output"], root)
    if pooled.get("enabled", True):
        pooled_command = [
            sys.executable,
            "ml/train_frozen_8598_models.py",
            "--catalog",
            str(catalog),
            "--output-root",
            str(pooled_root),
            "--seed",
            seed,
            "--cpu-threads",
            str(pooled.get("cpu_threads", 2)),
            "--representation-name",
            representation_name,
            "--embedding-width",
            embedding_width,
        ]
        if representation.get("preserve_catalog_split", False):
            pooled_command.append("--preserve-catalog-split")
        stages.append(
            Stage(
                "pooled_linear_tree",
                tuple(pooled_command),
                pooled_root / "frozen_model_registry.json",
            )
        )

    cnn = training["cnn"]
    cnn_root = _path(cnn["output"], root)
    if cnn.get("enabled", True):
        stages.append(
            Stage(
                "full_matrix_cnn",
                (
                    sys.executable,
                    "ml/train_residue_embedding_models.py",
                    "--catalog",
                    str(catalog),
                    "--embedding-manifest",
                    str(manifest),
                    "--output",
                    str(cnn_root),
                    "--seed",
                    seed,
                    "--device",
                    str(cnn.get("device", "auto")),
                    "--batch-size",
                    str(cnn.get("batch_size", 8)),
                    "--max-epochs",
                    str(cnn.get("max_epochs", 30)),
                    "--patience",
                    str(cnn.get("patience", 5)),
                    "--representation-name",
                    representation_name,
                    "--embedding-width",
                    embedding_width,
                    "--models",
                    *[
                        str(value)
                        for value in cnn.get("models", ["residue_cnn", "residue_cnn_expanded"])
                    ],
                ),
                cnn_root / "progress.json",
            )
        )

    sae = training["sae"]
    sae_root = _path(sae["output"], root)
    frozen_sae = _path(sae["frozen_output"], root)
    if sae.get("enabled", True):
        sae_train_command = [
            sys.executable,
            "ml/train_seed42_test_sae.py",
            "--catalog",
            str(catalog),
            "--output-root",
            str(sae_root),
            "--device",
            str(sae.get("device", "auto")),
            "--epochs",
            str(sae.get("epochs", 40)),
            "--batch-size",
            str(sae.get("batch_size", 1024)),
            "--residues-per-protein",
            str(sae.get("residues_per_protein", 64)),
            "--top-k",
            str(sae.get("top_k", 64)),
            "--expansion-factor",
            str(sae.get("expansion_factor", 4)),
            "--input-dim",
            embedding_width,
            "--representation-name",
            representation_name,
            "--seed",
            seed,
        ]
        if sae.get("latent_dim") is not None:
            sae_train_command.extend(("--latent-dim", str(sae["latent_dim"])))
        stages.extend(
            [
                Stage(
                    "sae_train",
                    tuple(sae_train_command),
                    sae_root / "progress.json",
                ),
                Stage(
                    "sae_heldout_evaluation",
                    (
                        sys.executable,
                        "ml/evaluate_seed42_sae.py",
                        "--catalog",
                        str(catalog),
                        "--checkpoint-root",
                        str(sae_root),
                        "--output-root",
                        str(sae_root / "heldout"),
                        "--device",
                        str(sae.get("device", "auto")),
                    ),
                    sae_root / "heldout/heldout_reconstruction_metrics.json",
                ),
                Stage(
                    "sae_freeze",
                    (
                        sys.executable,
                        "ml/freeze_seed42_sae.py",
                        "--source",
                        str(sae_root),
                        "--destination",
                        str(frozen_sae),
                    ),
                    frozen_sae / "manifest.json",
                ),
            ]
        )
    else:
        frozen_sae = _path(sae["frozen_output"], root)

    transition = interpretation["transition_pairs"]
    transition_root = _path(transition["output"], root)
    transition_summary = transition_root / "pair_summary.csv"
    transition_displacements = transition_root / "residue_ca_displacements.csv"
    transition_prs = transition_root / "prs_scores.csv"
    transition_progress = transition_root / "progress.json"
    if transition.get("enabled", True):
        stages.append(
            Stage(
                "transition_pairs_prs_displacement",
                (
                    sys.executable,
                    "scripts/run_test_transition_analysis.py",
                    "--candidates",
                    str(_path(paths["transition_candidates"], root)),
                    "--catalog",
                    str(catalog),
                    "--summary-output",
                    str(transition_summary),
                    "--displacement-output",
                    str(transition_displacements),
                    "--prs-output",
                    str(transition_prs),
                    "--progress-output",
                    str(transition_progress),
                ),
                transition_progress,
            )
        )

    transition_embeddings = interpretation["transition_embeddings"]
    transition_embeddings_root = _path(transition_embeddings["output"], root)
    if transition_embeddings.get("enabled", True):
        stages.append(
            Stage(
                "transition_embedding_contrasts",
                (
                    sys.executable,
                    "interpretability/analyze_transition_residue_embeddings.py",
                    "--catalog",
                    str(catalog),
                    "--displacement",
                    str(transition_displacements),
                    "--prs",
                    str(transition_prs),
                    "--weights",
                    str(pooled_root / f"{representation_name}_single_linear/linear_weights.npz"),
                    "--output",
                    str(transition_embeddings_root),
                ),
                transition_embeddings_root / "summary.json",
            )
        )

    associations = interpretation["sae_associations"]
    associations_root = _path(associations["output"], root)
    if associations.get("enabled", True):
        association_command = [
            sys.executable,
            "interpretability/analyze_sae_transition_residue_associations.py",
            "--catalog",
            str(catalog),
            "--displacement",
            str(transition_displacements),
            "--prs",
            str(transition_prs),
            "--sae-root",
            str(frozen_sae),
            "--output",
            str(associations_root),
            "--device",
            str(associations.get("device", "auto")),
            "--permutations",
            str(associations.get("permutations", 10000)),
        ]
        if associations.get("allow_incomplete", False):
            association_command.append("--allow-incomplete")
        stages.append(
            Stage(
                "sae_transition_associations",
                tuple(association_command),
                associations_root / "progress.json",
            )
        )

    router = interpretation["sae_router_tests"]
    router_root = _path(router["output"], root)
    if router.get("enabled", True):
        stages.append(
            Stage(
                "sae_router_feature_tests",
                (
                    sys.executable,
                    "interpretability/analyze_sae_router_feature_tests.py",
                    "--catalog",
                    str(catalog),
                    "--sae-root",
                    str(frozen_sae),
                    "--models-root",
                    str(pooled_root),
                    "--annotations",
                    str(_path(paths["annotations"], root)),
                    "--displacement",
                    str(transition_displacements),
                    "--prs",
                    str(transition_prs),
                    "--output",
                    str(router_root),
                    "--device",
                    str(router.get("device", "auto")),
                    "--feature-batch-size",
                    str(router.get("feature_batch_size", 4)),
                    "--random-controls",
                    str(router.get("random_controls", 100)),
                    "--bootstrap-draws",
                    str(router.get("bootstrap_draws", 2000)),
                    "--representation-name",
                    representation_name,
                ),
                router_root / "progress.json",
            )
        )

    structural = interpretation["sae_structural_roles"]
    structural_root = _path(structural["output"], root)
    if structural.get("enabled", True):
        stages.append(
            Stage(
                "sae_structural_roles",
                (
                    sys.executable,
                    "interpretability/analyze_sae_feature_structural_roles.py",
                    "--catalog",
                    str(catalog),
                    "--full-catalog",
                    str(_path("data/lifecycle/final/initial_8598_dataset/catalog.parquet", root)),
                    "--transition-catalog",
                    str(_path(paths["transition_candidates"], root)),
                    "--transition-summary",
                    str(transition_summary),
                    "--associations",
                    str(associations_root / "sae_feature_associations.csv"),
                    "--sae-root",
                    str(frozen_sae),
                    "--output",
                    str(structural_root),
                    "--device",
                    str(structural.get("device", "auto")),
                    "--features-per-track",
                    str(structural.get("features_per_track", 10)),
                ),
                structural_root / "progress.json",
            )
        )
    return stages


def run(config_path: Path) -> None:
    root = Path.cwd()
    config = yaml.safe_load(config_path.read_text())
    if config.get("version") != 1:
        raise ValueError("unsupported rerun YAML version")
    catalog_source, manifest_frame = _require_inputs(config, root)
    ml_root = _path(config["paths"]["ml_root"], root)
    interpretation_root = _path(config["paths"]["interpretability_root"], root)
    ml_root.mkdir(parents=True, exist_ok=True)
    catalog_frame = pd.read_parquet(catalog_source).drop(columns="embedding_path", errors="ignore")
    catalog_frame = catalog_frame.merge(manifest_frame, on="protein_id", validate="one_to_one")
    catalog = ml_root / "selected_embedding_catalog.parquet"
    manifest = ml_root / "selected_embedding_manifest.csv"
    catalog_frame.to_parquet(catalog, index=False)
    manifest_frame.to_csv(manifest, index=False)
    progress_path = ml_root / "rerun_progress.json"
    stages = _stage_list(config, root, catalog, manifest)
    state = {
        "status": "running",
        "started_at_utc": _now(),
        "updated_at_utc": _now(),
        "config": str(config_path),
        "stages": [
            {"name": stage.name, "status": "pending", "marker": str(stage.marker)}
            for stage in stages
        ],
    }
    _write_json(progress_path, state)
    print(
        json.dumps({"event": "preflight_complete", "catalog": str(catalog), "stages": len(stages)}),
        flush=True,
    )
    for index, stage in enumerate(stages):
        record = state["stages"][index]
        if config.get("resume", True) and _completed(stage.marker):
            record["status"] = "skipped_completed"
            print(json.dumps({"event": "stage_skipped", "stage": stage.name}), flush=True)
            continue
        record.update(
            {"status": "running", "started_at_utc": _now(), "command": list(stage.command)}
        )
        state["updated_at_utc"] = _now()
        _write_json(progress_path, state)
        print(
            json.dumps(
                {"event": "stage_started", "stage": stage.name, "command": list(stage.command)}
            ),
            flush=True,
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            stage.command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{stage.name}] {line.rstrip()}", flush=True)
        return_code = process.wait()
        if return_code != 0:
            record.update(
                {"status": "failed", "return_code": return_code, "finished_at_utc": _now()}
            )
            state.update({"status": "failed", "updated_at_utc": _now()})
            _write_json(progress_path, state)
            raise RuntimeError(f"stage failed: {stage.name} (exit {return_code})")
        record.update({"status": "completed", "finished_at_utc": _now()})
        state["updated_at_utc"] = _now()
        _write_json(progress_path, state)
        print(json.dumps({"event": "stage_completed", "stage": stage.name}), flush=True)
    state.update({"status": "complete", "updated_at_utc": _now()})
    _write_json(progress_path, state)
    print(
        json.dumps(
            {
                "event": "rerun_complete",
                "ml_root": str(ml_root),
                "interpretability_root": str(interpretation_root),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/homology35_rerun.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
