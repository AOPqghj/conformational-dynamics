"""Run the homology-aware pLDDT confounder suite from one YAML configuration.

The linear and tree evaluations are refit for each grouped split.  Frozen
Seed-42 CNN predictions are intentionally reported only as descriptive,
single-held-out-split pLDDT strata, so this runner never assigns them a
repeated-split p-value.
"""

# ruff: noqa: E402 - legacy backend bootstraps sibling modules for direct import tests.

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import plddt_confounder_benchmark as benchmark
import plddt_followup as followup
import yaml
from repeated_split_benchmark import split_catalog


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _claim_run_lock(output_root: Path) -> None:
    """Prevent two runners from mutating one checkpoint tree concurrently."""
    lock = output_root / ".run.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"another confounder runner may already own {output_root}; inspect or remove {lock}"
        ) from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at_utc": _now(),
                "command": sys.argv,
                "plan_sha256": os.environ.get("PROTEIN_EXPERIMENT_PLAN_SHA"),
            },
            handle,
        )
        handle.write("\n")

    def release() -> None:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _settings(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("configuration must be a YAML mapping")
    required = {
        "paths",
        "split_seeds",
        "followup_seeds",
        "cpu_threads",
        "stages",
        "progress_file",
        "parallel",
    }
    if missing := required - set(values):
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    split_seeds = tuple(int(seed) for seed in values["split_seeds"])
    followup_seeds = tuple(int(seed) for seed in values["followup_seeds"])
    expected_split_count = int(values.get("expected_split_count", 10))
    if len(split_seeds) != expected_split_count or len(set(split_seeds)) != len(split_seeds):
        count_label = "ten" if expected_split_count == 10 else str(expected_split_count)
        raise ValueError(
            f"configuration must specify exactly {count_label} unique split seeds"
        )
    if not followup_seeds or len(set(followup_seeds)) != len(followup_seeds):
        raise ValueError("configuration must specify one or more unique follow-up seeds")
    if not set(followup_seeds).issubset(split_seeds):
        raise ValueError("follow-up seeds must be included in split seeds")
    stages = tuple(str(stage) for stage in values["stages"])
    allowed_stages = {"pooled", "followups"}
    if not stages or len(stages) != len(set(stages)) or not set(stages).issubset(allowed_stages):
        raise ValueError("stages must contain unique values from: pooled, followups")
    progress_file = Path(str(values["progress_file"]))
    if progress_file.name != str(values["progress_file"]):
        raise ValueError("progress_file must be a filename within the output root")
    return values


def _run_child(label: str, command: tuple[str, ...], root: Path) -> None:
    print(json.dumps({"event": "branch_started", "branch": label, "command": command}), flush=True)
    process = subprocess.Popen(
        command,
        cwd=root,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{label}] {line.rstrip()}", flush=True)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{label} branch failed with exit code {return_code}")
    print(json.dumps({"event": "branch_completed", "branch": label}), flush=True)


def _run_followup_children(
    root: Path,
    catalog_path: Path,
    manifest_path: Path,
    benchmark_root: Path,
    followup_root: Path,
    seeds: tuple[int, ...],
) -> None:
    common = (
        "--catalog",
        str(catalog_path),
        "--source",
        str(benchmark_root),
        "--output",
        str(followup_root),
        "--embedding-manifest",
        str(manifest_path),
        "--seeds",
        *(str(seed) for seed in seeds),
    )
    _run_child(
        "residualization",
        (sys.executable, "ml/plddt_followup.py", "--residual-only", *common),
        root,
    )
    _run_child(
        "plddt_stratification",
        (sys.executable, "ml/plddt_followup.py", "--stratified-only", *common),
        root,
    )


def _validate_inputs(root: Path, values: dict[str, Any]) -> tuple[Path, Path, pd.DataFrame]:
    paths = values["paths"]
    catalog_path = _resolve(root, paths["catalog"])
    manifest_path = _resolve(root, paths["embedding_manifest"])
    catalog = pd.read_parquet(catalog_path)
    required = {
        "protein_id",
        "dataset_label",
        "homology_group_id",
        "alphafold_mean_plddt",
    }
    if missing := required - set(catalog):
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    expected_catalog_rows = int(values.get("cohort", {}).get("expected_catalog_rows", 8598))
    if len(catalog) != expected_catalog_rows or catalog.protein_id.duplicated().any():
        raise ValueError(
            f"expected one unique {expected_catalog_rows:,}-protein homology-aware catalog"
        )
    manifest = pd.read_csv(manifest_path)
    if manifest.protein_id.duplicated().any() or not set(manifest.protein_id).issubset(
        set(catalog.protein_id)
    ):
        raise ValueError("embedding manifest contains duplicate or unknown proteins")
    representation = values.get("representation", {})
    expected_rows = int(representation.get("expected_plddt_rows", 7032))
    eligible = catalog.loc[
        catalog.alphafold_mean_plddt.notna() & catalog.protein_id.isin(manifest.protein_id)
    ].copy()
    if len(eligible) != expected_rows or eligible.dataset_label.nunique() != 2:
        raise ValueError(
            f"expected {expected_rows:,} pLDDT-observed, embedding-covered proteins with both classes"
        )
    return catalog_path, manifest_path, eligible


def _configure_modules(
    catalog_path: Path,
    manifest_path: Path,
    split_root: Path,
    benchmark_root: Path,
    followup_root: Path,
    split_seeds: tuple[int, ...],
    followup_seeds: tuple[int, ...],
    representation_name: str,
    expected_rows: int,
) -> None:
    dataset_root = catalog_path.parent.parent
    benchmark.ROOT = dataset_root
    benchmark.CATALOG_PATH = catalog_path
    benchmark.SPLITS = split_root
    benchmark.OUTPUT = benchmark_root
    benchmark.EMBEDDING_MANIFEST = manifest_path
    benchmark.SEEDS = split_seeds
    benchmark.REPRESENTATION_NAME = representation_name
    benchmark.EXPECTED_ROWS = expected_rows
    followup.ROOT = dataset_root
    followup.CATALOG_PATH = catalog_path
    followup.SOURCE = benchmark_root
    followup.OUTPUT = followup_root
    followup.STRATIFIED_OUTPUT = followup_root / "stratified"
    followup.EMBEDDING_MANIFEST = manifest_path
    followup.SEEDS = followup_seeds
    followup.REPRESENTATION_NAME = representation_name
    followup.EXPECTED_ROWS = expected_rows


def _frozen_cnn_strata(catalog: pd.DataFrame, cnn_root: Path, output: Path) -> pd.DataFrame:
    """Summarize immutable Seed-42 CNN predictions by established pLDDT strata."""
    metadata = catalog.loc[
        catalog.alphafold_mean_plddt.notna(),
        ["protein_id", "dataset_label", "alphafold_mean_plddt"],
    ].copy()
    metadata["plddt_stratum"] = followup.plddt_stratum(metadata.alphafold_mean_plddt)
    records: list[dict[str, object]] = []
    for model in ("residue_cnn", "residue_cnn_expanded"):
        path = cnn_root / model / "test_predictions.parquet"
        predictions = pd.read_parquet(path)[["protein_id", "dataset_label", "probability"]]
        if predictions.protein_id.duplicated().any():
            raise ValueError(f"frozen CNN predictions have duplicate IDs: {model}")
        joined = predictions.merge(
            metadata,
            on="protein_id",
            how="inner",
            suffixes=("_prediction", "_catalog"),
            validate="one_to_one",
        )
        if joined.empty or not joined.dataset_label_prediction.equals(joined.dataset_label_catalog):
            raise ValueError(f"frozen CNN label or identity mismatch: {model}")
        for stratum in followup.SELECTED_STRATA:
            values = joined.loc[joined.plddt_stratum.eq(stratum)]
            metrics = followup.bin_metrics(
                values.dataset_label_catalog.to_numpy(), values.probability.to_numpy()
            )
            records.append(
                {
                    "model": model,
                    "plddt_stratum": stratum,
                    **metrics,
                    "inference_status": "n/a_single_frozen_split",
                    "p_value_status": "n/a_single_frozen_split",
                }
            )
    result = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def _write_index(output_root: Path, cnn: pd.DataFrame) -> None:
    html = (
        "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
        "<title>Homology-aware confounder rerun</title>"
        "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;color:#17324d}"
        "table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #d8e0e8;padding:7px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style>"
        "<h1>Homology-aware pLDDT confounder rerun</h1>"
        "<p>Linear and tree models are retrained over configured homology-grouped splits. "
        "Frozen CNN summaries are descriptive Seed-42 strata and deliberately have no p-value.</p>"
        "<h2>Frozen CNN pLDDT strata</h2>"
        + cnn.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    )
    (output_root / "index.html").write_text(html)


def run(config_path: Path) -> None:
    root = Path.cwd()
    values = _settings(config_path)
    paths = values["paths"]
    split_seeds = tuple(int(seed) for seed in values["split_seeds"])
    followup_seeds = tuple(int(seed) for seed in values["followup_seeds"])
    stages = tuple(str(stage) for stage in values["stages"])
    parallel = bool(values["parallel"])
    followups_first = bool(values.get("followups_first", False))
    thread_count = int(values["cpu_threads"])
    os.environ["PLDDT_BENCHMARK_SEARCH"] = str(values.get("benchmark_search", "standard"))
    if thread_count < 1:
        raise ValueError("cpu_threads must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(thread_count)
    catalog_path, manifest_path, eligible = _validate_inputs(root, values)
    output_root = _resolve(root, paths["output_root"])
    _claim_run_lock(output_root)
    split_root = output_root / "splits"
    benchmark_root = output_root / "pooled_confounder"
    followup_root = output_root / "followup"
    progress = output_root / str(values["progress_file"])
    representation = values.get("representation", {})
    representation_name = str(representation.get("name", "esmfold"))
    expected_rows = int(representation.get("expected_plddt_rows", len(eligible)))
    requested_strata = tuple(values.get("stratification_bins", followup.STRATUM_LABELS))
    invalid_strata = set(requested_strata) - set(followup.STRATUM_LABELS)
    if not requested_strata or invalid_strata:
        raise ValueError(f"invalid stratification_bins: {sorted(invalid_strata)}")
    # Materialize the embedding-covered pLDDT subset so every downstream module
    # sees the same cohort, including controls with a sequence-length limit.
    output_root.mkdir(parents=True, exist_ok=True)
    confounder_catalog = output_root / "embedding_covered_plddt_catalog.parquet"
    eligible.to_parquet(confounder_catalog, index=False)
    _configure_modules(
        confounder_catalog,
        manifest_path,
        split_root,
        benchmark_root,
        followup_root,
        split_seeds,
        followup_seeds,
        representation_name,
        expected_rows,
    )
    followup.SELECTED_STRATA = requested_strata
    requested_views = tuple(values.get("stratification_views", followup.STRATIFIED_VIEWS))
    invalid_views = set(requested_views) - set(followup.STRATIFIED_VIEWS)
    if not requested_views or invalid_views:
        raise ValueError(f"invalid stratification_views: {sorted(invalid_views)}")
    followup.STRATIFIED_VIEWS = requested_views
    benchmark.FEATURE_VIEWS = tuple(
        str(view) for view in values.get("pooled_views", benchmark.FEATURE_VIEWS)
    )
    valid_pooled_views = {"covariates", "embedding", f"covariates_plus_{representation_name}"}
    if not benchmark.FEATURE_VIEWS or not set(benchmark.FEATURE_VIEWS).issubset(valid_pooled_views):
        raise ValueError(f"invalid pooled_views: {benchmark.FEATURE_VIEWS}")
    state: dict[str, Any] = {"status": "running", "updated_at_utc": _now(), "stages": {}}
    _write_json(progress, state)
    print(
        json.dumps(
            {
                "event": "preflight_complete",
                "eligible_proteins": len(eligible),
                "split_seeds": split_seeds,
                "followup_seeds": followup_seeds,
            }
        ),
        flush=True,
    )

    active_stage = "splits"
    try:
        for seed in split_seeds:
            path = split_root / f"split_{seed}.parquet"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                split_catalog(eligible, seed, path)
            split = pd.read_parquet(path)
            if split.groupby("homology_group_id").split.nunique().max() != 1:
                raise ValueError(f"homology group crosses a split in seed {seed}")
            source_split = benchmark_root / f"split_{seed}.parquet"
            if not source_split.exists():
                source_split.parent.mkdir(parents=True, exist_ok=True)
                split.to_parquet(source_split, index=False)
            print(json.dumps({"event": "split_ready", "seed": seed}), flush=True)
        state["stages"]["splits"] = "completed"
        _write_json(progress, state)

        if followups_first and "followups" in stages:
            active_stage = "residualization"
            print(json.dumps({"event": "stage_started", "stage": active_stage}), flush=True)
            followup.main(residual_only=True, seeds=followup_seeds)
            state["stages"][active_stage] = "completed"
            _write_json(progress, state)

        if parallel and not followups_first and set(stages) == {"pooled", "followups"}:
            active_stage = "parallel_pooled_and_followups"
            pooled_command = (
                sys.executable,
                "ml/plddt_confounder_benchmark.py",
                "--catalog",
                str(catalog_path),
                "--splits",
                str(split_root),
                "--output",
                str(benchmark_root),
                "--embedding-manifest",
                str(manifest_path),
                "--seeds",
                *(str(seed) for seed in split_seeds),
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                pooled_future = executor.submit(_run_child, "pooled_10", pooled_command, root)
                followup_future = executor.submit(
                    _run_followup_children,
                    root,
                    catalog_path,
                    manifest_path,
                    benchmark_root,
                    followup_root,
                    followup_seeds,
                )
                pooled_future.result()
                followup_future.result()
            state["stages"]["pooled_confounder"] = "completed"
            state["stages"]["residualization"] = "completed"
            state["stages"]["plddt_stratification"] = "completed"
            _write_json(progress, state)
        elif "pooled" in stages:
            active_stage = "pooled_confounder"
            print(json.dumps({"event": "stage_started", "stage": active_stage}), flush=True)
            benchmark.main()
            state["stages"][active_stage] = "completed"
            _write_json(progress, state)

        if "followups" in stages and not parallel:
            if not followups_first:
                active_stage = "residualization"
                print(json.dumps({"event": "stage_started", "stage": active_stage}), flush=True)
                followup.main(residual_only=True, seeds=followup_seeds)
                state["stages"][active_stage] = "completed"
                _write_json(progress, state)

            active_stage = "plddt_stratification"
            print(json.dumps({"event": "stage_started", "stage": active_stage}), flush=True)
            followup.main(stratified_only=True, seeds=followup_seeds)
            state["stages"][active_stage] = "completed"
            _write_json(progress, state)

        if "followups" in stages and bool(values.get("frozen_cnn", {}).get("enabled", True)):
            active_stage = "frozen_cnn_strata"
            frozen_cnn = _resolve(root, values["frozen_cnn"]["root"])
            cnn = _frozen_cnn_strata(eligible, frozen_cnn, output_root / "frozen_cnn_strata.csv")
            _write_index(output_root, cnn)
            state["stages"][active_stage] = "completed"
        state.update({"status": "completed", "updated_at_utc": _now()})
        _write_json(progress, state)
        print(json.dumps({"event": "completed", "output": str(output_root)}), flush=True)
    except BaseException as error:
        state.update(
            {
                "status": "failed",
                "failed_stage": active_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at_utc": _now(),
            }
        )
        _write_json(progress, state)
        print(
            json.dumps(
                {
                    "event": "failed",
                    "stage": active_stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )
        raise


def main() -> None:
    if not os.environ.get("PROTEIN_EXPERIMENT_PLAN_SHA"):
        raise RuntimeError(
            "direct execution is disabled for long-running confounder jobs; create and run an "
            "immutable plan with `uv run python scripts/run_experiment.py plan ...`"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/homology35_confounder_rerun.yaml")
    )
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
