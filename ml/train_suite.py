"""Run a complete router benchmark suite with live JSON/HTML status."""

# ruff: noqa: E402 - direct execution needs the repository root on sys.path.

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from interpretability.contracts import load_residue_matrix, pool_residue_matrix
from protein_state_router.experiments.benchmark import (
    BenchmarkConfig,
    embedding_cnn_tensor,
    run_benchmark,
    sequence_cnn_tensor,
    sequence_feature_matrix,
    single_embedding_feature_matrix,
)
from protein_state_router.representations.registry import (
    representation_choices,
    representation_spec,
)

try:
    from ml.report import build_report
except ModuleNotFoundError:  # Direct `python ml/train_suite.py` execution.
    from report import build_report

CENTRAL = ZoneInfo("America/Chicago")
FAMILIES = ("linear", "tree", "random_forest", "mlp")
EXHAUSTIVE_FAMILIES = ("svm", "knn", "naive_bayes")


def _central_now() -> datetime:
    return datetime.now(CENTRAL)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _central_now()).strftime("%Y-%m-%d %H:%M:%S %Z (Central Time)")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _pooled_single_features(
    catalog: pd.DataFrame, bundle_manifest: Path, representation_name: str = "esmfold"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read and pool each residue embedding exactly once for one catalog."""
    manifest = (
        pd.read_parquet(bundle_manifest)
        if bundle_manifest.suffix == ".parquet"
        else pd.read_csv(bundle_manifest)
    )
    if "embedding_path" in manifest:
        if manifest.protein_id.duplicated().any():
            raise ValueError("embedding manifest protein IDs must be unique")
        indexed = manifest.set_index("protein_id")
        missing = set(catalog.protein_id) - set(indexed.index)
        if missing:
            raise ValueError(f"embedding manifest is missing {len(missing)} catalog proteins")
        features: np.ndarray | None = None
        for index, row in enumerate(catalog.itertuples(index=False)):
            path = Path(indexed.at[row.protein_id, "embedding_path"])
            if hasattr(row, "embedding_path") and str(path) != str(row.embedding_path):
                raise ValueError(f"catalog and manifest embedding paths differ: {row.protein_id}")
            values = load_residue_matrix(
                path,
                protein_id=row.protein_id,
                sequence=row.sequence,
                sequence_sha256=row.sequence_sha256,
                sequence_length=int(row.sequence_length),
            )
            pooled = pool_residue_matrix(values)
            if features is None:
                features = np.empty((len(catalog), pooled.size), dtype=np.float32)
            elif features.shape[1] != pooled.size:
                raise ValueError("embedding widths must be consistent")
            features[index] = pooled
        if features is None:
            raise ValueError("embedding manifest is empty")
        names = tuple(
            f"{representation_name}_single_{stat}_{index}"
            for stat in ("mean", "std", "max")
            for index in range(features.shape[1] // 3)
        )
        return features, names
    features, names = single_embedding_feature_matrix(catalog, manifest, bundle_manifest.parent)
    return features, names


def _load_features(
    dataset: Path,
    feature_view: str,
    family: str,
    bundle_manifest: Path,
    pooled_single: tuple[np.ndarray, tuple[str, ...]] | None = None,
    representation_name: str = "esmfold",
) -> tuple[np.ndarray, tuple[str, ...]]:
    catalog = pd.read_parquet(dataset)
    if family == "cnn":
        return sequence_cnn_tensor(catalog)
    if feature_view == "sequence":
        return sequence_feature_matrix(catalog)
    features, names = pooled_single or _pooled_single_features(
        catalog, bundle_manifest, representation_name
    )
    if family == "embedding_cnn":
        return embedding_cnn_tensor(features)
    if feature_view == f"sequence_plus_{representation_name}":
        sequence_features, sequence_names = sequence_feature_matrix(catalog)
        features = np.concatenate((sequence_features, features), axis=1)
        names = (*sequence_names, *names)
    return features, names


def _jobs(
    dataset: Path,
    embedding_dataset: Path,
    search: str,
    include_cnn: bool = True,
    representation_name: str = "esmfold",
) -> list[dict[str, str]]:
    families = (*FAMILIES, *(EXHAUSTIVE_FAMILIES if search == "exhaustive" else ()))
    jobs = [
        {
            "name": f"sequence_{family}",
            "family": family,
            "feature_view": "sequence",
            "dataset": str(dataset),
        }
        for family in (*families, *(("cnn",) if include_cnn else ()))
    ] + [
        {
            "name": f"{view}_{family}",
            "family": family,
            "feature_view": view,
            "dataset": str(embedding_dataset),
        }
        for view in (
            f"{representation_name}_single",
            f"sequence_plus_{representation_name}",
        )
        for family in families
    ]
    if search == "exhaustive":
        jobs.append(
            {
                "name": f"{representation_name}_single_embedding_cnn",
                "family": "embedding_cnn",
                "feature_view": f"{representation_name}_single",
                "dataset": str(embedding_dataset),
            }
        )
    return jobs


def _selection_summary(model_dir: Path) -> tuple[str | None, float | None]:
    path = model_dir / "validation_selection.json"
    if not path.is_file():
        return None, None
    selection = json.loads(path.read_text())
    trials = selection["trials"]
    best = max(float(item["validation_metric"]) for item in trials)
    return str(selection["selected_candidate"]), best


def _completed_record(job: dict[str, str], model_dir: Path) -> dict[str, object]:
    metrics = json.loads((model_dir / "metrics.json").read_text())
    selected, validation = _selection_summary(model_dir)
    return {
        **job,
        "status": "completed",
        "selected_candidate": selected,
        "best_validation_score": validation,
        "test_auroc": metrics.get("auroc"),
        "test_auprc": metrics.get("auprc"),
        "test_accuracy": metrics.get("accuracy"),
    }


def run_suite(
    output: Path,
    dataset: Path,
    embedding_dataset: Path,
    bundle_manifest: Path,
    *,
    device: str = "auto",
    seed: int = 42,
    search: str = "standard",
    include_cnn: bool = True,
    save_models: bool = False,
    representation_name: str = "esmfold",
) -> dict[str, object]:
    """Run or resume all bounded model families and refresh live artifacts."""
    representation_spec(representation_name)
    output.mkdir(parents=True, exist_ok=True)
    readme = output / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Training run artifacts\n\n"
            "`progress.json` and `dashboard.html` are live status artifacts. Each model subdirectory "
            "contains validation selection, held-out predictions, metrics, and (for neural models) "
            "learning curves. Resume the same command to reuse completed model outputs.\n"
        )
    progress_path, dashboard_path = output / "progress.json", output / "dashboard.html"
    jobs = _jobs(
        dataset,
        embedding_dataset,
        search,
        include_cnn=include_cnn,
        representation_name=representation_name,
    )
    started = _central_now()
    records: list[dict[str, object]] = []
    completed_durations: list[float] = []
    pooled_cache: dict[Path, tuple[np.ndarray, tuple[str, ...]]] = {}

    def publish(current: str | None, state: str) -> dict[str, object]:
        completed = sum(record["status"] == "completed" for record in records)
        average = (
            sum(completed_durations) / len(completed_durations) if completed_durations else None
        )
        remaining = len(jobs) - completed
        eta = _central_now() + timedelta(seconds=average * remaining) if average else None
        progress: dict[str, object] = {
            "status": state,
            "started_at_central": _timestamp(started),
            "updated_at_central": _timestamp(),
            "current_model": current,
            "total_models": len(jobs),
            "completed_models": completed,
            "progress_percent": round(100 * completed / len(jobs), 1),
            "average_seconds_per_completed_model": round(average, 1) if average else None,
            "estimated_completion_central": _timestamp(eta) if eta else None,
            "device": device,
            "models": records,
        }
        _write_json(progress_path, progress)
        build_report(output, dashboard_path, progress_path)
        return progress

    for job in jobs:
        model_dir = output / job["name"]
        if (model_dir / "metrics.json").is_file() and (
            model_dir / "validation_selection.json"
        ).is_file():
            records.append(_completed_record(job, model_dir))
            continue
        records.append({**job, "status": "running", "started_at_central": _timestamp()})
        publish(job["name"], "running")
        began = time.monotonic()
        features = None
        try:

            def heartbeat(
                event: dict[str, object],
                began_at: float = began,
                model_name: str = job["name"],
            ) -> None:
                records[-1].update(event)
                records[-1]["elapsed_seconds"] = round(time.monotonic() - began_at, 1)
                publish(model_name, "running")

            dataset_path = Path(job["dataset"])
            pooled_single = None
            if job["feature_view"] != "sequence":
                cache_key = dataset_path.resolve()
                if cache_key not in pooled_cache:
                    pooled_cache[cache_key] = _pooled_single_features(
                        pd.read_parquet(dataset_path), bundle_manifest, representation_name
                    )
                pooled_single = pooled_cache[cache_key]
            features, names = _load_features(
                dataset_path,
                job["feature_view"],
                job["family"],
                bundle_manifest,
                pooled_single,
                representation_name,
            )
            run_benchmark(
                job["dataset"],
                model_dir,
                BenchmarkConfig(  # type: ignore[arg-type]
                    job["family"], seed, device=device, search=search, save_model=save_models
                ),
                features=features,
                feature_names=names,
                progress_callback=heartbeat,
            )
            elapsed = time.monotonic() - began
            records[-1] = {
                **_completed_record(job, model_dir),
                "started_at_central": records[-1]["started_at_central"],
                "completed_at_central": _timestamp(),
                "elapsed_seconds": round(elapsed, 1),
            }
            completed_durations.append(elapsed)
        except Exception as error:
            records[-1].update(
                {
                    "status": "failed",
                    "completed_at_central": _timestamp(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            del features
            gc.collect()
        publish(None, "running")
    failures = [record for record in records if record.get("status") == "failed"]
    if failures:
        publish(None, "failed")
        names = [str(record.get("name")) for record in failures]
        raise RuntimeError(f"training suite failed models: {names}")
    if sum(record.get("status") == "completed" for record in records) != len(jobs):
        publish(None, "failed")
        raise RuntimeError("training suite ended without every requested model artifact")
    return publish(None, "completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--embedding-dataset",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--bundle-manifest",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embedding_manifest.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--search", choices=("standard", "expanded", "exhaustive"), default="standard"
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted checkpoints. Metrics and predictions are always retained.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_suite(
                args.output or Path(f"ml/results/homology35_{args.search}_training"),
                args.dataset,
                args.embedding_dataset,
                args.bundle_manifest,
                device=args.device,
                seed=args.seed,
                search=args.search,
                save_models=args.save_models,
                representation_name=args.representation_name,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
