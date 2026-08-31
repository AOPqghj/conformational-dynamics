"""Select classical probes on learned residue-level representations using validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch
from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.training.trainer import resolve_device
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train_residue_embedding_models import Candidate, ResidueDataset, _loader, _model

try:
    from report import build_report
except ModuleNotFoundError:
    from ml.report import build_report


CENTRAL = ZoneInfo("America/Chicago")
CNN_EXTRACTORS = ("residue_cnn", "residue_cnn_expanded")


def _now() -> str:
    return datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M:%S %Z (Central Time)")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representations(
    model: torch.nn.Module, loader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval().to(device)
    identifiers, labels, rows = [], [], []
    with torch.no_grad():
        for names, batch_labels, values, mask in loader:
            rows.append(model.encode(values.to(device), mask.to(device)).cpu().numpy())
            identifiers.extend(names)
            labels.extend(batch_labels.numpy())
    return identifiers, np.asarray(labels, dtype=int), np.vstack(rows).astype(np.float32)


def _classifiers(seed: int) -> dict[str, list[tuple[str, object]]]:
    return {
        "logistic": [
            (
                f"l2_C{c}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression(C=c, max_iter=2000, random_state=seed)),
                    ]
                ),
            )
            for c in (0.01, 0.1, 1.0)
        ]
        + [
            (
                f"l1_C{c}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c,
                                solver="saga",
                                l1_ratio=1.0,
                                max_iter=2000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
            for c in (0.01, 0.1, 1.0)
        ],
        "extra_trees": [
            (
                f"trees600_leaf{leaf}_features{features}",
                ExtraTreesClassifier(
                    n_estimators=600,
                    min_samples_leaf=leaf,
                    max_features=features,
                    class_weight="balanced",
                    n_jobs=2,
                    random_state=seed,
                ),
            )
            for leaf in (1, 5, 10)
            for features in (0.3, "sqrt")
        ],
    }


def run_suite(
    catalog_path: Path,
    embedding_manifest: Path,
    representations_root: Path,
    output: Path,
    *,
    seed: int = 42,
    device: str = "auto",
    batch_size: int = 16,
    extractor_names: tuple[str, ...] = CNN_EXTRACTORS,
    selection_metric: str = "auroc",
    embedding_width: int = 1024,
) -> dict[str, object]:
    """Select Logistic Regression and ExtraTrees heads without test-set tuning."""
    catalog = pd.read_parquet(catalog_path)
    paths = pd.read_csv(embedding_manifest).set_index("protein_id").embedding_path
    if catalog.empty or not all(Path(path).exists() for path in paths):
        raise ValueError("the catalog and all embedding paths are required")
    progress = json.loads((representations_root / "progress.json").read_text())
    if progress.get("status") != "completed":
        raise RuntimeError("residue-level models must complete before stacking")
    for extractor_name in extractor_names:
        if not (representations_root / extractor_name / "model.pt").is_file():
            raise RuntimeError(f"missing completed CNN extractor: {extractor_name}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(
        "# Learned-representation classical probes\n\nEach fixed head consumes a frozen representation from a completed residue-level model.\n"
    )
    target = resolve_device(device)
    if target.type == "cpu":
        torch.set_num_threads(2)
    datasets = {
        name: ResidueDataset(
            catalog.loc[catalog.split.eq(name)], paths, width=embedding_width
        )
        for name in ("train", "val", "test")
    }
    filter_frame = pd.read_csv(
        "data/lifecycle/final/initial_8598_dataset/analysis/cross_split_similarity.csv"
    )
    filtered_ids = set(
        filter_frame.loc[
            (filter_frame.split == "test") & (filter_frame.nearest_train_kmer3_cosine < 0.70),
            "protein_id",
        ]
    ) if {"split", "nearest_train_kmer3_cosine"} <= set(filter_frame) else set()
    records = []

    def publish(current: str | None, status: str) -> None:
        completed = sum(record.get("status") == "completed" for record in records)
        progress = {
            "status": status,
            "started_at_central": _now(),
            "updated_at_central": _now(),
            "current_model": current,
            "total_models": len(extractor_names) * 2,
            "completed_models": completed,
            "progress_percent": round(100 * completed / (len(extractor_names) * 2), 1),
            "device": str(target),
            "models": records,
        }
        _write_json(output / "progress.json", progress)
        build_report(output, output / "dashboard.html")

    for extractor_name in extractor_names:
        source = representations_root / extractor_name
        payload = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        extractor = _model(
            str(payload["model"]), Candidate(**payload["candidate"]), embedding_width
        )
        extractor.load_state_dict(payload["state_dict"])
        _, train_labels, train_features = _representations(
            extractor, _loader(datasets["train"], batch_size, False, seed), target
        )
        _, validation_labels, validation_features = _representations(
            extractor, _loader(datasets["val"], batch_size, False, seed), target
        )
        identifiers, test_labels, test_features = _representations(
            extractor, _loader(datasets["test"], batch_size, False, seed), target
        )
        combined_features = np.vstack((train_features, validation_features))
        combined_labels = np.concatenate((train_labels, validation_labels))
        for head_name, candidates in _classifiers(seed).items():
            name, began = f"{extractor_name}_{head_name}", time.monotonic()
            destination = output / name
            if (destination / "metrics.json").is_file() and (
                destination / "classifier.joblib"
            ).is_file():
                metrics = json.loads((destination / "metrics.json").read_text())
                records.append(
                    {
                        "name": name,
                        "status": "completed",
                        "selected_candidate": "predeclared",
                        "test_accuracy": metrics["accuracy"],
                        "test_auroc": metrics["auroc"],
                        "test_auprc": metrics["auprc"],
                    }
                )
                continue
            records.append({"name": name, "status": "running", "started_at_central": _now()})
            publish(name, "running")
            trials = []
            selected: tuple[float, str, object] | None = None
            for candidate_name, candidate in candidates:
                classifier = clone(candidate)
                classifier.fit(train_features, train_labels)
                validation_probability = classifier.predict_proba(validation_features)[:, 1]
                validation_metrics = classification_metrics(
                    validation_labels, validation_probability
                )
                trials.append(
                    {"candidate": candidate_name, "validation_metric": validation_metrics[selection_metric]}
                )
                if selected is None or validation_metrics[selection_metric] > selected[0]:
                    selected = (validation_metrics[selection_metric], candidate_name, candidate)
            assert selected is not None
            _, selected_name, classifier = selected
            classifier = clone(classifier)
            classifier.fit(combined_features, combined_labels)
            probability = classifier.predict_proba(test_features)[:, 1]
            predictions = pd.DataFrame(
                {
                    "protein_id": identifiers,
                    "dataset_label": test_labels,
                    "probability": probability,
                }
            )
            metrics = classification_metrics(test_labels, probability)
            filtered = predictions.loc[predictions.protein_id.isin(filtered_ids)]
            destination.mkdir(exist_ok=True)
            predictions.to_parquet(destination / "test_predictions.parquet", index=False)
            _write_json(destination / "metrics.json", metrics)
            _write_json(
                destination / "filtered_test_metrics.json",
                classification_metrics(
                    filtered.dataset_label.to_numpy(), filtered.probability.to_numpy()
                ),
            )
            _write_json(
                destination / "validation_selection.json",
                {
                    "primary_metric": selection_metric,
                    "selected_candidate": selected_name,
                    "trials": trials,
                },
            )
            _write_json(
                destination / "manifest.json",
                {
                    "candidate": "predeclared",
                    "extractor": extractor_name,
                    "head": head_name,
                    "seed": seed,
                    "selection": "validation-selected; test evaluated once",
                    "input": "full residue matrix through frozen CNN encoder",
                    "feature_dimension": int(train_features.shape[1]),
                    "extractor_checkpoint": str(
                        (representations_root / extractor_name / "model.pt").resolve()
                    ),
                    "split_counts": {
                        split: int(catalog.split.eq(split).sum())
                        for split in ("train", "val", "test")
                    },
                },
            )
            joblib.dump(classifier, destination / "classifier.joblib")
            records[-1] = {
                "name": name,
                "status": "completed",
                "selected_candidate": selected_name,
                "test_accuracy": metrics["accuracy"],
                "test_auroc": metrics["auroc"],
                "test_auprc": metrics["auprc"],
                "elapsed_seconds": round(time.monotonic() - began, 1),
                "completed_at_central": _now(),
            }
            publish(None, "running")
    registry = {
        f"{extractor_name}_{head_name}/classifier.joblib": _sha256(
            output / f"{extractor_name}_{head_name}" / "classifier.joblib"
        )
        for extractor_name in extractor_names
        for head_name in _classifiers(seed)
    }
    (output / "frozen_residue_stacker_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    publish(None, "completed")
    return json.loads((output / "progress.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/catalog.parquet"),
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embedding_manifest.csv"),
    )
    parser.add_argument(
        "--representations-root",
        type=Path,
        default=Path("ml/results/initial_8598_residue_embedding_models"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("ml/results/initial_8598_residue_embedding_stackers")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extractors", nargs="+", choices=("residue_cnn", "residue_cnn_expanded"), default=list(CNN_EXTRACTORS))
    parser.add_argument("--selection-metric", choices=("accuracy", "auroc", "auprc"), default="auroc")
    parser.add_argument("--embedding-width", type=int, default=1024)
    args = parser.parse_args()
    print(
        json.dumps(
            run_suite(
                args.catalog,
                args.embedding_manifest,
                args.representations_root,
                args.output,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                extractor_names=tuple(args.extractors),
                selection_metric=args.selection_metric,
                embedding_width=args.embedding_width,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
