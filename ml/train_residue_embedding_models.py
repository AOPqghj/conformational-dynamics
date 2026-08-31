"""Train five compact models over full frozen ESMFold residue-embedding matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from protein_state_router.data.dataset import ResidueMatrixDataset as ResidueDataset
from protein_state_router.data.dataset import residue_matrix_loader as _loader
from protein_state_router.evaluation.metrics import classification_metrics
from protein_state_router.models.probes import (
    AttentionPoolClassifier,
    ResidueEmbeddingCNN,
    SegmentPoolClassifier,
)
from protein_state_router.representations.registry import (
    representation_choices,
    representation_spec,
)
from protein_state_router.training.trainer import resolve_device

try:
    from report import build_report
except ModuleNotFoundError:  # Supports importing this runner from repository-root tests.
    from ml.report import build_report
from torch import nn
from torch.utils.data import DataLoader

CENTRAL = ZoneInfo("America/Chicago")


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    learning_rate: float
    dropout: float
    width: int
    attention_dim: int = 0
    heads: int = 0
    channels: int = 0
    depth: int = 0
    kernel_size: int = 0


MODEL_CANDIDATES: dict[str, tuple[Candidate, ...]] = {
    "attention_quick": (
        Candidate("attention64_hidden128_dropout0.1", 3e-4, 0.1, 128, attention_dim=64, heads=1),
    ),
    "attention_global": (
        Candidate("attention64_hidden128_dropout0.1", 3e-4, 0.1, 128, attention_dim=64, heads=1),
        Candidate("attention128_hidden256_dropout0.2", 1e-3, 0.2, 256, attention_dim=128, heads=1),
        Candidate("attention256_hidden256_dropout0.3", 3e-4, 0.3, 256, attention_dim=256, heads=1),
    ),
    "attention_multihead": (
        Candidate(
            "heads4_attention64_hidden128_dropout0.1", 3e-4, 0.1, 128, attention_dim=64, heads=4
        ),
        Candidate(
            "heads4_attention128_hidden256_dropout0.2", 1e-3, 0.2, 256, attention_dim=128, heads=4
        ),
        Candidate(
            "heads4_attention128_hidden256_dropout0.3", 3e-4, 0.3, 256, attention_dim=128, heads=4
        ),
    ),
    "attention_multihead_expanded": (
        Candidate(
            "heads4_attention128_hidden384_dropout0.05", 1e-3, 0.05, 384, attention_dim=128, heads=4
        ),
        Candidate(
            "heads8_attention128_hidden256_dropout0.15", 3e-4, 0.15, 256, attention_dim=128, heads=8
        ),
        Candidate(
            "heads8_attention256_hidden384_dropout0.2", 3e-4, 0.2, 384, attention_dim=256, heads=8
        ),
    ),
    "attention_hybrid": (
        Candidate(
            "heads1_attention128_hidden256_dropout0.1", 1e-3, 0.1, 256, attention_dim=128, heads=1
        ),
        Candidate(
            "heads4_attention128_hidden256_dropout0.15", 3e-4, 0.15, 256, attention_dim=128, heads=4
        ),
        Candidate(
            "heads4_attention256_hidden384_dropout0.25", 3e-4, 0.25, 384, attention_dim=256, heads=4
        ),
    ),
    "segment3": (
        Candidate("segments3_hidden128_dropout0.1", 3e-4, 0.1, 128),
        Candidate("segments3_hidden256_dropout0.2", 1e-3, 0.2, 256),
        Candidate("segments3_hidden384_dropout0.3", 3e-4, 0.3, 384),
    ),
    "segment5": (
        Candidate("segments5_hidden128_dropout0.1", 3e-4, 0.1, 128),
        Candidate("segments5_hidden256_dropout0.2", 1e-3, 0.2, 256),
        Candidate("segments5_hidden384_dropout0.3", 3e-4, 0.3, 384),
    ),
    "residue_cnn": (
        Candidate(
            "channels64_depth2_kernel5_dropout0.1",
            3e-4,
            0.1,
            0,
            channels=64,
            depth=2,
            kernel_size=5,
        ),
        Candidate(
            "channels96_depth3_kernel5_dropout0.2",
            1e-3,
            0.2,
            0,
            channels=96,
            depth=3,
            kernel_size=5,
        ),
        Candidate(
            "channels128_depth3_kernel7_dropout0.3",
            3e-4,
            0.3,
            0,
            channels=128,
            depth=3,
            kernel_size=7,
        ),
    ),
    "residue_cnn_expanded": (
        Candidate(
            "channels64_depth4_kernel3_dropout0.1",
            1e-3,
            0.1,
            0,
            channels=64,
            depth=4,
            kernel_size=3,
        ),
        Candidate(
            "channels96_depth4_kernel5_dropout0.1",
            3e-4,
            0.1,
            0,
            channels=96,
            depth=4,
            kernel_size=5,
        ),
        Candidate(
            "channels128_depth4_kernel5_dropout0.2",
            3e-4,
            0.2,
            0,
            channels=128,
            depth=4,
            kernel_size=5,
        ),
    ),
    "residue_cnn_large": (
        Candidate("channels192_depth5_kernel7_dropout0.15", 3e-4, 0.15, 0, channels=192, depth=5, kernel_size=7),
        Candidate("channels256_depth4_kernel7_dropout0.2", 3e-4, 0.2, 0, channels=256, depth=4, kernel_size=7),
        Candidate("channels192_depth6_kernel5_dropout0.2", 1e-3, 0.2, 0, channels=192, depth=6, kernel_size=5),
    ),
}


def _model(name: str, candidate: Candidate, embedding_dim: int = 1024) -> nn.Module:
    if name.startswith("attention"):
        return AttentionPoolClassifier(
            embedding_dim=embedding_dim,
            attention_dim=candidate.attention_dim,
            heads=candidate.heads,
            hidden_dim=candidate.width,
            dropout=candidate.dropout,
            include_global_stats=name == "attention_hybrid",
        )
    if name == "segment3":
        return SegmentPoolClassifier(
            3,
            embedding_dim=embedding_dim,
            include_segment_std=True,
            hidden_dim=candidate.width,
            dropout=candidate.dropout,
        )
    if name == "segment5":
        return SegmentPoolClassifier(
            5,
            embedding_dim=embedding_dim,
            include_segment_std=False,
            hidden_dim=candidate.width,
            dropout=candidate.dropout,
        )
    if name.startswith("residue_cnn"):
        return ResidueEmbeddingCNN(
            embedding_dim=embedding_dim,
            channels=candidate.channels,
            depth=candidate.depth,
            kernel_size=candidate.kernel_size,
            dropout=candidate.dropout,
        )
    raise ValueError(f"unknown model name: {name}")


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    identifiers: list[str] = []
    labels, probabilities = [], []
    with torch.no_grad():
        for names, batch_labels, values, mask in loader:
            probability = torch.sigmoid(model(values.to(device), mask.to(device))).cpu().numpy()
            identifiers.extend(names)
            labels.extend(batch_labels.numpy())
            probabilities.extend(probability)
    return identifiers, np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def _fit(
    model: nn.Module,
    train: DataLoader,
    validation: DataLoader,
    device: torch.device,
    learning_rate: float,
    *,
    max_epochs: int,
    patience: int,
    selection_metric: str,
    heartbeat: Callable[[dict[str, float]], None] | None = None,
) -> tuple[nn.Module, list[dict[str, float]], int]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_score, stale, best_epoch, best_state = -float("inf"), 0, 0, None
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for _, labels, values, mask in train:
            optimizer.zero_grad()
            logits = model(values.to(device), mask.to(device))
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, validation_labels, probability = _predict(model, validation, device)
        validation_metrics = classification_metrics(validation_labels, probability)
        score = float(validation_metrics[selection_metric])
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_auroc": float(validation_metrics["auroc"]),
            "validation_auprc": float(validation_metrics["auprc"]),
        }
        history.append(record)
        if heartbeat is not None:
            heartbeat(record)
        if score > best_score:
            best_score, stale, best_epoch = score, 0, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history, best_epoch


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


def run_suite(
    catalog_path: Path,
    embedding_manifest: Path,
    output: Path,
    *,
    seed: int = 42,
    device: str = "auto",
    batch_size: int = 8,
    max_epochs: int = 30,
    patience: int = 5,
    model_names: tuple[str, ...] | None = None,
    representation_name: str = "esmfold",
    embedding_width: int = 1024,
    selection_metric: str = "auroc",
) -> dict[str, object]:
    """Train, validate, refit, and test the five residue-level models."""
    representation_spec(representation_name, embedding_width)
    catalog = pd.read_parquet(catalog_path)
    manifest = pd.read_csv(embedding_manifest)
    required = {"protein_id", "sequence", "sequence_length", "dataset_label", "split"}
    if required - set(catalog) or catalog.protein_id.duplicated().any() or catalog.empty:
        raise ValueError("expected a non-empty unique frozen catalog")
    paths = manifest.set_index("protein_id").embedding_path
    if paths.index.duplicated().any() or not all(Path(path).exists() for path in paths):
        raise ValueError("embedding manifest has missing or duplicate paths")
    if selection_metric not in {"accuracy", "auroc", "auprc"}:
        raise ValueError(f"unsupported selection metric: {selection_metric}")
    selected_models = tuple(model_names or MODEL_CANDIDATES)
    unknown = set(selected_models) - set(MODEL_CANDIDATES)
    if not selected_models or unknown:
        raise ValueError(f"unknown or empty model selection: {sorted(unknown)}")
    target = resolve_device(device)
    if target.type == "cpu":
        torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(
        f"# Residue-level {representation_name} models\n\nThis run streams frozen variable-length residue embeddings and selects each architecture by validation {selection_metric}.\n"
    )
    splits = {
        name: ResidueDataset(catalog.loc[catalog.split.eq(name)], paths, width=embedding_width)
        for name in ("train", "val", "test")
    }
    filtered = pd.read_csv(
        "data/lifecycle/final/initial_8598_dataset/analysis/cross_split_similarity.csv"
    )
    filtered_ids = set(
        filtered.loc[
            (filtered.split == "test") & (filtered.nearest_train_kmer3_cosine < 0.70), "protein_id"
        ]
    )
    started, records, durations = datetime.now(CENTRAL), [], []

    def publish(current: str | None, status: str) -> None:
        completed = sum(record.get("status") == "completed" for record in records)
        average = float(np.mean(durations)) if durations else None
        eta = (
            datetime.now(CENTRAL) + timedelta(seconds=average * (len(selected_models) - completed))
            if average
            else None
        )
        progress = {
            "status": status,
            "started_at_central": started.strftime("%Y-%m-%d %H:%M:%S %Z (Central Time)"),
            "updated_at_central": _now(),
            "current_model": current,
            "total_models": len(selected_models),
            "completed_models": completed,
            "progress_percent": round(100 * completed / len(selected_models), 1),
            "average_seconds_per_completed_model": round(average, 1) if average else None,
            "estimated_completion_central": eta.strftime("%Y-%m-%d %H:%M:%S %Z (Central Time)")
            if eta
            else None,
            "device": str(target),
            "models": records,
        }
        _write_json(output / "progress.json", progress)
        build_report(output, output / "dashboard.html")

    for model_name in selected_models:
        candidates = MODEL_CANDIDATES[model_name]
        destination = output / model_name
        if (destination / "metrics.json").is_file() and (
            destination / "validation_selection.json"
        ).is_file():
            metrics = json.loads((destination / "metrics.json").read_text())
            records.append(
                {
                    "name": model_name,
                    "status": "completed",
                    "test_accuracy": metrics["accuracy"],
                    "test_auroc": metrics["auroc"],
                    "test_auprc": metrics["auprc"],
                }
            )
            continue
        began = time.monotonic()
        records.append({"name": model_name, "status": "running", "started_at_central": _now()})
        publish(model_name, "running")
        trials, curves, selected = [], [], None
        for index, candidate in enumerate(candidates):
            torch.manual_seed(seed)
            train = _loader(splits["train"], batch_size, True, seed + index)
            validation = _loader(splits["val"], batch_size, False, seed)

            def heartbeat(
                record: dict[str, float],
                trial: Candidate = candidate,
                began_at: float = began,
                current_model: str = model_name,
            ) -> None:
                records[-1].update(
                    {
                        "candidate": trial.name,
                        **record,
                        "elapsed_seconds": round(time.monotonic() - began_at, 1),
                    }
                )
                publish(current_model, "running")

            fitted, history, best_epoch = _fit(
                _model(model_name, candidate, embedding_width),
                train,
                validation,
                target,
                candidate.learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                selection_metric=selection_metric,
                heartbeat=heartbeat,
            )
            _, labels, probability = _predict(fitted, validation, target)
            metric = float(classification_metrics(labels, probability)[selection_metric])
            trials.append(
                {
                    "candidate": candidate.name,
                    "validation_metric": metric,
                    "best_epoch": best_epoch,
                    **asdict(candidate),
                }
            )
            curves.extend({"candidate": candidate.name, **point} for point in history)
            if selected is None or metric > selected[0]:
                selected = (metric, candidate, best_epoch)
        assert selected is not None
        _, candidate, best_epoch = selected
        torch.manual_seed(seed)
        combined = pd.concat(
            (catalog.loc[catalog.split.eq("train")], catalog.loc[catalog.split.eq("val")]),
            ignore_index=True,
        )
        final_model = _model(model_name, candidate, embedding_width)
        final_loader = _loader(
            ResidueDataset(combined, paths, width=embedding_width), batch_size, True, seed
        )
        final_model.to(target)
        optimizer, loss_fn = (
            torch.optim.AdamW(
                final_model.parameters(), lr=candidate.learning_rate, weight_decay=1e-4
            ),
            nn.BCEWithLogitsLoss(),
        )
        for _ in range(best_epoch):
            final_model.train()
            for _, labels, values, mask in final_loader:
                optimizer.zero_grad()
                loss = loss_fn(final_model(values.to(target), mask.to(target)), labels.to(target))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
        identifiers, labels, probability = _predict(
            final_model, _loader(splits["test"], batch_size, False, seed), target
        )
        predictions = pd.DataFrame(
            {"protein_id": identifiers, "dataset_label": labels, "probability": probability}
        )
        metrics = classification_metrics(labels, probability)
        subset = predictions.loc[predictions.protein_id.isin(filtered_ids)]
        filtered_metrics = classification_metrics(
            subset.dataset_label.to_numpy(), subset.probability.to_numpy()
        )
        destination.mkdir(exist_ok=True)
        predictions.to_parquet(destination / "test_predictions.parquet", index=False)
        pd.DataFrame(curves).to_csv(destination / "learning_curves.csv", index=False)
        _write_json(destination / "metrics.json", metrics)
        _write_json(destination / "filtered_test_metrics.json", filtered_metrics)
        _write_json(
            destination / "validation_selection.json",
            {
                "primary_metric": selection_metric,
                "selected_candidate": candidate.name,
                "trials": trials,
            },
        )
        _write_json(
            destination / "manifest.json",
            {
                "seed": seed,
                "candidate": candidate.name,
                "best_epoch": best_epoch,
                "model": model_name,
                "input": f"masked full residue x {embedding_width} {representation_name} matrix",
                "embedding_width": embedding_width,
                "embedding_representation": representation_name,
                "representation": "full_matrix_cnn_encode",
                "representation_dim": candidate.channels * 2,
                "split_counts": {
                    split: int(catalog.split.eq(split).sum()) for split in ("train", "val", "test")
                },
            },
        )
        torch.save(
            {
                "model": model_name,
                "candidate": asdict(candidate),
                "state_dict": final_model.cpu().state_dict(),
            },
            destination / "model.pt",
        )
        elapsed = time.monotonic() - began
        durations.append(elapsed)
        records[-1] = {
            "name": model_name,
            "status": "completed",
            "selected_candidate": candidate.name,
            "best_validation_score": selected[0],
            "test_accuracy": metrics["accuracy"],
            "test_auroc": metrics["auroc"],
            "test_auprc": metrics["auprc"],
            "elapsed_seconds": round(elapsed, 1),
            "completed_at_central": _now(),
        }
        publish(None, "running")
    registry = {f"{name}/model.pt": _sha256(output / name / "model.pt") for name in selected_models}
    (output / "frozen_residue_model_registry.json").write_text(
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
        "--output", type=Path, default=Path("ml/results/initial_8598_residue_embedding_models")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    parser.add_argument("--embedding-width", type=int, default=1024)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "auroc", "auprc"),
        default="auroc",
        help="Validation metric used for early stopping and candidate selection.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CANDIDATES),
        default=tuple(MODEL_CANDIDATES),
        help="model families to train; use residue_cnn residue_cnn_expanded for the matrix CNN run",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".run.lock"
    progress_path = args.output / "progress.json"
    if lock.exists() and progress_path.is_file():
        if json.loads(progress_path.read_text()).get("status") == "completed":
            lock.unlink()
    created = False
    try:
        with lock.open("x"):
            created = True
            print(
                json.dumps(
                    run_suite(
                        args.catalog,
                        args.embedding_manifest,
                        args.output,
                        seed=args.seed,
                        device=args.device,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        model_names=tuple(args.models),
                        representation_name=args.representation_name,
                        embedding_width=args.embedding_width,
                        selection_metric=args.selection_metric,
                    ),
                    indent=2,
                )
            )
    except FileExistsError as error:
        raise RuntimeError(f"another residue-embedding run holds {lock}") from error
    except Exception as error:
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text())
            progress["status"] = "failed"
            progress["updated_at_central"] = _now()
            for record in reversed(progress.get("models", [])):
                if record.get("status") == "running":
                    record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                    break
            _write_json(progress_path, progress)
            build_report(args.output, args.output / "dashboard.html")
        raise
    finally:
        if created:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
