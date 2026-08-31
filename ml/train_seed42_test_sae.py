"""Train a resumable TopK SAE on seed-42 train-protein residue embeddings.

The default fit partition is seed-42 train, preserving validation and test proteins
for downstream SAE selection and analysis.
"""
# ruff: noqa: E402 - script execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interpretability.contracts import load_residue_matrix
from interpretability.methods import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    balanced_residue_sample,
)
from protein_state_router.representations.registry import (
    representation_choices,
    representation_spec,
)
from protein_state_router.training.trainer import resolve_device


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int = 42
    input_dim: int = 1024
    expansion_factor: int = 4
    latent_dim_override: int | None = None
    top_k: int = 64
    residues_per_protein: int = 64
    batch_size: int = 1024
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    fit_partition: str = "train"
    representation_name: str = "esmfold"

    @property
    def latent_dim(self) -> int:
        return self.latent_dim_override or self.input_dim * self.expansion_factor

    def __post_init__(self) -> None:
        if self.fit_partition not in {"train", "val", "test"}:
            raise ValueError("fit_partition must be train, val, or test")
        representation_spec(self.representation_name, self.input_dim)
        if self.latent_dim_override is not None and self.latent_dim_override < 1:
            raise ValueError("latent_dim_override must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _status(event: str, **details: object) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def _checkpoint(
    path: Path,
    *,
    model: SparseAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: RunConfig,
    center: np.ndarray,
    catalog_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "center": center,
            "catalog_sha256": catalog_sha256,
            "torch_version": torch.__version__,
        },
        temporary,
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_fit_residues(catalog: pd.DataFrame, config: RunConfig) -> tuple[np.ndarray, pd.DataFrame]:
    fit = catalog.loc[catalog.split.eq(config.fit_partition)].copy()
    if fit.empty:
        raise ValueError(f"seed-42 {config.fit_partition} partition is empty")
    matrices = []
    for index, row in enumerate(fit.itertuples(index=False), start=1):
        matrices.append(
            load_residue_matrix(
                Path(row.embedding_path),
                protein_id=row.protein_id,
                sequence=row.sequence,
                sequence_sha256=row.sequence_sha256,
                sequence_length=int(row.sequence_length),
                expected_width=config.input_dim,
            )
        )
        if index == 1 or index % 100 == 0 or index == len(fit):
            _status("loading_fit_embeddings", completed=index, total=len(fit))
    values, owners = balanced_residue_sample(
        matrices, residues_per_protein=config.residues_per_protein, random_seed=config.seed
    )
    counts = np.bincount(owners, minlength=len(fit))
    sample = fit[["protein_id", "sequence_sha256", "sequence_length"]].copy()
    sample["sampled_residues"] = counts
    return values, sample


def train(
    catalog_path: Path,
    output_root: Path,
    config: RunConfig,
    *,
    resume: bool = True,
) -> dict[str, object]:
    """Fit one fixed TopK SAE and persist portable, resumable artifacts."""
    catalog = pd.read_parquet(catalog_path)
    required = {
        "protein_id",
        "split",
        "embedding_path",
        "sequence",
        "sequence_sha256",
        "sequence_length",
    }
    if catalog.empty or catalog.protein_id.duplicated().any() or required - set(catalog):
        raise ValueError("expected a non-empty unique seed-42 catalog")
    catalog_sha256 = _sha256_file(catalog_path)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config.json"
    expected_config = asdict(config)
    if config_path.exists() and json.loads(config_path.read_text()) != expected_config:
        raise ValueError("existing SAE output has a different configuration")
    _write_json(config_path, expected_config)
    _seed_everything(config.seed)
    target = resolve_device(config.device)
    _status("started", device=str(target), catalog=str(catalog_path), catalog_sha256=catalog_sha256)
    values, sample = _load_fit_residues(catalog, config)
    center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = values - center
    sample.to_parquet(output_root / "sampled_fit_proteins.parquet", index=False)
    np.save(output_root / "input_center.npy", center)
    _status(
        "sample_ready", proteins=len(sample), residues=len(centered), input_dim=centered.shape[1]
    )
    model = SparseAutoencoder(
        SparseAutoencoderConfig(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            l1_coefficient=0.0,
            top_k=config.top_k,
        )
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    checkpoint_path = output_root / "latest.pt"
    start_epoch = 1
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint["config"] != expected_config
            or checkpoint["catalog_sha256"] != catalog_sha256
        ):
            raise ValueError("checkpoint does not match this SAE configuration or catalog")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _status("resumed", completed_epochs=start_epoch - 1)
    dataset = TensorDataset(torch.from_numpy(centered))
    history: list[dict[str, float]] = []
    history_path = output_root / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())["epochs"]
    for epoch in range(start_epoch, config.epochs + 1):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, generator=generator
        )
        total_loss = total_reconstruction = total_active = total_items = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(target)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_decoder_()
            items = float(len(batch))
            total_items += items
            total_loss += float(loss.total.detach().cpu()) * items
            total_reconstruction += float(loss.reconstruction.detach().cpu()) * items
            total_active += float((model.encode(batch) > 0).sum().detach().cpu())
        record = {
            "epoch": float(epoch),
            "loss": total_loss / total_items,
            "reconstruction_mse": total_reconstruction / total_items,
            "mean_active_features": total_active / total_items,
        }
        history = [row for row in history if int(row["epoch"]) != epoch] + [record]
        _write_json(history_path, {"epochs": history})
        _checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            center=center,
            catalog_sha256=catalog_sha256,
        )
        _write_json(
            output_root / "progress.json",
            {
                "status": "completed" if epoch == config.epochs else "running",
                "epoch": epoch,
                "epochs": config.epochs,
                "device": str(target),
                "residues": len(centered),
                **record,
            },
        )
        _status("epoch_completed", **record, device=str(target))
    model.eval()
    with torch.no_grad():
        probe_rng = np.random.default_rng(config.seed + 100_003)
        probe_indices = np.sort(
            probe_rng.choice(len(centered), size=min(len(centered), 8192), replace=False)
        )
        probe = torch.from_numpy(centered[probe_indices]).to(target)
        _, latents = model(probe)
        feature_density = (latents > 0).float().mean(dim=0).cpu().numpy()
    metrics = {
        "fit_partition": f"seed_42_{config.fit_partition}",
        "unseen_partitions": [
            name for name in ("train", "val", "test") if name != config.fit_partition
        ],
        "proteins": len(sample),
        "sampled_residues": len(centered),
        "input_dim": config.input_dim,
        "latent_dim": config.latent_dim,
        "top_k": config.top_k,
        "dead_feature_fraction_on_probe": float((feature_density == 0).mean()),
        "dead_feature_probe_sampling": "seeded_uniform_without_replacement",
        "dead_feature_probe_residues": len(probe_indices),
        "final_epoch": history[-1],
        "device": str(target),
        "catalog_sha256": catalog_sha256,
    }
    _write_json(output_root / "metrics.json", metrics)
    _status("completed", **metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("ml/results/homology35_sae_seed42_topk64")
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--residues-per-protein", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--expansion-factor", type=int, default=4)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--representation-name", choices=representation_choices(), default="esmfold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-partition", choices=("train", "val", "test"), default="train")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = RunConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        residues_per_protein=args.residues_per_protein,
        top_k=args.top_k,
        expansion_factor=args.expansion_factor,
        latent_dim_override=args.latent_dim,
        seed=args.seed,
        fit_partition=args.fit_partition,
        input_dim=args.input_dim,
        representation_name=args.representation_name,
    )
    print(
        json.dumps(
            train(args.catalog, args.output_root, config, resume=not args.no_resume), indent=2
        )
    )


if __name__ == "__main__":
    main()
