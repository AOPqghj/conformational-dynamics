"""Evaluate a train-fit SAE on held-out partitions and summarize its features."""
# ruff: noqa: E402 - script execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interpretability.contracts import load_residue_matrix
from interpretability.methods import SparseAutoencoder, SparseAutoencoderConfig
from ml.freeze_seed42_sae import configured_latent_dim


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(event: str, **details: object) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_device(device: str) -> torch.device:
    """Resolve the CLI's automatic device choice before constructing torch.device."""
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device)


def _sample_partition(
    catalog: pd.DataFrame,
    partition: str,
    residues_per_protein: int,
    seed: int,
    input_dim: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = catalog.loc[catalog.split.eq(partition)].copy()
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    protein_ids: list[str] = []
    for owner, row in enumerate(rows.itertuples(index=False)):
        matrix = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=input_dim,
        )
        count = min(residues_per_protein, len(matrix))
        selected = np.sort(rng.choice(len(matrix), size=count, replace=False))
        values.append(matrix[selected].astype(np.float32, copy=False))
        owners.append(np.full(count, owner, dtype=np.int64))
        positions.append(selected.astype(np.int64))
        protein_ids.append(row.protein_id)
        if owner == 0 or (owner + 1) % 100 == 0 or owner + 1 == len(rows):
            _status(
                "loading_heldout_embeddings",
                partition=partition,
                completed=owner + 1,
                total=len(rows),
            )
    return (
        np.concatenate(values),
        np.concatenate(owners),
        np.concatenate(positions),
        protein_ids,
    )


def _update_top(
    current_values: np.ndarray,
    current_owners: np.ndarray,
    current_positions: np.ndarray,
    batch_values: torch.Tensor,
    owner_batch: np.ndarray,
    position_batch: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(limit, batch_values.shape[0])
    values, indices = torch.topk(batch_values, k=count, dim=0)
    new_values = values.detach().cpu().numpy()
    new_owners = owner_batch[indices.detach().cpu().numpy()]
    new_positions = position_batch[indices.detach().cpu().numpy()]
    merged_values = np.concatenate((current_values, new_values), axis=0)
    merged_owners = np.concatenate((current_owners, new_owners), axis=0)
    merged_positions = np.concatenate((current_positions, new_positions), axis=0)
    keep = np.argpartition(merged_values, -limit, axis=0)[-limit:]
    columns = np.arange(merged_values.shape[1])[None, :]
    return (
        merged_values[keep, columns],
        merged_owners[keep, columns],
        merged_positions[keep, columns],
    )


def evaluate(
    catalog_path: Path,
    checkpoint_root: Path,
    output_root: Path,
    *,
    device: str = "auto",
    top_features_per_partition: int = 20,
    batch_size: int = 2048,
) -> dict[str, object]:
    catalog = pd.read_parquet(catalog_path)
    checkpoint = torch.load(checkpoint_root / "latest.pt", map_location="cpu", weights_only=False)
    config = json.loads((checkpoint_root / "config.json").read_text())
    catalog_hash = _sha256_file(catalog_path)
    if checkpoint["catalog_sha256"] != catalog_hash:
        raise ValueError("SAE checkpoint catalog hash does not match evaluation catalog")
    target = _resolve_device(device)
    model = SparseAutoencoder(
        SparseAutoencoderConfig(
            input_dim=config["input_dim"],
            latent_dim=configured_latent_dim(config),
            l1_coefficient=0.0,
            top_k=config["top_k"],
        )
    ).to(target)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    center = np.load(checkpoint_root / "input_center.npy")
    output_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for partition in ("val", "test"):
        values, owners, positions, protein_ids = _sample_partition(
            catalog,
            partition,
            config["residues_per_protein"],
            config["seed"],
            config["input_dim"],
        )
        centered = values - center
        feature_count = configured_latent_dim(config)
        sums = np.zeros(feature_count, dtype=np.float64)
        maxima = np.zeros(feature_count, dtype=np.float32)
        active = np.zeros(feature_count, dtype=np.int64)
        top_values = np.full((top_features_per_partition, feature_count), -np.inf, dtype=np.float32)
        top_owners = np.zeros((top_features_per_partition, feature_count), dtype=np.int64)
        top_positions = np.zeros((top_features_per_partition, feature_count), dtype=np.int64)
        squared_error = 0.0
        total = 0
        for start in range(0, len(centered), batch_size):
            stop = min(start + batch_size, len(centered))
            batch = torch.from_numpy(centered[start:stop]).to(target)
            with torch.no_grad():
                reconstruction, latents = model(batch)
            error = reconstruction - batch
            squared_error += float(error.square().sum().cpu())
            total += stop - start
            latent_cpu = latents.detach().cpu()
            latent_values = latent_cpu.numpy()
            sums += latent_values.sum(axis=0)
            maxima = np.maximum(maxima, latent_values.max(axis=0))
            active += (latent_values > 0).sum(axis=0)
            owner_batch = owners[start:stop]
            position_batch = positions[start:stop]
            top_values, top_owners, top_positions = _update_top(
                top_values,
                top_owners,
                top_positions,
                latent_cpu,
                owner_batch,
                position_batch,
                top_features_per_partition,
            )
        variance = float(centered.var())
        mse = squared_error / (total * config["input_dim"])
        metrics[partition] = {
            "proteins": len(protein_ids),
            "sampled_residues": total,
            "reconstruction_mse": mse,
            "explained_variance": 1.0 - mse / variance if variance else None,
            "mean_active_features": float(active.sum() / total),
            "feature_density_mean": float((active / total).mean()),
        }
        for feature in range(feature_count):
            summary_rows.append(
                {
                    "partition": partition,
                    "feature": feature,
                    "mean_activation": float(sums[feature] / total),
                    "max_activation": float(maxima[feature]),
                    "active_residue_count": int(active[feature]),
                    "activation_density": float(active[feature] / total),
                }
            )
            order = np.argsort(top_values[:, feature])[::-1]
            active_top = [index for index in order if top_values[index, feature] > 0]
            for rank, index in enumerate(active_top[:top_features_per_partition], start=1):
                top_rows.append(
                    {
                        "partition": partition,
                        "feature": feature,
                        "rank": rank,
                        "activation": float(top_values[index, feature]),
                        "protein_id": protein_ids[top_owners[index, feature]],
                        "residue_position": int(top_positions[index, feature]),
                    }
                )
        _status("partition_completed", partition=partition, **metrics[partition])
    pd.DataFrame(summary_rows).to_parquet(
        output_root / "feature_activation_summary.parquet", index=False
    )
    pd.DataFrame(top_rows).to_parquet(output_root / "feature_top_activations.parquet", index=False)
    result = {
        "catalog_sha256": catalog_hash,
        "checkpoint": str(checkpoint_root / "latest.pt"),
        "device": str(target),
        "partitions": metrics,
        "top_features_per_partition": top_features_per_partition,
        "feature_summary": "feature_activation_summary.parquet",
        "top_activation_table": "feature_top_activations.parquet",
    }
    _write_json(output_root / "heldout_reconstruction_metrics.json", result)
    _status("completed", **result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("ml/results/homology35_sae_seed42_topk64")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ml/results/homology35_sae_seed42_topk64/heldout"),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--top-features-per-partition", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.catalog,
                args.checkpoint_root,
                args.output_root,
                device=args.device,
                top_features_per_partition=args.top_features_per_partition,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
