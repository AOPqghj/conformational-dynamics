"""Freeze a completed seed-42 matrix SAE into a portable inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_latent_dim(config: dict[str, Any]) -> int:
    """Resolve an SAE latent width, honoring an explicit dimensional override."""
    override = config.get("latent_dim_override")
    latent_dim = (
        int(override)
        if override is not None
        else int(config["input_dim"]) * int(config["expansion_factor"])
    )
    if latent_dim < 1:
        raise ValueError("SAE latent dimension must be positive")
    return latent_dim


def freeze(source: Path, destination: Path) -> dict[str, object]:
    """Create an inference-only, hash-pinned frozen SAE bundle."""
    required = {
        "config.json",
        "input_center.npy",
        "latest.pt",
        "metrics.json",
        "heldout/heldout_reconstruction_metrics.json",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"completed SAE is missing: {missing}")
    config = json.loads((source / "config.json").read_text())
    metrics = json.loads((source / "metrics.json").read_text())
    heldout = json.loads((source / "heldout/heldout_reconstruction_metrics.json").read_text())
    if config.get("fit_partition") != "train" or metrics.get("fit_partition") != "seed_42_train":
        raise ValueError("only the completed seed-42 train-fit SAE may be frozen")
    if set(heldout.get("partitions", ())) != {"val", "test"}:
        raise ValueError("frozen SAE requires held-out validation and test metrics")
    checkpoint = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("config") != config:
        raise ValueError("checkpoint configuration differs from run configuration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="freeze-sae-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        torch.save(
            {
                "state_dict": checkpoint["state_dict"],
                "config": config,
                "center_artifact": "input_center.npy",
                "catalog_sha256": checkpoint["catalog_sha256"],
                "epoch": checkpoint["epoch"],
                "torch_version": checkpoint["torch_version"],
            },
            staging / "model.pt",
        )
        shutil.copy2(source / "input_center.npy", staging / "input_center.npy")
        (staging / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (staging / "heldout_metrics.json").write_text(
            json.dumps(heldout, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "name": f"{config.get('representation_name', 'esmfold')}_matrix_topk{config['top_k']}_seed{config.get('seed', 42)}",
            "embedding_representation": config.get("representation_name", "esmfold"),
            "kind": "full_matrix_residue_sae",
            "fit_partition": "seed_42_train",
            "unseen_partitions": ["val", "test"],
            "artifacts": {
                name: sha256_file(staging / name) for name in ("model.pt", "input_center.npy")
            },
            "architecture": {
                "input_dim": config["input_dim"],
                "latent_dim": configured_latent_dim(config),
                "activation": "TopK-ReLU",
                "top_k": config["top_k"],
                "decoder_unit_norm": True,
            },
            "catalog_sha256": checkpoint["catalog_sha256"],
            "heldout_metrics": "heldout_metrics.json",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            raise FileExistsError(f"frozen SAE destination already exists: {destination}")
        staging.replace(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("ml/results/homology35_sae_seed42_topk64")
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("ml/results/homology35_frozen_saes/esmfold_matrix_topk64_seed42"),
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
