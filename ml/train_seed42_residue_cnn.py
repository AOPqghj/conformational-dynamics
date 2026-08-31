"""Train the direct matrix CNN on the frozen Seed-42 catalog only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.train_residue_embedding_models import run_suite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(
            "ml/results/archive/legacy_seed42/frozen_models/seed_42_catalog.parquet"
        ),
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embedding_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/results/archive/legacy_seed42/frozen_seed42_residue_models"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    result = run_suite(
        args.catalog,
        args.embedding_manifest,
        args.output,
        seed=42,
        device=args.device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        model_names=("residue_cnn_expanded",),
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
