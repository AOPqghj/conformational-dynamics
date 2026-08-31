"""Generate a persisted leakage-aware split."""

import argparse

from protein_state_router.data.schema import read_catalog
from protein_state_router.data.splitting import make_splits, save_splits

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", required=True)
parser.add_argument("--mode", default="cluster", choices=["random", "cluster", "family", "source"])
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--output", default=None)
parser.add_argument("--holdout-source", default=None)
args = parser.parse_args()
splits = make_splits(read_catalog(args.catalog), args.mode, args.seed, args.holdout_source)
save_splits(splits, args.output or f"data/support/splits/{args.mode}_seed_{args.seed}.parquet")
