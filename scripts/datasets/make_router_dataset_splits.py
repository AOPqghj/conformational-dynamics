"""Create deterministic homology-grouped train/validation/test router splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from protein_state_router.data.splitting import make_grouped_splits

make_splits = make_grouped_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column")
    args = parser.parse_args()
    splits, report = make_grouped_splits(
        pd.read_parquet(args.dataset), args.seed, group_column=args.group_column
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(args.output, index=False)
    args.report_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    args.report_md.write_text(
        "# Router dataset split report\n\n```json\n"
        + json.dumps(report, indent=2, default=str)
        + "\n```\n"
    )


if __name__ == "__main__":
    main()
