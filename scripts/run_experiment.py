"""Canonical preflight, execution, status, and comparison interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_state_router.experiments.control import (
    assert_comparable,
    build_plan,
    process_is_live,
    run_verified_plan,
    write_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--skip-file-check", action="store_true")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--confirm", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("contracts", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "plan":
        plan = build_plan(
            ROOT,
            (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config,
            verify_embedding_files=not args.skip_file_check,
        )
        write_plan(args.output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.command == "run":
        run_verified_plan(ROOT, args.plan, args.confirm)
    elif args.command == "compare":
        contracts = [json.loads(path.read_text()) for path in args.contracts]
        assert_comparable(contracts)
        print(json.dumps({"status": "comparable", "runs": len(contracts)}))
    else:
        root = args.output_root
        progress_path = root / "progress.json"
        lock_path = root / ".run.lock"
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        lock = json.loads(lock_path.read_text()) if lock_path.is_file() else {}
        reported = str(progress.get("status", "unknown"))
        live = process_is_live(lock) if lock else False
        effective = "stale" if reported == "running" and not live else reported
        print(json.dumps({"reported_status": reported, "effective_status": effective, "process_live": live, "progress": progress}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
