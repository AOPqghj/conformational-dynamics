# ruff: noqa: E402 - executable script needs repository root before local imports.

"""Flatten validated BioEmu checkpoint archives without overwriting embeddings.

The Colab worker writes groups of 25 raw ``<sequence_sha256>.npz`` results in
tar archives.  The frozen pooled-model trainer intentionally consumes the same
flat layout as the initial 4,000-result lane, so this importer is deliberately
strict about archive provenance and individual embedding contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from interpretability.contracts import load_residue_matrix

from scripts.bioemu_tpu_combined_worker import QUERY_ONLY_CONFIG

EMBEDDING_WIDTH = 384
REQUIRED_SIDECAR_KEYS = {
    "archive_sha256",
    "checkpoint_index",
    "embedding_width",
    "model_id",
    "model_revision",
    "sequence_sha256",
    "size",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_by_hash(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_parquet(catalog_path)
    required = {"protein_id", "sequence", "sequence_sha256", "sequence_length"}
    missing = required - set(catalog)
    if missing or catalog.sequence_sha256.duplicated().any():
        raise ValueError(f"catalog lacks a unique embedding identity: {sorted(missing)}")
    return catalog.set_index("sequence_sha256", verify_integrity=True)


def _sidecar(path: Path, archive: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint sidecar: {path}") from error
    missing = REQUIRED_SIDECAR_KEYS - set(payload)
    if missing:
        raise ValueError(f"checkpoint sidecar missing keys {sorted(missing)}: {path}")
    hashes = payload["sequence_sha256"]
    if (
        not isinstance(hashes, list)
        or not hashes
        or len(hashes) != len(set(hashes))
        or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
    ):
        raise ValueError(f"checkpoint sidecar has invalid sequence hashes: {path}")
    if payload["embedding_width"] != EMBEDDING_WIDTH:
        raise ValueError(f"unexpected embedding width in {path}")
    if payload["size"] != archive.stat().st_size or payload["archive_sha256"] != sha256(archive):
        raise ValueError(f"checkpoint archive integrity mismatch: {archive}")
    return hashes


def import_archives(
    archive_root: Path,
    catalog_path: Path,
    output_root: Path,
    audit_path: Path,
    *,
    expected_rows: int = 3875,
    representation_name: str = "bioemu",
) -> dict[str, object]:
    """Validate and atomically flatten every checkpoint result into ``output_root``."""
    if representation_name not in {"bioemu", "bioemu_no_msa"}:
        raise ValueError("representation_name must be bioemu or bioemu_no_msa")
    catalog = _catalog_by_hash(catalog_path)
    archives = sorted(archive_root.glob("embedding_*.tar.gz"))
    if not archives:
        raise ValueError(f"no checkpoint archives found under {archive_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    expected_all: set[str] = set()
    imported: list[str] = []
    for archive in archives:
        sidecar = archive.with_suffix("").with_suffix(".json")
        expected = _sidecar(sidecar, archive)
        overlap = expected_all.intersection(expected)
        if overlap:
            raise ValueError(f"checkpoint archives duplicate {len(overlap)} sequence hashes")
        expected_all.update(expected)
        absent = set(expected) - set(catalog.index)
        if absent:
            raise ValueError(f"checkpoint archive has {len(absent)} hashes outside catalog")
        with tarfile.open(archive, "r:gz") as bundle:
            members = {
                Path(member.name).stem: member
                for member in bundle.getmembers()
                if member.isfile()
                and member.name.startswith("embeddings/")
                and member.name.endswith(".npz")
            }
            if set(members) != set(expected):
                raise ValueError(f"checkpoint member set mismatch: {archive}")
            for digest in expected:
                destination = output_root / f"{digest}.npz"
                if destination.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing embedding: {destination}"
                    )
                source = bundle.extractfile(members[digest])
                if source is None:
                    raise ValueError(f"unreadable checkpoint member: {digest}")
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".npz", prefix=f".{digest}.", dir=output_root, delete=False
                ) as temporary:
                    temporary.write(source.read())
                    temporary_path = Path(temporary.name)
                try:
                    row = catalog.loc[digest]
                    load_residue_matrix(
                        temporary_path,
                        protein_id=str(row.protein_id),
                        sequence=str(row.sequence),
                        sequence_sha256=digest,
                        sequence_length=int(row.sequence_length),
                        expected_width=EMBEDDING_WIDTH,
                    )
                    with np.load(temporary_path, allow_pickle=False) as values:
                        metadata = json.loads(str(values["metadata"].item()))
                    actual_config = metadata.get("extraction_config")
                    if representation_name == "bioemu_no_msa" and actual_config != QUERY_ONLY_CONFIG:
                        raise ValueError(
                            f"embedding extraction mode mismatch for {row.protein_id}"
                        )
                    if representation_name == "bioemu" and actual_config == QUERY_ONLY_CONFIG:
                        raise ValueError(
                            f"query-only embedding cannot enter the full-MSA lane: {row.protein_id}"
                        )
                    os.replace(temporary_path, destination)
                except Exception:
                    temporary_path.unlink(missing_ok=True)
                    raise
                imported.append(digest)
    if len(imported) != expected_rows or len(expected_all) != expected_rows:
        raise ValueError(
            f"checkpoint row count mismatch: imported={len(imported)}, expected={expected_rows}"
        )
    report = {
        "archive_root": str(archive_root),
        "archive_count": len(archives),
        "catalog": str(catalog_path),
        "embedding_width": EMBEDDING_WIDTH,
        "representation_name": representation_name,
        "imported": len(imported),
        "imported_sha256": hashlib.sha256("\n".join(sorted(imported)).encode()).hexdigest(),
        "output_root": str(output_root),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embeddings/bioemu_af2_model3"),
    )
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=3875)
    parser.add_argument(
        "--representation-name",
        choices=("bioemu", "bioemu_no_msa"),
        default="bioemu",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            import_archives(
                args.archive_root,
                args.catalog,
                args.output_root,
                args.audit_path,
                expected_rows=args.expected_rows,
                representation_name=args.representation_name,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
