"""Prepare and import single-only BioEmu Evoformer embeddings.

BioEmu obtains these representations with its inline AlphaFold2 model 3
Evoformer.  This script deliberately imports only the per-residue single
representation, not the quadratic pair representation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from protein_state_router.representations.bundle_io import (
    EmbeddingRequest,
    build_embedding_output_path,
    validate_single_embedding_bundle,
    write_embedding_bundle,
)
from protein_state_router.representations.embeddings import (
    EmbeddingSource,
    ProteinEmbeddings,
    SingleEmbedding,
)
from protein_state_router.representations.query import sequence_sha256

BIOEMU_VERSION = "1.4.1"
BIOEMU_MODEL_ID = "bioemu.colabfold_inline.alphafold2_model_3"
BIOEMU_OUTPUT_WIDTH = 384
BIOEMU_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")
BIOEMU_EXTRACTION_CONFIG = {
    "representation": "alphafold2_evoformer_single",
    "model_type": "alphafold2",
    "model_number": 3,
    "num_recycles": 0,
    "num_ensemble": 1,
    "templates": False,
    "msa_source": "colabfold_remote",
    "output_dtype": "float32",
}
BIOEMU_NO_MSA_EXTRACTION_CONFIG = {
    **BIOEMU_EXTRACTION_CONFIG,
    "msa_source": "query_only_a3m",
    "msa_depth": 1,
    "msa_network_access": False,
}
REQUIRED_COLUMNS = {"protein_id", "sequence", "sequence_sha256", "sequence_length"}
METADATA_COLUMNS = (
    "dataset_label",
    "label_class",
    "split",
    "source_dataset",
    "source_record_id",
)
MANIFEST_COLUMNS = (
    "protein_id",
    "sequence",
    "sequence_sha256",
    "sequence_length",
    *METADATA_COLUMNS,
    "cache_status",
    "cache_provenance",
    "a3m_path",
)


def prepare_manifest(
    dataset: Path,
    output: Path,
    *,
    new_only: bool = False,
    exclude_manifest: Path | None = None,
    require_new_a3m: bool = False,
    supported_residues_only: bool = False,
    msa_mode: str = "colabfold_remote",
) -> dict[str, object]:
    """Write a deterministic BioEmu manifest from a canonical or final cohort."""
    if msa_mode not in {"colabfold_remote", "query_only"}:
        raise ValueError("msa_mode must be colabfold_remote or query_only")
    catalog = pd.read_parquet(dataset)
    missing = REQUIRED_COLUMNS - set(catalog)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if catalog.protein_id.duplicated().any() or catalog.sequence_sha256.duplicated().any():
        raise ValueError("dataset protein IDs and sequence hashes must be unique")
    if new_only:
        if "cache_status" not in catalog:
            raise ValueError("--new-only requires a final cohort with cache_status")
        catalog = catalog.loc[catalog.cache_status.eq("new_a3m_required")].copy()
    excluded_rows = 0
    if exclude_manifest is not None:
        excluded = pd.read_csv(exclude_manifest, usecols=["sequence_sha256"])
        if excluded.sequence_sha256.duplicated().any():
            raise ValueError("excluded manifest sequence hashes must be unique")
        missing_hashes = set(excluded.sequence_sha256) - set(catalog.sequence_sha256)
        if missing_hashes:
            raise ValueError(
                f"excluded manifest contains {len(missing_hashes)} hashes absent from the dataset"
            )
        excluded_rows = len(excluded)
        catalog = catalog.loc[~catalog.sequence_sha256.isin(excluded.sequence_sha256)].copy()
    if require_new_a3m:
        catalog["cache_status"] = (
            "query_only" if msa_mode == "query_only" else "new_a3m_required"
        )
        catalog["cache_provenance"] = ""
        catalog["a3m_path"] = catalog.sequence_sha256.map(lambda value: f"a3m/{value}.a3m")
    rejected_path: Path | None = None
    rejected_rows = 0
    if supported_residues_only:
        supported = catalog.sequence.astype(str).str.upper().map(
            lambda value: bool(value) and set(value) <= BIOEMU_PROTEIN_ALPHABET
        )
        rejected = catalog.loc[~supported].copy()
        rejected_rows = len(rejected)
        if rejected_rows:
            rejected["rejection_reason"] = "unsupported_residue"
            rejected_path = output.with_name(f"{output.stem}_rejected.csv")
            rejected.to_csv(rejected_path, index=False)
        catalog = catalog.loc[supported].copy()
    manifest = catalog.loc[:, [column for column in MANIFEST_COLUMNS if column in catalog]]
    manifest = manifest.copy()
    manifest["sequence"] = manifest.sequence.astype(str).str.upper()
    manifest["sequence_length"] = manifest.sequence.str.len()
    if not manifest.sequence.map(sequence_sha256).equals(manifest.sequence_sha256):
        raise ValueError("dataset sequences do not match sequence_sha256")
    manifest = manifest.sort_values("sequence_sha256").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return {
        "rows": len(manifest),
        "new_only": new_only,
        "excluded_rows": excluded_rows,
        "require_new_a3m": require_new_a3m,
        "supported_residues_only": supported_residues_only,
        "msa_mode": msa_mode,
        "rejected_rows": rejected_rows,
        "rejected_manifest": str(rejected_path) if rejected_path else None,
        "manifest": str(output),
        "manifest_sha256": _sha256(output),
        "drive_destination": "MyDrive/dynamic_protein_router/bioemu_io/outputs/manifest.csv",
    }


def import_results(manifest_path: Path, results_dir: Path, output_root: Path) -> dict[str, object]:
    """Validate Colab NPZs and write package-native single-embedding bundles."""
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    output_root.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    missing_results: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        request = _request(row)
        source = results_dir / f"{request.sequence_sha256}.npz"
        if not source.is_file():
            missing_results.append(_record(row, "missing_result"))
            continue
        try:
            values, metadata = _load_result(source, request, row)
            embedding_source = EmbeddingSource(
                request.backend,
                request.backend_version,
                request.model_name,
                request.extraction_config_hash,
                request.sequence_sha256,
                "alphafold2_evoformer_single",
                0,
            )
            embeddings = ProteinEmbeddings(
                request.protein_id,
                request.sequence,
                request.sequence_sha256,
                embedding_source,
                SingleEmbedding(
                    torch.from_numpy(values.copy()),
                    torch.ones(len(values), dtype=torch.bool),
                    embedding_source,
                ),
                None,
                metadata={**metadata, **_record_metadata(row)},
            )
            destination = build_embedding_output_path(
                output_root,
                request.backend,
                request.backend_version,
                request.protein_id,
                request.sequence_sha256,
                request.extraction_config_hash,
            )
            write_embedding_bundle(embeddings, request, destination)
            validate_single_embedding_bundle(destination, request, width=BIOEMU_OUTPUT_WIDTH)
            accepted.append(
                {
                    **_record(row, "accepted"),
                    "bundle_path": str(destination),
                    "shape": f"{len(values)}x{BIOEMU_OUTPUT_WIDTH}",
                    "dtype": "float32",
                }
            )
        except Exception as error:
            rejected.append(_record(row, f"{type(error).__name__}: {error}"))
    manifest_root = output_root / "bioemu_af2_model3"
    manifest_root.mkdir(exist_ok=True)
    readme = manifest_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# BioEmu AlphaFold2 Evoformer embeddings\n\n"
            "Generated by `scripts/bioemu_dataset.py import`.\n"
            "These are single-only `[L, 384]` Evoformer conditioning representations, not BioEmu diffusion latents.\n"
        )
    pd.DataFrame(accepted).to_csv(manifest_root / "manifest.csv", index=False)
    pd.DataFrame(rejected).to_csv(manifest_root / "rejected.csv", index=False)
    pd.DataFrame(missing_results).to_csv(manifest_root / "missing.csv", index=False)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "missing": len(missing_results),
        "manifest": str(manifest_root / "manifest.csv"),
    }


def _request(row: dict[str, object]) -> EmbeddingRequest:
    sequence = str(row["sequence"]).upper()
    digest = str(row["sequence_sha256"])
    if sequence_sha256(sequence) != digest:
        raise ValueError(f"sequence hash mismatch for {row['protein_id']}")
    if len(sequence) != int(float(row["sequence_length"])):
        raise ValueError(f"sequence length mismatch for {row['protein_id']}")
    return EmbeddingRequest(
        str(row["protein_id"]),
        sequence,
        digest,
        "bioemu_af2_model3",
        BIOEMU_VERSION,
        BIOEMU_MODEL_ID,
        BIOEMU_EXTRACTION_CONFIG,
        ("single",),
    )


def _load_result(
    path: Path, request: EmbeddingRequest, row: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"single", "metadata"}:
            raise ValueError("result must contain only single and metadata")
        values = archive["single"]
        metadata = json.loads(str(archive["metadata"].item()))
    expected = {
        "protein_id": request.protein_id,
        "sequence_sha256": request.sequence_sha256,
        "sequence_length": len(request.sequence),
        "embedding_width": BIOEMU_OUTPUT_WIDTH,
        "model_id": BIOEMU_MODEL_ID,
        "model_revision": BIOEMU_VERSION,
        "extraction_config": BIOEMU_EXTRACTION_CONFIG,
    }
    if row.get("a3m_path"):
        expected["a3m_provenance"] = str(row["a3m_path"])
    if metadata != expected:
        raise ValueError("result metadata does not match the canonical request")
    if values.dtype != np.float32 or values.shape != (len(request.sequence), BIOEMU_OUTPUT_WIDTH):
        raise ValueError(
            f"single tensor must be float32 [{len(request.sequence)}, {BIOEMU_OUTPUT_WIDTH}]"
        )
    if not np.isfinite(values).all():
        raise ValueError("single tensor must be finite")
    return values, metadata


def _record(row: dict[str, object], status: str) -> dict[str, object]:
    return {
        "protein_id": row["protein_id"],
        "sequence_sha256": row["sequence_sha256"],
        "sequence_length": int(float(row["sequence_length"])),
        "status": status,
    }


def _record_metadata(row: dict[str, object]) -> dict[str, object]:
    return {
        column: row[column] for column in METADATA_COLUMNS if column in row and row[column] != ""
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/catalog.parquet"),
    )
    prepare.add_argument(
        "--msa-mode",
        choices=("colabfold_remote", "query_only"),
        default="colabfold_remote",
    )
    prepare.add_argument(
        "--exclude-manifest",
        type=Path,
        help="exclude sequence hashes already completed in another manifest",
    )
    prepare.add_argument(
        "--require-new-a3m",
        action="store_true",
        help="mark every emitted row for fresh MSA generation",
    )
    prepare.add_argument(
        "--supported-residues-only",
        action="store_true",
        help="reject rows outside BioEmu's supported protein alphabet",
    )
    prepare.add_argument(
        "--new-only",
        action="store_true",
        help="emit only final-cohort rows requiring a newly generated A3M",
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/bioemu_input_manifest.csv"),
    )
    importer = commands.add_parser("import")
    importer.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/bioemu_input_manifest.csv"),
    )
    importer.add_argument("--results-dir", type=Path, required=True)
    importer.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embeddings"),
    )
    args = parser.parse_args()
    report = (
        prepare_manifest(
            args.dataset,
            args.output,
            new_only=args.new_only,
            exclude_manifest=args.exclude_manifest,
            require_new_a3m=args.require_new_a3m,
            supported_residues_only=args.supported_residues_only,
            msa_mode=args.msa_mode,
        )
        if args.command == "prepare"
        else import_results(args.manifest, args.results_dir, args.output_root)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
