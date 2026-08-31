"""Prepare ESMFold sequence inputs and import validated trunk embeddings."""

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
from protein_state_router.representations.esmfold_runner import (
    ESMFOLD_EXTRACTION_CONFIG,
    ESMFOLD_MODEL_ID,
    ESMFOLD_MODEL_REVISION,
    ESMFOLD_OUTPUT_WIDTH,
    normalize_esmfold_sequence,
)
from protein_state_router.representations.query import sequence_sha256

REQUIRED_DATASET_COLUMNS = {"protein_id", "sequence"}
RECORD_METADATA_COLUMNS = (
    "dataset_label",
    "label_class",
    "split",
    "dataset_row_id",
    "source_dataset",
    "source_record_id",
    "positive_subtype",
)


def prepare_manifest(
    dataset: Path, output: Path, drive_manifest_name: str = "manifest.csv"
) -> dict[str, object]:
    """Write the complete 2,000-protein sequence manifest for Google Drive."""
    frame = pd.read_parquet(dataset)
    missing = REQUIRED_DATASET_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if frame["protein_id"].duplicated().any():
        raise ValueError("dataset protein IDs must be unique")
    records = []
    for row in frame.itertuples(index=False):
        sequence = str(row.sequence).upper()
        digest = sequence_sha256(sequence)
        existing_hash = getattr(row, "sequence_hash", None)
        if (
            existing_hash not in {None, ""}
            and not pd.isna(existing_hash)
            and existing_hash != digest
        ):
            raise ValueError(f"sequence hash mismatch for {row.protein_id}")
        model_sequence = normalize_esmfold_sequence(sequence)
        eligible = len(sequence) <= int(ESMFOLD_EXTRACTION_CONFIG["max_sequence_length"])
        record_metadata = {
            column: _json_value(getattr(row, column))
            for column in RECORD_METADATA_COLUMNS
            if hasattr(row, column) and _has_value(getattr(row, column))
        }
        records.append(
            {
                "protein_id": row.protein_id,
                "sequence": sequence,
                "sequence_sha256": digest,
                "model_sequence": model_sequence,
                "model_sequence_sha256": sequence_sha256(model_sequence),
                "sequence_length": len(sequence),
                "record_metadata_json": json.dumps(record_metadata, sort_keys=True),
                "eligible": eligible,
                "exclusion_reason": "" if eligible else "sequence_length_exceeds_1022",
            }
        )
    manifest = pd.DataFrame(records).sort_values("sequence_sha256").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(output)
    return {
        "rows": len(manifest),
        "eligible": int(manifest["eligible"].sum()),
        "excluded": int((~manifest["eligible"]).sum()),
        "manifest": str(output),
        "manifest_sha256": _sha256(output),
        "drive_destination": (
            f"MyDrive/dynamic_protein_router/esmfold_io/inputs/{drive_manifest_name}"
        ),
    }


def import_results(manifest_path: Path, results_dir: Path, output_root: Path) -> dict[str, object]:
    """Validate Drive NPZs and create package-native ESMFold bundles."""
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    output_root.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        if not _as_bool(row["eligible"]):
            excluded.append(row)
            continue
        source_path = results_dir / f"{row['sequence_sha256']}.npz"
        if not source_path.is_file():
            continue
        try:
            values, metadata = _load_result(source_path, row)
            request = _request(row)
            source = EmbeddingSource(
                request.backend,
                request.backend_version,
                request.model_name,
                request.extraction_config_hash,
                request.sequence_sha256,
                "folding_trunk_s_s",
                0,
            )
            embeddings = ProteinEmbeddings(
                request.protein_id,
                request.sequence,
                request.sequence_sha256,
                source,
                SingleEmbedding(
                    torch.from_numpy(values.copy()),
                    torch.ones(len(request.sequence), dtype=torch.bool),
                    source,
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
            validate_single_embedding_bundle(destination, request, width=ESMFOLD_OUTPUT_WIDTH)
            manifest_root = output_root / request.backend
            accepted.append(
                {
                    "protein_id": request.protein_id,
                    "sequence_sha256": request.sequence_sha256,
                    "sequence_length": len(request.sequence),
                    **_record_metadata(row),
                    "bundle_path": str(destination.relative_to(manifest_root)),
                    "shape": f"{len(request.sequence)}x{ESMFOLD_OUTPUT_WIDTH}",
                    "dtype": "float32",
                    "model_id": ESMFOLD_MODEL_ID,
                    "model_revision": ESMFOLD_MODEL_REVISION,
                }
            )
        except Exception as error:
            rejected.append(
                {
                    "protein_id": row["protein_id"],
                    "sequence_sha256": row["sequence_sha256"],
                    "source_result": str(source_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    manifest_root = output_root / "esmfold_v1"
    manifest_root.mkdir(exist_ok=True)
    readme = manifest_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Validated ESMFold v1 embeddings\n\n"
            "This folder is generated by `scripts/esmfold_dataset.py import`. "
            "`manifest.parquet` and `manifest.csv` index accepted bundle paths; "
            "`rejected.csv` and `excluded.csv` explain records without usable embeddings.\n"
        )
    accepted_frame = pd.DataFrame(accepted)
    accepted_frame.to_csv(manifest_root / "manifest.csv", index=False)
    accepted_frame.to_parquet(manifest_root / "manifest.parquet", index=False)
    pd.DataFrame(rejected).to_csv(manifest_root / "rejected.csv", index=False)
    pd.DataFrame(excluded).to_csv(manifest_root / "excluded.csv", index=False)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "excluded": len(excluded),
        "missing": int(frame["eligible"].map(_as_bool).sum()) - len(accepted) - len(rejected),
        "manifest": str(manifest_root / "manifest.parquet"),
    }


def _request(row: dict[str, object]) -> EmbeddingRequest:
    return EmbeddingRequest(
        str(row["protein_id"]),
        str(row["sequence"]),
        str(row["sequence_sha256"]),
        "esmfold_v1",
        ESMFOLD_MODEL_REVISION,
        ESMFOLD_MODEL_ID,
        ESMFOLD_EXTRACTION_CONFIG,
        ("single",),
    )


def _load_result(path: Path, row: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"single", "metadata"}:
            raise ValueError("result must contain only single and metadata")
        values = archive["single"]
        metadata = json.loads(str(archive["metadata"].item()))
    expected_shape = (len(str(row["sequence"])), ESMFOLD_OUTPUT_WIDTH)
    if (
        values.dtype != np.float32
        or values.shape != expected_shape
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"single tensor must be finite float32 {expected_shape}")
    expected = {
        "protein_id": row["protein_id"],
        "sequence_sha256": row["sequence_sha256"],
        "model_sequence_sha256": row["model_sequence_sha256"],
        "model_id": ESMFOLD_MODEL_ID,
        "model_revision": ESMFOLD_MODEL_REVISION,
        "extraction_config": ESMFOLD_EXTRACTION_CONFIG,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("result metadata does not match the canonical manifest")
    return values, metadata


def _as_bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def _record_metadata(row: dict[str, object]) -> dict[str, object]:
    """Read optional catalog metadata without coupling embedding I/O to labels."""
    encoded = row.get("record_metadata_json")
    if encoded not in {None, ""} and not pd.isna(encoded):
        value = json.loads(str(encoded))
        if isinstance(value, dict):
            return value
    return {
        column: _json_value(row[column])
        for column in RECORD_METADATA_COLUMNS
        if column in row and _has_value(row[column])
    }


def _has_value(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and value != ""


def _json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


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
        "--output",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/esmfold_input_manifest.csv"),
    )
    prepare.add_argument(
        "--drive-manifest-name",
        default="manifest.csv",
        help="Filename to use when uploading this local manifest to Google Drive.",
    )
    importer = commands.add_parser("import")
    importer.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/esmfold_input_manifest.csv"),
    )
    importer.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/esmfold_results"),
    )
    importer.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lifecycle/final/initial_8598_dataset/embeddings"),
    )
    args = parser.parse_args()
    report = (
        prepare_manifest(args.dataset, args.output, args.drive_manifest_name)
        if args.command == "prepare"
        else import_results(args.manifest, args.results_dir, args.output_root)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
