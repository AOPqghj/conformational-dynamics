"""Stream AlphaFold structures and cache compact single-structure geometry features."""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import requests
from Bio import Align
from requests.adapters import HTTPAdapter
from scipy.spatial import cKDTree

DEFAULT_CATALOG = Path(
    "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
)
DEFAULT_VECTOR_CACHE = Path("data/cache/alphafold_plddt_vectors.parquet")
DEFAULT_OUTPUT = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/"
    "alphafold_single_structure_geometry_features.parquet"
)
FEATURE_NAMES = (
    "helix_fraction",
    "sheet_fraction",
    "coil_fraction",
    "radius_gyration_normalized",
    "sasa_per_residue",
    "mean_relative_sasa",
    "exposed_residue_fraction",
    "contact_count",
    "contacts_per_residue",
    "contact_density",
    "relative_contact_order",
    "long_range_contact_fraction",
    "chain_compactness",
    "chain_asphericity",
)
MAX_RESIDUE_SASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}
THREAD_LOCAL = threading.local()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "protein-state-router/0.1 geometry-control"})
        session.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=1))
        THREAD_LOCAL.session = session
    return session


def _download(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            response = _session().get(url, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(float(response.headers.get("Retry-After", 2**attempt)), 20.0))
                continue
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"request failed after {attempts} attempts: {url}") from error
            time.sleep(min(2**attempt, 10.0))
    raise RuntimeError(f"request failed: {url}")


def map_query(target: str, query: str) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Map query residues onto an AFDB model sequence."""
    start = target.find(query)
    if start >= 0:
        query_indices = np.arange(len(query), dtype=np.int32)
        return (
            np.arange(start, start + len(query), dtype=np.int32),
            query_indices,
            1.0,
            1.0,
            "exact_subsequence",
        )
    aligner = Align.PairwiseAligner(
        mode="local",
        match_score=2.0,
        mismatch_score=-1.0,
        open_gap_score=-3.0,
        extend_gap_score=-0.5,
    )
    alignment = aligner.align(target, query)[0]
    target_indices: list[int] = []
    query_indices_list: list[int] = []
    for (target_start, target_stop), (query_start, query_stop) in zip(
        *alignment.aligned, strict=True
    ):
        target_indices.extend(range(int(target_start), int(target_stop)))
        query_indices_list.extend(range(int(query_start), int(query_stop)))
    target_array = np.asarray(target_indices, dtype=np.int32)
    query_array = np.asarray(query_indices_list, dtype=np.int32)
    if len(target_array) == 0:
        return target_array, query_array, 0.0, 0.0, "local_alignment"
    matches = sum(target[i] == query[j] for i, j in zip(target_array, query_array, strict=True))
    identity = matches / len(target_array)
    coverage = len(query_array) / len(query)
    return target_array, query_array, identity, coverage, "local_alignment"


def _parse_structure(content: bytes, sequence_length: int) -> Any:
    cif = pdbx.CIFFile.read(io.StringIO(content.decode("utf-8")))
    atoms = pdbx.get_structure(cif, model=1)
    ca = atoms[atoms.atom_name == "CA"]
    if len(ca) == 0:
        raise ValueError("structure has no alpha carbons")
    chain_ids, counts = np.unique(ca.chain_id, return_counts=True)
    chain_id = chain_ids[int(np.argmax(counts))]
    atoms = atoms[atoms.chain_id == chain_id]
    residue_count = len(struc.get_residue_starts(atoms))
    if residue_count != sequence_length:
        raise ValueError(
            f"structure/sequence residue mismatch: structure={residue_count}, sequence={sequence_length}"
        )
    return atoms


def geometry_features(
    atoms: Any, target_indices: np.ndarray, query_indices: np.ndarray, query_length: int
) -> dict[str, float]:
    """Calculate compact geometry descriptors for one mapped query fragment."""
    residue_count = len(struc.get_residue_starts(atoms))
    residue_ordinals = struc.spread_residue_wise(atoms, np.arange(residue_count))
    fragment = atoms[np.isin(residue_ordinals, target_indices)]
    ca = fragment[fragment.atom_name == "CA"]
    n_residues = len(ca)
    if n_residues != len(target_indices) or n_residues < 4:
        raise ValueError("mapped structure does not contain one CA per mapped residue")

    sse = struc.annotate_sse(ca)
    coordinates = np.asarray(ca.coord, dtype=np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    gyration_tensor = centered.T @ centered / n_residues
    eigenvalues = np.clip(np.linalg.eigvalsh(gyration_tensor), 0.0, None)
    radius_gyration = float(np.sqrt(eigenvalues.sum()))
    denominator = float(eigenvalues.sum() ** 2)
    asphericity = (
        float(1.5 * np.square(eigenvalues - eigenvalues.mean()).sum() / denominator)
        if denominator > 0
        else 0.0
    )

    atom_sasa = struc.sasa(fragment, point_number=100)
    residue_starts = struc.get_residue_starts(fragment)
    residue_sasa = np.add.reduceat(np.nan_to_num(atom_sasa, nan=0.0), residue_starts)
    residue_names = fragment.res_name[residue_starts]
    maximum_sasa = np.asarray([MAX_RESIDUE_SASA.get(name, np.nan) for name in residue_names])
    relative_sasa = residue_sasa / maximum_sasa
    valid_relative_sasa = relative_sasa[np.isfinite(relative_sasa)]
    if len(valid_relative_sasa) == 0:
        raise ValueError("mapped structure has no residues with defined relative SASA")

    pairs = cKDTree(coordinates).query_pairs(8.0, output_type="ndarray")
    if len(pairs):
        separations = np.abs(query_indices[pairs[:, 0]] - query_indices[pairs[:, 1]])
        pairs = pairs[separations > 2]
        separations = separations[separations > 2]
    else:
        separations = np.asarray([], dtype=np.int32)
    contact_count = len(separations)
    eligible_pairs = max(n_residues * (n_residues - 1) / 2 - (2 * n_residues - 3), 1)
    relative_contact_order = float(separations.mean() / query_length) if contact_count else 0.0
    long_range_fraction = float(np.mean(separations >= 24)) if contact_count else 0.0
    sphere_volume = 4.0 * np.pi * radius_gyration**3 / 3.0

    return {
        "helix_fraction": float(np.mean(sse == "a")),
        "sheet_fraction": float(np.mean(sse == "b")),
        "coil_fraction": float(np.mean(sse == "c")),
        "radius_gyration_normalized": radius_gyration / n_residues ** (1 / 3),
        "sasa_per_residue": float(residue_sasa.sum() / n_residues),
        "mean_relative_sasa": float(valid_relative_sasa.mean()),
        "exposed_residue_fraction": float(np.mean(valid_relative_sasa >= 0.25)),
        "contact_count": float(contact_count),
        "contacts_per_residue": float(2 * contact_count / n_residues),
        "contact_density": float(contact_count / eligible_pairs),
        "relative_contact_order": relative_contact_order,
        "long_range_contact_fraction": long_range_fraction,
        "chain_compactness": float(n_residues / sphere_volume) if sphere_volume > 0 else 0.0,
        "chain_asphericity": asphericity,
    }


def _process_accession(
    accession: str,
    sequence: str,
    structure_url: str,
    mappings: list[dict[str, object]],
) -> list[dict[str, object]]:
    try:
        atoms = _parse_structure(_download(structure_url), len(sequence))
        rows = []
        for mapping in mappings:
            rows.append(
                {
                    **mapping["metadata"],
                    "status": "ok",
                    "structure_url": structure_url,
                    "mapping_method": mapping["method"],
                    "mapping_identity": mapping["identity"],
                    "mapping_query_coverage": mapping["coverage"],
                    "mapped_residue_count": len(mapping["target_indices"]),
                    **geometry_features(
                        atoms,
                        mapping["target_indices"],
                        mapping["query_indices"],
                        int(mapping["query_length"]),
                    ),
                }
            )
        return rows
    except Exception as error:  # noqa: BLE001 - checkpoint accession-level failures
        return [
            {
                **mapping["metadata"],
                "status": f"error:{type(error).__name__}",
                "structure_url": structure_url,
                "error": str(error),
            }
            for mapping in mappings
        ]


def build_features(
    catalog: pd.DataFrame,
    vector_cache: pd.DataFrame,
    output_path: Path,
    workers: int,
    checkpoint_every: int,
) -> pd.DataFrame:
    vectors = vector_cache.set_index("uniprot_accession")
    cached = pd.read_parquet(output_path) if output_path.is_file() else pd.DataFrame()
    records = (
        cached.drop_duplicates("protein_id", keep="last").set_index("protein_id").to_dict("index")
        if not cached.empty
        else {}
    )
    by_accession: dict[str, dict[str, object]] = {}
    mapping_failures = 0
    for row in catalog.itertuples(index=False):
        existing = records.get(row.protein_id, {})
        if existing.get("status") in {"ok", "mapping_failed"}:
            continue
        vector = vectors.loc[row.uniprot_accession]
        target_indices, query_indices, identity, coverage, method = map_query(
            str(vector.sequence), str(row.sequence)
        )
        metadata = {
            "protein_id": row.protein_id,
            "uniprot_accession": row.uniprot_accession,
            "sequence_sha256": row.sequence_sha256,
        }
        if identity < 0.90 or coverage < 0.80:
            records[row.protein_id] = {
                **metadata,
                "status": "mapping_failed",
                "mapping_method": method,
                "mapping_identity": identity,
                "mapping_query_coverage": coverage,
                "error": "AFDB sequence does not cover the benchmark sequence",
            }
            mapping_failures += 1
            continue
        structure_url = (
            str(vector.plddt_url).replace("-confidence_", "-model_").replace(".json", ".cif")
        )
        item = by_accession.setdefault(
            row.uniprot_accession,
            {"sequence": str(vector.sequence), "structure_url": structure_url, "mappings": []},
        )
        item["mappings"].append(
            {
                "metadata": metadata,
                "target_indices": target_indices,
                "query_indices": query_indices,
                "query_length": len(row.sequence),
                "identity": identity,
                "coverage": coverage,
                "method": method,
            }
        )

    def checkpoint(completed: int) -> None:
        frame = pd.DataFrame(
            [{"protein_id": protein_id, **value} for protein_id, value in records.items()]
        )
        _write_parquet(frame, output_path)
        print(
            json.dumps(
                {
                    "event": "geometry_checkpoint",
                    "completed_accessions": completed,
                    "pending_accessions": len(by_accession),
                    "available_proteins": int(frame.status.eq("ok").sum()),
                    "mapping_failures": int(frame.status.eq("mapping_failed").sum()),
                    "other_failures": int(frame.status.str.startswith("error:").sum()),
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "geometry_cache_started",
                "cached_proteins": len(records) - mapping_failures,
                "new_mapping_failures": mapping_failures,
                "pending_accessions": len(by_accession),
                "workers": workers,
            }
        ),
        flush=True,
    )
    completed = 0
    items = iter(by_accession.items())
    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[list[dict[str, object]]], str] = {}
    try:
        for accession, item in items:
            futures[
                executor.submit(
                    _process_accession,
                    accession,
                    item["sequence"],
                    item["structure_url"],
                    item["mappings"],
                )
            ] = accession
            if len(futures) == workers:
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                for result in future.result():
                    protein_id = str(result.pop("protein_id"))
                    records[protein_id] = result
                completed += 1
                next_item = next(items, None)
                if next_item is not None:
                    accession, item = next_item
                    futures[
                        executor.submit(
                            _process_accession,
                            accession,
                            item["sequence"],
                            item["structure_url"],
                            item["mappings"],
                        )
                    ] = accession
                if completed % checkpoint_every == 0 or completed == len(by_accession):
                    checkpoint(completed)
    except KeyboardInterrupt:
        checkpoint(completed)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()

    result = pd.read_parquet(output_path)
    errors = result.status.str.startswith("error:")
    if errors.any():
        examples = result.loc[errors, ["protein_id", "status", "error"]].head().to_dict("records")
        raise RuntimeError(f"{int(errors.sum())} structure downloads failed: {examples}")
    if len(result) != len(catalog) or result.protein_id.nunique() != len(catalog):
        raise RuntimeError("geometry cache does not cover the complete cohort")
    return result


def run(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint_every must be positive")
    catalog = pd.read_parquet(args.catalog)
    catalog = catalog.loc[catalog.alphafold_mean_plddt.notna()].copy()
    if len(catalog) != 7_032 or catalog.protein_id.duplicated().any():
        raise ValueError("expected the 7,032-protein pLDDT-observed cohort")
    vector_cache = pd.read_parquet(args.vector_cache)
    result = build_features(catalog, vector_cache, args.output, args.workers, args.checkpoint_every)
    print(
        json.dumps(
            {
                "event": "geometry_features_completed",
                "proteins": len(result),
                "available": int(result.status.eq("ok").sum()),
                "mapping_failed": int(result.status.eq("mapping_failed").sum()),
                "output": str(args.output),
            }
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
