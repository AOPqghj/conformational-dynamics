"""Normalize the public ProMiSE conformational-pair tables into protein records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from protein_state_router.representations.query import sequence_sha256

PROMISE_REPOSITORY = "https://github.com/seoklab/promise-bench.git"
PROMISE_SUBTYPES = {
    "intrinsic": "intrinsic_multistate",
    "ligand-induced": "ligand_induced",
    "protein-induced": "protein_induced",
}
PAIR_COLUMNS = (
    "cluster",
    "pdb_a",
    "asm_a",
    "chain_a",
    "conf_label_a",
    "pdb_b",
    "asm_b",
    "chain_b",
    "conf_label_b",
)
CHAIN_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id}/{chain_id}"
ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


@dataclass(frozen=True, slots=True)
class PromiseChain:
    """Sequence and optional UniProt mapping for one ProMiSE assembly chain."""

    pdb_id: str
    chain_id: str
    sequence: str
    uniprot_id: str | None


def read_pair_tables(source_dir: str | Path) -> pd.DataFrame:
    """Read the three official tables and attach their normalized mechanism."""
    root = Path(source_dir)
    tables: list[pd.DataFrame] = []
    for filename, subtype in PROMISE_SUBTYPES.items():
        path = root / f"{filename}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing official ProMiSE table: {path}")
        table = pd.read_csv(path, dtype=str).fillna("")
        missing = set(PAIR_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        table = pd.DataFrame(table.loc[:, list(PAIR_COLUMNS)].copy())
        table["positive_subtype"] = subtype
        table["state_mechanism"] = subtype
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def build_positive_candidates(
    pairs: pd.DataFrame,
    *,
    chain_cache: pd.DataFrame | None = None,
    cache_path: str | Path | None = None,
    session: requests.Session | None = None,
    timeout: float = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve one sequence per ProMiSE cluster and preserve every failed record.

    A ProMiSE cluster can contain several reference pairs.  The first state chain
    with a resolvable canonical polymer sequence represents the protein; all
    pair and assembly references remain attached as JSON provenance.
    """
    required = set(PAIR_COLUMNS) | {"positive_subtype", "state_mechanism"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"ProMiSE pairs are missing columns: {sorted(missing)}")
    cache = _normalize_chain_cache(chain_cache)
    client = session or _session()
    candidates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    grouped = pairs.groupby(["positive_subtype", "cluster"], sort=True)
    for index, ((subtype, cluster), group) in enumerate(grouped, start=1):
        state_a = _state_records(group, "a")
        state_b = _state_records(group, "b")
        sequence_record, cache = _first_resolved_chain([*state_a, *state_b], cache, client, timeout)
        record_id = f"{subtype}:{cluster}"
        if sequence_record is None:
            exclusions.append(
                {
                    "source_record_id": record_id,
                    "positive_subtype": subtype,
                    "exclusion_reason": "sequence_lookup_failed",
                    "n_state_pairs": len(group),
                }
            )
            continue
        structures = _unique([item["structure_id"] for item in [*state_a, *state_b]])
        candidates.append(
            {
                "protein_id": f"promise:{subtype}:{cluster}",
                "uniprot_id": sequence_record.uniprot_id,
                "sequence": sequence_record.sequence,
                "sequence_hash": sequence_sha256(sequence_record.sequence),
                "sequence_length": len(sequence_record.sequence),
                "source_dataset": "ProMiSE",
                "source_record_id": record_id,
                "source_reference": "ProMiSE-bench official dataset",
                "positive_subtype": subtype,
                "state_mechanism": str(group.state_mechanism.iloc[0]),
                "state_a_structure_ids_json": json.dumps(
                    _unique([item["structure_id"] for item in state_a])
                ),
                "state_b_structure_ids_json": json.dumps(
                    _unique([item["structure_id"] for item in state_b])
                ),
                "structure_ids_json": json.dumps(structures),
                "ligand_context_json": json.dumps(
                    {"mechanism": subtype, "available_from_pair_table": False}, sort_keys=True
                ),
                "protein_partner_context_json": json.dumps(
                    {"mechanism": subtype, "available_from_pair_table": False}, sort_keys=True
                ),
                "biological_assembly_ids_json": json.dumps(
                    _unique([item["assembly_id"] for item in [*state_a, *state_b]])
                ),
                "label": 1,
                "dataset_label": 1,
                "label_class": "alternate_structured_state",
                "label_confidence": "gold",
                "single_structure_insufficient_derived": True,
                "is_training_ready_positive": True,
                "requires_manual_audit": False,
                "label_notes": "Curated ProMiSE conformational-pair benchmark record.",
                "n_reference_pairs": len(group),
            }
        )
        if cache_path is not None and index % 25 == 0:
            write_chain_cache(cache, cache_path)
    columns = _candidate_columns()
    if cache_path is not None:
        write_chain_cache(cache, cache_path)
    return (
        pd.DataFrame(candidates, columns=columns),
        pd.DataFrame(exclusions),
        _normalize_chain_cache(cache),
    )


def select_unique_candidates(
    candidates: pd.DataFrame,
    *,
    existing_sequences: Iterable[str] = (),
    existing_uniprot_ids: Iterable[str] = (),
    limit: int = 2500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove cross-source duplicates and take a deterministic proportional sample."""
    if limit < 1:
        raise ValueError("limit must be positive")
    result = candidates.copy()
    existing_hashes = {sequence_sha256(str(value)) for value in existing_sequences if value}
    existing_ids = {str(value) for value in existing_uniprot_ids if value and str(value) != "nan"}
    exclusion_rows: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    seen_uniprot: set[str] = set()
    records = result.sort_values(["positive_subtype", "source_record_id"]).to_dict("records")
    for raw_row in records:
        row = {str(key): value for key, value in raw_row.items()}
        digest = str(row["sequence_hash"])
        accession = str(row.get("uniprot_id") or "")
        reason = ""
        if digest in existing_hashes:
            reason = "overlaps_dynamicmpnn_sequence"
        elif accession and accession in existing_ids:
            reason = "overlaps_dynamicmpnn_uniprot"
        elif digest in seen_hashes:
            reason = "duplicate_promise_sequence"
        elif accession and accession in seen_uniprot:
            reason = "duplicate_promise_uniprot"
        if reason:
            exclusion_rows.append(
                {"source_record_id": row["source_record_id"], "exclusion_reason": reason}
            )
            continue
        retained.append(row)
        seen_hashes.add(digest)
        if accession:
            seen_uniprot.add(accession)
    eligible = pd.DataFrame(retained, columns=result.columns)
    selected = _proportional_sample(eligible, limit, seed)
    selected_ids = set(selected.source_record_id)
    outside = eligible.loc[~eligible.source_record_id.isin(selected_ids)].to_dict("records")
    for outside_row in outside:
        exclusion_rows.append(
            {
                "source_record_id": outside_row["source_record_id"],
                "exclusion_reason": "outside_target_sample",
            }
        )
    return selected.reset_index(drop=True), pd.DataFrame(exclusion_rows)


def source_manifest(source_dir: str | Path, revision: str) -> dict[str, object]:
    """Describe the exact official CSV snapshot used by one build."""
    root = Path(source_dir)
    files = {name: _file_sha256(root / f"{name}.csv") for name in PROMISE_SUBTYPES}
    return {
        "source_dataset": "ProMiSE",
        "repository": PROMISE_REPOSITORY,
        "revision": revision,
        "files": files,
    }


def read_chain_cache(path: str | Path) -> pd.DataFrame:
    """Load the durable RCSB sequence cache, or an empty one for a new run."""
    location = Path(path)
    return (
        _normalize_chain_cache(pd.read_parquet(location))
        if location.is_file()
        else _normalize_chain_cache(None)
    )


def write_chain_cache(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically checkpoint resolved chain sequences for safe resume."""
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    temporary = location.with_name(f".{location.name}.tmp")
    _normalize_chain_cache(frame).to_parquet(temporary, index=False)
    temporary.replace(location)


def _state_records(group: pd.DataFrame, state: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in group.itertuples(index=False):
        pdb_id = str(getattr(row, f"pdb_{state}")).lower()
        assembly = str(getattr(row, f"asm_{state}"))
        chain = str(getattr(row, f"chain_{state}"))
        structure_id = f"{pdb_id}_{assembly}_{chain}"
        if structure_id not in seen:
            records.append(
                {
                    "pdb_id": pdb_id,
                    "assembly_id": assembly,
                    "chain_id": chain,
                    "structure_id": structure_id,
                }
            )
            seen.add(structure_id)
    return records


def _first_resolved_chain(
    state_records: list[dict[str, str]],
    cache: pd.DataFrame,
    session: requests.Session,
    timeout: float,
) -> tuple[PromiseChain | None, pd.DataFrame]:
    current = cache
    for state in state_records:
        key = (state["pdb_id"].upper(), state["chain_id"])
        cached = current.loc[(current.pdb_id == key[0]) & (current.chain_id == key[1])]
        cached_ok = cached.loc[cached.fetch_status.eq("ok")]
        if cached_ok.empty:
            try:
                chain = fetch_chain(*key, session=session, timeout=timeout)
                row = {
                    "pdb_id": key[0],
                    "chain_id": key[1],
                    "sequence": chain.sequence,
                    "uniprot_id": chain.uniprot_id,
                    "fetch_status": "ok",
                    "error": "",
                }
            except requests.RequestException as error:
                row = {
                    "pdb_id": key[0],
                    "chain_id": key[1],
                    "sequence": "",
                    "uniprot_id": "",
                    "fetch_status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            current = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
            cached_ok = current.tail(1)
        record = cached_ok.iloc[-1]
        if record.fetch_status == "ok" and str(record.sequence):
            return (
                PromiseChain(
                    key[0], key[1], str(record.sequence), _optional_string(record.uniprot_id)
                ),
                current,
            )
    return None, current


def fetch_chain(
    pdb_id: str, chain_id: str, *, session: requests.Session, timeout: float
) -> PromiseChain:
    """Fetch a canonical chain sequence and an optional UniProt cross-reference."""
    instance_response = session.get(
        CHAIN_URL.format(pdb_id=pdb_id, chain_id=_chain_letter(chain_id)), timeout=timeout
    )
    instance_response.raise_for_status()
    instance: dict[str, Any] = instance_response.json()
    entity_id = (instance.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get(
        "entity_id"
    )
    if not entity_id:
        raise requests.RequestException(f"RCSB returned no entity for {pdb_id}/{chain_id}")
    entity_response = session.get(
        ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id), timeout=timeout
    )
    entity_response.raise_for_status()
    payload: dict[str, Any] = entity_response.json()
    sequence = _clean_sequence(
        (payload.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can")
    )
    if not sequence:
        raise requests.RequestException(
            f"RCSB returned no canonical sequence for {pdb_id}/{chain_id}"
        )
    identifiers = (payload.get("rcsb_polymer_entity_container_identifiers") or {}).get(
        "reference_sequence_identifiers"
    ) or []
    uniprot_id = next(
        (
            str(item.get("database_accession"))
            for item in identifiers
            if item.get("database_name") == "UniProt" and item.get("database_accession")
        ),
        None,
    )
    return PromiseChain(pdb_id, chain_id, sequence, uniprot_id)


def _proportional_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.sort_values("source_record_id")
    total = len(frame)
    allocations = {
        subtype: int(len(group) * limit / total)
        for subtype, group in frame.groupby("positive_subtype", sort=True)
    }
    remaining = limit - sum(allocations.values())
    remainders = sorted(
        (
            (len(group) * limit / total - allocations[subtype], subtype)
            for subtype, group in frame.groupby("positive_subtype", sort=True)
        ),
        reverse=True,
    )
    for _, subtype in remainders[:remaining]:
        allocations[subtype] += 1
    samples = []
    for subtype, group in frame.groupby("positive_subtype", sort=True):
        ordered = group.assign(
            _sample_key=group.source_record_id.map(
                lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
            )
        ).sort_values("_sample_key")
        samples.append(ordered.head(allocations[subtype]).drop(columns="_sample_key"))
    return pd.concat(samples, ignore_index=True).sort_values("source_record_id")


def _candidate_columns() -> list[str]:
    return [
        "protein_id",
        "uniprot_id",
        "sequence",
        "sequence_hash",
        "sequence_length",
        "source_dataset",
        "source_record_id",
        "source_reference",
        "positive_subtype",
        "state_mechanism",
        "state_a_structure_ids_json",
        "state_b_structure_ids_json",
        "structure_ids_json",
        "ligand_context_json",
        "protein_partner_context_json",
        "biological_assembly_ids_json",
        "label",
        "dataset_label",
        "label_class",
        "label_confidence",
        "single_structure_insufficient_derived",
        "is_training_ready_positive",
        "requires_manual_audit",
        "label_notes",
        "n_reference_pairs",
    ]


def _normalize_chain_cache(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["pdb_id", "chain_id", "sequence", "uniprot_id", "fetch_status", "error"]
    if frame is None:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = ""
    result["pdb_id"] = result.pdb_id.astype(str).str.upper()
    result["chain_id"] = result.chain_id.astype(str)
    return result.loc[:, columns].drop_duplicates(["pdb_id", "chain_id"], keep="last")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "protein-state-router/0.1 promise-ingestion"
    return session


def _clean_sequence(value: object) -> str:
    return re.sub(r"[^A-Z]", "", str(value or "").upper())


def _chain_letter(chain_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", chain_id)
    return match.group(1) if match else chain_id


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _optional_string(value: object) -> str | None:
    text = str(value or "")
    return text if text and text.lower() != "nan" else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
