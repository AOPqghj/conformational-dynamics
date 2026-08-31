"""API-backed discovery helpers for conservative negative candidates.

Discovery only gathers identifiers and metadata; it never computes or invents
structural RMSD/TM-score evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import requests

from protein_state_router.external.dynamicmpnn_positive import (
    RCSB_COLUMNS,
    SIFTS_COLUMNS,
    _http_session,
    fetch_rcsb,
    fetch_sifts,
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


def discover_structure_candidates(
    uniprot_ids: Iterable[str],
    *,
    session: requests.Session | None = None,
    timeout: float = 60,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Find experimental PDB entities mapped to each UniProt accession.

    The RCSB search service is queried per accession. Returned rows are only
    identifiers; downstream SIFTS/GraphQL enrichment supplies metadata.
    """
    client = session or _http_session()
    rows: list[dict[str, str]] = []
    for accession in sorted({str(x).strip() for x in uniprot_ids if str(x).strip()}):
        payload: dict[str, Any] = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "operator": "exact_match",
                            "value": accession,
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "operator": "exact_match",
                            "value": "UniProt",
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        },
                    },
                ],
            },
            "return_type": "polymer_entity",
            "request_options": {"paginate": {"start": 0, "rows": max_rows}},
        }
        response = client.post(RCSB_SEARCH_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        for item in response.json().get("result_set", []):
            identifier = str(item.get("identifier", ""))
            if "_" not in identifier:
                continue
            pdb_id, entity_id = identifier.split("_", 1)
            rows.append({"pdb_id": pdb_id.upper(), "entity_id": entity_id, "uniprot_id": accession})
    return pd.DataFrame(rows, columns=["pdb_id", "entity_id", "uniprot_id"]).drop_duplicates()


def enrich_negative_candidates(
    candidates: pd.DataFrame,
    *,
    cache_dir: str,
    session: requests.Session | None = None,
    timeout: float = 60,
    refresh: bool = False,
) -> pd.DataFrame:
    """Attach SIFTS and RCSB metadata, preserving missing evidence explicitly."""
    frame = candidates.copy()
    if "chain_id" not in frame.columns:
        frame = resolve_entity_chains(frame, session=session, timeout=timeout)
    if frame.empty:
        return frame
    if "chain_id" not in frame or "pdb_id" not in frame:
        raise ValueError(
            "candidates must contain pdb_id and chain_id columns; discovery entity_id is not a chain"
        )
    pairs = sorted(
        {(str(p).upper(), str(c)) for p, c in zip(frame.pdb_id, frame.chain_id, strict=False)}
    )
    client = session or _http_session()
    from pathlib import Path

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    sifts = fetch_sifts(pairs, pd.DataFrame(columns=SIFTS_COLUMNS), client, timeout, refresh)
    rcsb = fetch_rcsb(
        sorted({p for p, _ in pairs}), pd.DataFrame(columns=RCSB_COLUMNS), client, timeout, refresh
    )
    merged = frame.merge(sifts, on=["pdb_id", "chain_id"], how="left", suffixes=("", "_sifts"))
    return merged.merge(rcsb, on="pdb_id", how="left", suffixes=("", "_rcsb"))


def resolve_entity_chains(
    candidates: pd.DataFrame,
    *,
    session: requests.Session | None = None,
    timeout: float = 60,
) -> pd.DataFrame:
    """Expand RCSB polymer entities to author asym-chain IDs for SIFTS joins."""
    if not {"pdb_id", "entity_id"}.issubset(candidates.columns):
        raise ValueError("candidates require pdb_id and entity_id or chain_id")
    client = session or _http_session()
    rows: list[dict[str, str]] = []
    for pdb_id, entity_id in (
        candidates[["pdb_id", "entity_id"]].drop_duplicates().itertuples(index=False)
    ):
        response = client.get(
            RCSB_ENTITY_URL.format(pdb_id=str(pdb_id).upper(), entity_id=entity_id),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        identifiers = payload.get("rcsb_polymer_entity_container_identifiers") or {}
        chains = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []
        rows.extend(
            {"pdb_id": str(pdb_id).upper(), "entity_id": str(entity_id), "chain_id": str(chain)}
            for chain in chains
        )
    expanded = candidates.merge(pd.DataFrame(rows), on=["pdb_id", "entity_id"], how="inner")
    return expanded.drop_duplicates().reset_index(drop=True)
