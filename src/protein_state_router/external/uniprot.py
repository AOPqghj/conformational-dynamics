"""Public UniProt cache and REST helpers used by dataset workflows."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UNIPROT_COLUMNS = (
    "primary_accession",
    "uniprot_id",
    "reviewed",
    "protein_name",
    "sequence",
    "comments_json",
    "keywords_json",
    "features_json",
    "fetch_status",
)
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


def normalize_uniprot_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cache rows with the stable public UniProt schema."""
    result = frame.copy()
    for column in UNIPROT_COLUMNS:
        if column not in result:
            result[column] = None
    return result.loc[:, list(UNIPROT_COLUMNS)].reset_index(drop=True)


def read_uniprot_cache(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return (
        normalize_uniprot_records(pd.read_parquet(path))
        if path.is_file()
        else pd.DataFrame(columns=UNIPROT_COLUMNS)
    )


def write_uniprot_cache(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("w") as handle:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        normalize_uniprot_records(frame).to_parquet(temporary, index=False)
        temporary.replace(path)


def uniprot_session() -> requests.Session:
    """Create the retrying HTTP client shared by UniProt callers."""
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        ),
    )
    session.headers["User-Agent"] = "protein-state-router/0.1 dataset workflow"
    return session


def fetch_uniprot_records(
    accessions: Iterable[str],
    cached: pd.DataFrame | None = None,
    *,
    session: requests.Session | None = None,
    timeout: float = 60,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch requested accessions, retaining terminal cached records by default."""
    cached = normalize_uniprot_records(cached if cached is not None else pd.DataFrame())
    requested = sorted(set(map(str, accessions)))
    present = set(
        cached.loc[cached.fetch_status.isin({"ok", "missing"}), "primary_accession"]
        .dropna()
        .astype(str)
    )
    missing = requested if refresh else sorted(set(requested) - present)
    client = session or uniprot_session()
    rows: list[dict[str, Any]] = []
    for accession in missing:
        response = client.get(UNIPROT_URL.format(accession=accession), timeout=timeout)
        if response.status_code == 404:
            rows.append(_missing(accession))
            continue
        response.raise_for_status()
        value = response.json()
        description = value.get("proteinDescription") or {}
        rows.append(
            {
                "primary_accession": value.get("primaryAccession", accession),
                "uniprot_id": value.get("uniProtkbId"),
                "reviewed": value.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
                "protein_name": _name(description),
                "sequence": (value.get("sequence") or {}).get("value"),
                "comments_json": json.dumps(value.get("comments") or []),
                "keywords_json": json.dumps(value.get("keywords") or []),
                "features_json": json.dumps(value.get("features") or []),
                "fetch_status": "ok",
            }
        )
    retained = cached.loc[~cached.primary_accession.astype(str).isin(missing)]
    return normalize_uniprot_records(pd.concat([retained, pd.DataFrame(rows)], ignore_index=True))


def fetch_uniprot_records_incremental(
    accessions: Iterable[str],
    cache_path: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 60,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Fetch records in durable batches so an interrupted job resumes safely."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    requested = sorted(set(map(str, accessions)))
    records = read_uniprot_cache(cache_path)
    client = session or uniprot_session()
    for start in range(0, len(requested), batch_size):
        records = fetch_uniprot_records(
            requested[start : start + batch_size], records, session=client, timeout=timeout
        )
        write_uniprot_cache(records, cache_path)
    return records


def _missing(accession: str) -> dict[str, Any]:
    return {
        "primary_accession": accession,
        "uniprot_id": None,
        "reviewed": None,
        "protein_name": None,
        "sequence": None,
        "comments_json": "[]",
        "keywords_json": "[]",
        "features_json": "[]",
        "fetch_status": "missing",
    }


def _name(description: Mapping[str, Any]) -> str | None:
    recommended = (description.get("recommendedName") or {}).get("fullName") or {}
    return str(recommended.get("value")) if recommended.get("value") else None
