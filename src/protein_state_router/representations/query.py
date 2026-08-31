"""UniProt-backed protein query resolution with explicit name disambiguation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from protein_state_router.constants import AMINO_ACIDS
from protein_state_router.representations.errors import ProteinQueryError


@dataclass(frozen=True, slots=True)
class ProteinQuery:
    value: str
    kind: Literal["uniprot", "name", "sequence"]

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ProteinQueryError("protein query cannot be empty")
        if self.kind == "sequence":
            invalid = set(self.value.upper().replace(" ", "")) - AMINO_ACIDS
            if invalid:
                raise ProteinQueryError(
                    f"illegal amino-acid characters: {''.join(sorted(invalid))}"
                )


@dataclass(frozen=True, slots=True)
class ResolvedProtein:
    protein_id: str
    sequence: str
    sequence_sha256: str
    query: ProteinQuery
    uniprot_accession: str | None
    recommended_name: str | None
    organism_name: str | None
    reviewed: bool | None


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _fetch_json(url: str) -> dict:
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": "protein-state-router/0.1"}
    )
    try:
        with urlopen(request, timeout=20) as response:  # nosec B310 - fixed UniProt host
            return json.loads(response.read().decode())
    except OSError as error:
        raise ProteinQueryError(f"UniProt request failed: {error}") from error


def _resolved_from_uniprot(record: dict, query: ProteinQuery) -> ResolvedProtein:
    sequence = record["sequence"]["value"].upper()
    accession = record["primaryAccession"]
    name = (
        record.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    organism = record.get("organism", {}).get("scientificName")
    return ResolvedProtein(
        accession,
        sequence,
        sequence_sha256(sequence),
        query,
        accession,
        name,
        organism,
        record.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
    )


def resolve_protein_query(query: ProteinQuery) -> ResolvedProtein | list[ResolvedProtein]:
    """Resolve exact accession/sequence; return candidates instead of guessing for names."""
    if query.kind == "sequence":
        sequence = re.sub(r"\s+", "", query.value.upper())
        identifier = f"sequence_{sequence_sha256(sequence)[:12]}"
        return ResolvedProtein(
            identifier, sequence, sequence_sha256(sequence), query, None, None, None, None
        )
    if query.kind == "uniprot":
        return _resolved_from_uniprot(
            _fetch_json(f"https://rest.uniprot.org/uniprotkb/{query.value.upper()}.json"), query
        )
    params = urlencode(
        {"query": f"protein_name:{query.value} AND reviewed:true", "format": "json", "size": 10}
    )
    results = _fetch_json(f"https://rest.uniprot.org/uniprotkb/search?{params}").get("results", [])
    if not results:
        raise ProteinQueryError(f"no reviewed UniProt candidates for {query.value!r}")
    return [_resolved_from_uniprot(record, query) for record in results]
