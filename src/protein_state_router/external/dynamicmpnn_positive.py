"""Cached metadata enrichment and deterministic positive labels for DynamicMPNN."""

from __future__ import annotations

import io
import json
import os
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from protein_state_router.external.dynamicmpnn import parse_member_identifier

LABEL_CLASSES = {
    "alternate_structured_state",
    "condition_aware_structured_state",
    "both_alternate_and_condition_aware",
    "ambiguous_structural_diversity",
    "excluded",
}
LABEL_CONFIDENCES = {"gold", "silver", "bronze", "excluded"}
CONDITION_CLASSES = {
    "condition_aware_structured_state",
    "both_alternate_and_condition_aware",
}
SIFTS_COLUMNS = ("pdb_id", "chain_id", "uniprot_id", "mapping_found", "fetch_status")
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
RCSB_COLUMNS = (
    "pdb_id",
    "title",
    "experimental_method",
    "resolution",
    "polymer_entity_count",
    "assembly_count",
    "ligands_json",
    "polymer_entities_json",
    "assembly_instance_counts_json",
    "fetch_status",
)
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
RCSB_QUERY = """
query EntryMetadata($entry_ids: [String!]!) {
  entries(entry_ids: $entry_ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info {
      resolution_combined
      polymer_entity_count
      assembly_count
    }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
      rcsb_polymer_entity { pdbx_description }
    }
    nonpolymer_entities {
      nonpolymer_comp { chem_comp { id } }
    }
    assemblies {
      rcsb_assembly_info { polymer_entity_instance_count }
    }
  }
}
"""


def enrich_metadata(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    offline: bool = False,
    refresh: bool = False,
    strict: bool = False,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch or reuse the three small metadata caches required for labeling."""
    api = config["api"]
    cache_dir = Path(api["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(_candidate_pairs(candidates))
    client = session or _http_session()
    timeout = float(api.get("timeout_seconds", 60))

    sifts_path = cache_dir / "pdbe_sifts_pdb_chain_uniprot.parquet"
    sifts = _read_cache(sifts_path, SIFTS_COLUMNS)
    if api.get("use_sifts", True) and not offline:
        try:
            sifts = fetch_sifts(pairs, sifts, client, timeout, refresh)
        except (requests.RequestException, RuntimeError) as error:
            warnings.warn(f"SIFTS enrichment failed: {error}", RuntimeWarning, stacklevel=2)
            sifts = _record_sifts_errors(sifts, pairs, error)
            _write_cache(sifts, sifts_path)
            if strict:
                raise RuntimeError("SIFTS enrichment failed in strict mode") from error
        _write_cache(sifts, sifts_path)

    # `sifts` is a durable shared cache and can contain mappings from earlier,
    # much larger runs.  Restrict downstream UniProt work to this invocation's
    # PDB-chain pairs; otherwise a small labeling batch accidentally fans out
    # across the entire cache.
    requested_pairs = pd.MultiIndex.from_tuples(pairs, names=["pdb_id", "chain_id"])
    selected_sifts = sifts.loc[
        pd.MultiIndex.from_frame(sifts[["pdb_id", "chain_id"]]).isin(requested_pairs)
        & sifts.mapping_found.astype("boolean").fillna(False)
    ]
    accessions = sorted(set(selected_sifts.uniprot_id.dropna().astype(str)))
    uniprot_path = cache_dir / "uniprot_records.parquet"
    uniprot = _read_cache(uniprot_path, UNIPROT_COLUMNS)
    if api.get("use_uniprot", True) and not offline:
        # Persist bounded batches.  A long interrupted run then resumes from
        # completed records instead of losing thousands of successful calls.
        # A request failure only redoes this bounded unit.  The cache write is
        # intentionally frequent enough for unattended recovery without
        # rewriting the growing parquet after every individual HTTP response.
        for batch in _chunks(accessions, 25):
            try:
                uniprot = fetch_uniprot(batch, uniprot, client, timeout, refresh)
            except (requests.RequestException, RuntimeError) as error:
                warnings.warn(f"UniProt enrichment failed: {error}", RuntimeWarning, stacklevel=2)
                uniprot = _record_uniprot_errors(uniprot, batch, error)
            _write_cache(uniprot, uniprot_path)
            if strict and (
                uniprot.loc[uniprot.primary_accession.astype(str).isin(batch), "fetch_status"]
                .astype(str)
                .str.startswith("error:")
                .any()
            ):
                raise RuntimeError("UniProt enrichment failed in strict mode; retry from the cache")

    pdb_ids = sorted({pdb_id for pdb_id, _ in pairs})
    rcsb_path = cache_dir / "rcsb_entry_metadata.parquet"
    rcsb = _read_cache(rcsb_path, RCSB_COLUMNS)
    if api.get("use_rcsb", True) and not offline:
        # `fetch_rcsb` itself calls GraphQL in 75-ID units, so checkpoint at
        # that same unit rather than losing ten successful sub-requests.
        for batch in _chunks(pdb_ids, 75):
            try:
                rcsb = fetch_rcsb(batch, rcsb, client, timeout, refresh)
            except (requests.RequestException, RuntimeError) as error:
                warnings.warn(f"RCSB enrichment failed: {error}", RuntimeWarning, stacklevel=2)
                rcsb = _record_rcsb_errors(rcsb, batch, error)
            _write_cache(rcsb, rcsb_path)
            if strict and (
                rcsb.loc[rcsb.pdb_id.astype(str).isin(batch), "fetch_status"]
                .astype(str)
                .str.startswith("error:")
                .any()
            ):
                raise RuntimeError("RCSB enrichment failed in strict mode; retry from the cache")
    return sifts, uniprot, rcsb


def fetch_sifts(
    pairs: list[tuple[str, str]],
    cached: pd.DataFrame,
    session: requests.Session,
    timeout: float,
    refresh: bool = False,
) -> pd.DataFrame:
    """Filter the official SIFTS chain mapping down to requested PDB chains."""
    requested = set(pairs)
    terminal = cached.fetch_status.isin({"ok", "missing"})
    present = set(
        zip(
            cached.loc[terminal, "pdb_id"].astype(str),
            cached.loc[terminal, "chain_id"].astype(str),
            strict=False,
        )
    )
    missing = requested if refresh else requested - present
    if not missing:
        return cached
    response = session.get(SIFTS_URL, timeout=timeout)
    response.raise_for_status()
    found: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        io.BytesIO(response.content), compression="gzip", comment="#", chunksize=250_000
    ):
        renamed = {str(column).upper(): column for column in chunk.columns}
        pdb_column = renamed.get("PDB")
        chain_column = renamed.get("CHAIN")
        uniprot_column = renamed.get("SP_PRIMARY")
        if not all((pdb_column, chain_column, uniprot_column)):
            raise ValueError("unexpected SIFTS pdb_chain_uniprot columns")
        normalized = pd.DataFrame(
            {
                "pdb_id": chunk[pdb_column].astype(str).str.upper(),
                "chain_id": chunk[chain_column].astype(str),
                "uniprot_id": chunk[uniprot_column].astype(str),
            }
        )
        mask = [
            (pdb_id, chain_id) in missing
            for pdb_id, chain_id in zip(normalized.pdb_id, normalized.chain_id, strict=False)
        ]
        if any(mask):
            found.append(normalized.loc[mask])
    mapped = pd.concat(found, ignore_index=True) if found else pd.DataFrame()
    if mapped.empty:
        mapped = pd.DataFrame(columns=["pdb_id", "chain_id", "uniprot_id"])
    mapped["mapping_found"] = True
    mapped["fetch_status"] = "ok"
    found_pairs = set(zip(mapped.pdb_id, mapped.chain_id, strict=False))
    absent = pd.DataFrame(
        [
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "uniprot_id": None,
                "mapping_found": False,
                "fetch_status": "missing",
            }
            for pdb_id, chain_id in sorted(missing - found_pairs)
        ]
    )
    retained = cached.loc[~pd.MultiIndex.from_frame(cached[["pdb_id", "chain_id"]]).isin(missing)]
    return _merge_cache(retained, mapped, absent, columns=SIFTS_COLUMNS)


def fetch_uniprot(
    accessions: list[str],
    cached: pd.DataFrame,
    session: requests.Session,
    timeout: float,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch canonical UniProt records not already present in the local cache."""
    present = set(
        cached.loc[cached.fetch_status.isin({"ok", "missing"}), "primary_accession"]
        .dropna()
        .astype(str)
    )
    missing = accessions if refresh else sorted(set(accessions) - present)
    rows: list[dict[str, Any]] = []
    for accession in missing:
        response = session.get(UNIPROT_URL.format(accession=accession), timeout=timeout)
        if response.status_code == 404:
            rows.append(_missing_uniprot(accession))
            continue
        response.raise_for_status()
        value = response.json()
        description = value.get("proteinDescription") or {}
        rows.append(
            {
                "primary_accession": value.get("primaryAccession", accession),
                "uniprot_id": value.get("uniProtkbId"),
                "reviewed": value.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
                "protein_name": _protein_name(description),
                "sequence": (value.get("sequence") or {}).get("value"),
                "comments_json": json.dumps(value.get("comments") or []),
                "keywords_json": json.dumps(value.get("keywords") or []),
                "features_json": json.dumps(value.get("features") or []),
                "fetch_status": "ok",
            }
        )
    retained = cached.loc[~cached.primary_accession.astype(str).isin(missing)]
    return _merge_cache(retained, pd.DataFrame(rows), columns=UNIPROT_COLUMNS)


def fetch_rcsb(
    pdb_ids: list[str],
    cached: pd.DataFrame,
    session: requests.Session,
    timeout: float,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch entry metadata from the RCSB GraphQL endpoint in small batches."""
    present = set(
        cached.loc[cached.fetch_status.isin({"ok", "missing"}), "pdb_id"].dropna().astype(str)
    )
    missing = pdb_ids if refresh else sorted(set(pdb_ids) - present)
    rows: list[dict[str, Any]] = []
    for batch in _chunks(missing, 75):
        response = session.post(
            RCSB_GRAPHQL_URL,
            json={"query": RCSB_QUERY, "variables": {"entry_ids": batch}},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"RCSB GraphQL error: {payload['errors'][0].get('message')}")
        entries = (payload.get("data") or {}).get("entries") or []
        fetched = {str(entry.get("rcsb_id", "")).upper() for entry in entries if entry}
        rows.extend(_rcsb_row(entry) for entry in entries if entry)
        rows.extend(_missing_rcsb(pdb_id) for pdb_id in set(batch) - fetched)
    retained = cached.loc[~cached.pdb_id.astype(str).isin(missing)]
    return _merge_cache(retained, pd.DataFrame(rows), columns=RCSB_COLUMNS)


def label_positive_candidates(
    candidates: pd.DataFrame,
    sifts: pd.DataFrame,
    uniprot: pd.DataFrame,
    rcsb: pd.DataFrame,
    config: Mapping[str, Any],
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the positive-only DynamicMPNN rules without performing network I/O."""
    if config["labeling"].get("assign_negative_labels", False):
        raise ValueError("DynamicMPNN positive labeling cannot assign negative labels")
    _validate_candidate_columns(candidates)
    sifts_lookup = _sifts_lookup(sifts)
    uniprot_rows = [
        {str(key): value for key, value in row.items()} for row in uniprot.to_dict("records")
    ]
    rcsb_rows = [{str(key): value for key, value in row.items()} for row in rcsb.to_dict("records")]
    candidate_rows = [
        {str(key): value for key, value in row.items()} for row in candidates.to_dict("records")
    ]
    uniprot_lookup = {
        str(row["primary_accession"]): row
        for row in uniprot_rows
        if pd.notna(row.get("primary_accession"))
    }
    rcsb_lookup = {
        str(row["pdb_id"]).upper(): row for row in rcsb_rows if pd.notna(row.get("pdb_id"))
    }
    rows = [
        _label_row(row, sifts_lookup, uniprot_lookup, rcsb_lookup, config) for row in candidate_rows
    ]
    result = candidates.copy()
    labels = pd.DataFrame(rows, index=result.index)
    for column in labels:
        result[column] = labels[column]
    result = _apply_overrides(result, overrides, config)
    result["same_protein_identity_pass"] = result.same_protein_identity_pass.astype("boolean")
    result["single_structure_insufficient_derived"] = (
        result.single_structure_insufficient_derived.astype("boolean")
    )
    return result


def sample_label_audit(
    labeled: pd.DataFrame, n_per_group: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Return deterministic, explicitly grouped audit rows."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    groups = {
        confidence: labeled.label_confidence.eq(confidence)
        for confidence in ("gold", "silver", "bronze", "excluded")
    }
    groups["condition_aware"] = labeled.label_class.isin(CONDITION_CLASSES)
    groups["alternate_structured_state"] = labeled.label_class.isin(
        {"alternate_structured_state", "both_alternate_and_condition_aware"}
    )
    samples = []
    for offset, (name, mask) in enumerate(groups.items()):
        group = labeled.loc[mask]
        if group.empty:
            continue
        sample = group.sample(min(n_per_group, len(group)), random_state=seed + offset).copy()
        sample.insert(0, "audit_group", name)
        samples.append(sample)
    return pd.concat(samples, ignore_index=True) if samples else labeled.head(0).copy()


def _label_row(
    row: dict[str, Any],
    sifts: Mapping[tuple[str, str], set[str]],
    uniprot: Mapping[str, dict[str, Any]],
    rcsb: Mapping[str, dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    identity_method, identity_pass, uniprot_ids = _identity(row, sifts, config)
    structure = _structural_status(row, config["structure"])
    condition_type, metadata_complete, named_state, metadata_curated = _condition_status(
        row, uniprot_ids, uniprot, rcsb, config
    )
    # Titles and API metadata are useful evidence, but are not sufficient for
    # gold confidence.  Gold requires an explicit curated/manual annotation
    # supplied by the project (with provenance), preventing incidental words
    # such as ``open``/``closed`` from silently becoming gold labels.
    curated = row.get("curated_category") or row.get("trusted_category")
    exclusion = _hard_exclusion(row, identity_pass, structure, config)
    condition = condition_type is not None
    label_class = _positive_class(condition, named_state, structure)
    confidence = "silver"
    notes = [f"structural_status={structure}", f"identity={identity_method}"]
    if condition:
        notes.append(f"condition_signal={condition_type}")
    if curated:
        notes.append(f"curated_category={curated}")
    elif metadata_curated:
        notes.append(f"metadata_category={metadata_curated}")

    if exclusion:
        label_class, confidence = "excluded", "excluded"
    elif identity_pass is pd.NA or identity_pass is None:
        label_class, confidence = "ambiguous_structural_diversity", "bronze"
    elif structure in {"borderline", "unavailable"}:
        confidence = "bronze"
    elif curated:
        confidence = "gold"
    elif not metadata_complete:
        confidence = "bronze"

    training_ready = (
        confidence in _trainable_confidences(config)
        and label_class
        in {
            "alternate_structured_state",
            "condition_aware_structured_state",
            "both_alternate_and_condition_aware",
        }
        and exclusion is None
    )
    return {
        "label_class": label_class,
        "label_confidence": confidence,
        "single_structure_insufficient_derived": True if training_ready else pd.NA,
        "is_training_ready_positive": training_ready,
        "requires_manual_audit": not training_ready,
        "exclusion_reason": exclusion,
        "identity_check_method": identity_method,
        "same_protein_identity_pass": identity_pass,
        "uniprot_ids_json": json.dumps(sorted(uniprot_ids)),
        "has_condition_signal": condition,
        "condition_signal_type": condition_type,
        "label_notes": "; ".join(notes),
        "automatic_label_class": label_class,
        "automatic_label_confidence": confidence,
        "manual_override_applied": False,
        "manual_override_reason": None,
        "manual_override_source": None,
    }


def _identity(
    row: Mapping[str, Any],
    sifts: Mapping[tuple[str, str], set[str]],
    config: Mapping[str, Any],
) -> tuple[str, bool | Any, set[str]]:
    members = _json_list(row.get("cluster_members_json") or row.get("all_member_ids_json"))
    pairs = [parse_member_identifier(str(member)) for member in members]
    mapped = [sifts.get(pair, set()) if pair else set() for pair in pairs]
    uniprot_ids = set().union(*mapped) if mapped else set()
    if mapped and all(len(values) == 1 for values in mapped) and len(uniprot_ids) == 1:
        return "uniprot", True, uniprot_ids
    identity = _number(row.get("min_pairwise_sequence_identity"))
    if identity is None:
        return "unavailable", pd.NA, uniprot_ids
    if identity >= float(config["identity"]["min_sequence_identity"]):
        return "sequence_identity", True, uniprot_ids
    return "failed", False, uniprot_ids


def _structural_status(row: Mapping[str, Any], structure: Mapping[str, Any]) -> str:
    rmsd = _number(row.get("max_ca_rmsd"))
    tm = _number(row.get("min_pair_tm"))
    if (rmsd is not None and rmsd >= float(structure["strong_ca_rmsd"])) or (
        tm is not None and tm <= float(structure["strong_tm"])
    ):
        return "strong"
    if (rmsd is not None and rmsd >= float(structure["positive_ca_rmsd"])) or (
        tm is not None and tm <= float(structure["positive_tm"])
    ):
        return "positive"
    if rmsd is not None and rmsd >= float(structure["borderline_ca_rmsd"]):
        return "borderline"
    if (
        rmsd is not None
        and rmsd < float(structure["borderline_ca_rmsd"])
        and (tm is None or tm > float(structure["positive_tm"]))
    ):
        return "too_small"
    return "unavailable"


def _hard_exclusion(
    row: Mapping[str, Any],
    identity_pass: bool | Any,
    structure: str,
    config: Mapping[str, Any],
) -> str | None:
    source_reason = row.get("exclusion_reason")
    if source_reason is not None and not pd.isna(source_reason):
        return {
            "high_sequence_conflict": "sequence_conflict_too_high",
            "sequence_extraction_failed": "malformed_object",
        }.get(str(source_reason), str(source_reason))
    checks = (
        (_truth(row.get("load_failed")), "load_failed"),
        (_truth(row.get("malformed_object")), "malformed_object"),
        (
            float(_number(row.get("n_available_conformations"), 0) or 0) < 2,
            "not_enough_conformations",
        ),
        (
            float(_number(row.get("sequence_conflict_fraction"), 0) or 0)
            > float(config["identity"]["max_sequence_conflict_fraction"]),
            "sequence_conflict_too_high",
        ),
        (_truth(row.get("target_chain_ambiguous")), "ambiguous_target_chain"),
    )
    for failed, reason in checks:
        if failed:
            return reason
    coverage = _number(row.get("aligned_coverage"))
    if coverage is None:
        coverage = _number(row.get("min_aligned_coverage"))
    if coverage is not None and coverage < float(config["structure"]["min_aligned_coverage"]):
        return "aligned_coverage_below_threshold"
    if identity_pass is False:
        return "same_protein_identity_failed"
    if structure == "too_small":
        return "conformational_difference_below_threshold"
    return None


def _condition_status(
    row: Mapping[str, Any],
    uniprot_ids: set[str],
    uniprot: Mapping[str, dict[str, Any]],
    rcsb: Mapping[str, dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[str | None, bool, bool, str | None]:
    members = [
        parsed
        for member in _json_list(row.get("cluster_members_json") or row.get("all_member_ids_json"))
        if (parsed := parse_member_identifier(str(member))) is not None
    ]
    entries = [rcsb.get(pdb_id) for pdb_id, _ in members]
    metadata_complete = bool(entries) and all(
        entry and entry.get("fetch_status") == "ok" for entry in entries
    )
    # Keep the member/entry pairing intact.  Filtering missing entries before
    # pairing can assign a partner description to the wrong chain when an
    # intermediate RCSB record is unavailable.
    paired_values = [
        (entry, chain_id) for entry, (_, chain_id) in zip(entries, members, strict=False) if entry
    ]
    values = [entry for entry, _ in paired_values]
    titles = [str(entry.get("title") or "").lower() for entry in values]
    text = " ".join(titles)
    for accession in uniprot_ids:
        record = uniprot.get(accession)
        if record and record.get("fetch_status") == "ok":
            text += " " + str(record.get("protein_name") or "").lower()
    named_terms = (
        "open",
        "closed",
        "inward",
        "outward",
        "active",
        "inactive",
        "transporter",
        "metamorphic",
        "fold-switch",
        "fold switching",
    )
    named = any(term in text for term in named_terms)
    curated = _curated_category(text, titles)
    if not metadata_complete:
        return None, False, named, curated
    if any("apo" in title for title in titles) and any("holo" in title for title in titles):
        return "apo_holo", True, named, curated or "known_apo_holo"
    ignored = set(config["labeling"].get("ignored_ligands", ["HOH"]))
    ligand_sets = [set(_json_list(entry.get("ligands_json"))) - ignored for entry in values]
    if len({tuple(sorted(items)) for items in ligand_sets}) > 1:
        return "ligand", True, named, curated
    partner_sets = [_partner_descriptions(entry, chain_id) for entry, chain_id in paired_values]
    if len({tuple(sorted(items)) for items in partner_sets}) > 1:
        return "partner", True, named, curated
    assembly_counts = [
        tuple(sorted(_json_list(entry.get("assembly_instance_counts_json")))) for entry in values
    ]
    if len(set(assembly_counts)) > 1:
        return "oligomeric_state", True, named, curated
    entity_counts = [_number(entry.get("polymer_entity_count")) for entry in values]
    if len(set(entity_counts)) > 1:
        return "complex", True, named, curated
    return None, True, named, curated


def _positive_class(condition: bool, named: bool, structure: str) -> str:
    if condition and named:
        return "both_alternate_and_condition_aware"
    if condition:
        return "condition_aware_structured_state"
    if structure in {"positive", "strong"}:
        return "alternate_structured_state"
    return "ambiguous_structural_diversity"


def _curated_category(text: str, titles: list[str]) -> str | None:
    pairs = {
        "known_open_closed": ("open", "closed"),
        "known_inward_outward": ("inward", "outward"),
        "known_active_inactive": ("active", "inactive"),
        "known_apo_holo": ("apo", "holo"),
    }
    for name, (first, second) in pairs.items():
        if any(first in title for title in titles) and any(second in title for title in titles):
            return name
    if "metamorphic" in text or "fold-switch" in text or "fold switching" in text:
        return "known_fold_switching"
    return None


def _apply_overrides(
    result: pd.DataFrame,
    overrides: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if overrides is None or overrides.empty:
        return result
    required = {"dynamicmpnn_cluster_id", "label_class", "label_confidence", "reason", "source"}
    missing = required - set(overrides)
    if missing:
        raise ValueError(f"manual overrides are missing columns: {sorted(missing)}")
    if overrides.dynamicmpnn_cluster_id.astype(str).duplicated().any():
        raise ValueError("manual overrides contain duplicate cluster IDs")
    for override in overrides.itertuples(index=False):
        if override.label_class not in LABEL_CLASSES:
            raise ValueError(f"invalid override label_class: {override.label_class}")
        if override.label_confidence not in LABEL_CONFIDENCES:
            raise ValueError(f"invalid override label_confidence: {override.label_confidence}")
        if not str(override.reason).strip() or not str(override.source).strip():
            raise ValueError("manual overrides require reason and source")
        mask = result.dynamicmpnn_cluster_id.astype(str).eq(str(override.dynamicmpnn_cluster_id))
        if not mask.any():
            raise ValueError(
                f"manual override cluster not found: {override.dynamicmpnn_cluster_id}"
            )
        result.loc[mask, "label_class"] = override.label_class
        result.loc[mask, "label_confidence"] = override.label_confidence
        result.loc[mask, "manual_override_applied"] = True
        result.loc[mask, "manual_override_reason"] = str(override.reason)
        result.loc[mask, "manual_override_source"] = str(override.source)
        result.loc[mask, "label_notes"] += f"; manual_override={override.reason}"
    training = (
        result.label_confidence.isin(_trainable_confidences(config))
        & result.label_class.isin(
            {
                "alternate_structured_state",
                "condition_aware_structured_state",
                "both_alternate_and_condition_aware",
            }
        )
        & result.exclusion_reason.isna()
    )
    result["is_training_ready_positive"] = training
    result["requires_manual_audit"] = ~training
    result["single_structure_insufficient_derived"] = pd.array(
        [True if ready else pd.NA for ready in training], dtype="boolean"
    )
    return result


def _candidate_pairs(candidates: pd.DataFrame) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    member_column = (
        "cluster_members_json" if "cluster_members_json" in candidates else "all_member_ids_json"
    )
    for value in candidates[member_column]:
        for member in _json_list(value):
            parsed = parse_member_identifier(str(member))
            if parsed:
                pairs.add(parsed)
    return pairs


def _sifts_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    lookup: dict[tuple[str, str], set[str]] = {}
    for row in frame.itertuples(index=False):
        if _truth(row.mapping_found) and pd.notna(row.uniprot_id):
            lookup.setdefault((str(row.pdb_id).upper(), str(row.chain_id)), set()).add(
                str(row.uniprot_id)
            )
    return lookup


def _partner_descriptions(entry: Mapping[str, Any], target_chain: str) -> set[str]:
    partners = set()
    for entity in _json_list(entry.get("polymer_entities_json")):
        if not isinstance(entity, dict):
            continue
        chains = {str(value) for value in entity.get("auth_asym_ids") or []}
        if target_chain not in chains:
            description = str(entity.get("description") or "").strip().lower()
            if description:
                partners.add(description)
    return partners


def _rcsb_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    info = entry.get("rcsb_entry_info") or {}
    polymers = [
        {
            "auth_asym_ids": (item.get("rcsb_polymer_entity_container_identifiers") or {}).get(
                "auth_asym_ids"
            )
            or [],
            "description": (item.get("rcsb_polymer_entity") or {}).get("pdbx_description"),
        }
        for item in entry.get("polymer_entities") or []
    ]
    ligands = sorted(
        {
            str(identifier)
            for item in entry.get("nonpolymer_entities") or []
            if (
                identifier := ((item.get("nonpolymer_comp") or {}).get("chem_comp") or {}).get("id")
            )
        }
    )
    assemblies = [
        (item.get("rcsb_assembly_info") or {}).get("polymer_entity_instance_count")
        for item in entry.get("assemblies") or []
    ]
    resolution = info.get("resolution_combined") or []
    return {
        "pdb_id": str(entry.get("rcsb_id", "")).upper(),
        "title": (entry.get("struct") or {}).get("title"),
        "experimental_method": "; ".join(
            str(item.get("method")) for item in entry.get("exptl") or [] if item.get("method")
        ),
        "resolution": min(resolution) if resolution else None,
        "polymer_entity_count": info.get("polymer_entity_count"),
        "assembly_count": info.get("assembly_count"),
        "ligands_json": json.dumps(ligands),
        "polymer_entities_json": json.dumps(polymers),
        "assembly_instance_counts_json": json.dumps(
            sorted(value for value in assemblies if value is not None)
        ),
        "fetch_status": "ok",
    }


def _missing_uniprot(accession: str) -> dict[str, Any]:
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


def _missing_rcsb(pdb_id: str) -> dict[str, Any]:
    return {
        "pdb_id": pdb_id,
        "title": None,
        "experimental_method": None,
        "resolution": None,
        "polymer_entity_count": None,
        "assembly_count": None,
        "ligands_json": "[]",
        "polymer_entities_json": "[]",
        "assembly_instance_counts_json": "[]",
        "fetch_status": "missing",
    }


def _record_sifts_errors(
    cached: pd.DataFrame, pairs: list[tuple[str, str]], error: Exception
) -> pd.DataFrame:
    terminal = cached.fetch_status.isin({"ok", "missing"})
    present = set(
        zip(
            cached.loc[terminal, "pdb_id"].astype(str),
            cached.loc[terminal, "chain_id"].astype(str),
            strict=False,
        )
    )
    retryable = set(pairs) - present
    retained = cached.loc[
        ~(
            cached.fetch_status.astype(str).str.startswith("error:")
            & pd.MultiIndex.from_frame(cached[["pdb_id", "chain_id"]]).isin(retryable)
        )
    ]
    rows = [
        {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "uniprot_id": None,
            "mapping_found": False,
            "fetch_status": f"error:{type(error).__name__}",
        }
        for pdb_id, chain_id in sorted(retryable)
    ]
    return _merge_cache(retained, pd.DataFrame(rows), columns=SIFTS_COLUMNS)


def _record_uniprot_errors(
    cached: pd.DataFrame, accessions: list[str], error: Exception
) -> pd.DataFrame:
    terminal = cached.fetch_status.isin({"ok", "missing"})
    present = set(cached.loc[terminal, "primary_accession"].dropna().astype(str))
    retryable = set(accessions) - present
    retained = cached.loc[
        ~(
            cached.fetch_status.astype(str).str.startswith("error:")
            & cached.primary_accession.astype(str).isin(retryable)
        )
    ]
    rows = [
        {
            **_missing_uniprot(accession),
            "fetch_status": f"error:{type(error).__name__}",
        }
        for accession in sorted(retryable)
    ]
    return _merge_cache(retained, pd.DataFrame(rows), columns=UNIPROT_COLUMNS)


def _record_rcsb_errors(cached: pd.DataFrame, pdb_ids: list[str], error: Exception) -> pd.DataFrame:
    terminal = cached.fetch_status.isin({"ok", "missing"})
    present = set(cached.loc[terminal, "pdb_id"].dropna().astype(str))
    retryable = set(pdb_ids) - present
    retained = cached.loc[
        ~(
            cached.fetch_status.astype(str).str.startswith("error:")
            & cached.pdb_id.astype(str).isin(retryable)
        )
    ]
    rows = [
        {
            **_missing_rcsb(pdb_id),
            "fetch_status": f"error:{type(error).__name__}",
        }
        for pdb_id in sorted(retryable)
    ]
    return _merge_cache(retained, pd.DataFrame(rows), columns=RCSB_COLUMNS)


def _protein_name(description: Mapping[str, Any]) -> str | None:
    recommended = description.get("recommendedName") or {}
    full = recommended.get("fullName") or {}
    if full.get("value"):
        return str(full["value"])
    submissions = description.get("submissionNames") or []
    if submissions:
        return str((submissions[0].get("fullName") or {}).get("value") or "") or None
    return None


def _validate_candidate_columns(frame: pd.DataFrame) -> None:
    required = {
        "dynamicmpnn_cluster_id",
        "n_available_conformations",
        "sequence_conflict_fraction",
        "max_ca_rmsd",
        "min_pair_tm",
        "load_failed",
    }
    if not ({"cluster_members_json", "all_member_ids_json"} & set(frame)):
        required.add("cluster_members_json")
    missing = required - set(frame)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")


def _http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "protein-state-router/0.1 DynamicMPNN labeling"
    return session


def _read_cache(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=list(columns))
    return _normalize_cache(pd.read_parquet(path), columns)


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("w") as handle:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)


def _normalize_cache(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = None
    return result.loc[:, list(columns)].reset_index(drop=True)


def _merge_cache(
    retained: pd.DataFrame, *additions: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    """Combine cache rows without concatenating empty schema-only frames."""
    frames = [frame for frame in (retained, *additions) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))
    if len(frames) == 1:
        return _normalize_cache(frames[0], columns)
    return _normalize_cache(pd.concat(frames, ignore_index=True), columns)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or value is pd.NA or pd.isna(value):
        return default
    return float(value)


def _truth(value: Any) -> bool:
    return False if value is None or value is pd.NA or pd.isna(value) else bool(value)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _trainable_confidences(config: Mapping[str, Any]) -> set[str]:
    labeling = config["labeling"]
    return {
        confidence
        for confidence in ("gold", "silver", "bronze")
        if labeling.get(f"train_on_{confidence}", confidence != "bronze")
    }
