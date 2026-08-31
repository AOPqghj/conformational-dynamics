"""Localize frozen SAE feature hotspots to experimentally resolved structure context.

The frozen SAE was fit on Seed-42 training proteins only.  This workflow is
strictly interpretive: it scores the frozen test partition, selects residue
hotspots and low-activation controls, and obtains structural properties from
cached PDB/mmCIF biological assemblies. Raw coordinates are never retained in
outputs, while the shared download cache enables reproducible reruns.
"""

# ruff: noqa: E402 - direct execution needs repository imports after environment setup.

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/protein-state-router-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/protein-state-router-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from scipy.spatial.distance import cdist

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from protein_state_router.external.structure_geometry import parse_mmcif_ca_chains
from protein_state_router.interpretability.prs import map_structure_to_canonical
from scripts.analyze_transition_residue_displacements import parse_structure_id

from interpretability.analyze_sae_transition_residue_associations import (
    load_frozen_sae,
    sha256_file,
)
from interpretability.contracts import load_residue_matrix

DEFAULT_CATALOG = Path("ml/results/homology35_rerun/pooled_frozen_models/seed_42_catalog.parquet")
DEFAULT_FULL_CATALOG = Path("data/lifecycle/final/initial_8598_dataset/catalog.parquet")
DEFAULT_TRANSITION_CATALOG = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/transition_important_proteins.csv"
)
DEFAULT_TRANSITION_SUMMARY = Path(
    "interpretability/results/homology35_rerun/transition_pairs/pair_summary.csv"
)
DEFAULT_ASSOCIATIONS = Path(
    "interpretability/results/homology35_rerun/sae_transition_associations/"
    "sae_feature_associations.csv"
)
DEFAULT_SAE = Path("ml/results/homology35_rerun/frozen_saes/esmfold_matrix_topk64_seed42")
DEFAULT_OUTPUT = Path("interpretability/results/homology35_rerun/sae_structural_roles")
# Keep raw assemblies outside run outputs so ESMFold and BioEMU analyses share
# the same RCSB downloads.  Override with --rcsb-cache for a different volume.
DEFAULT_RCSB_CACHE = REPOSITORY_ROOT / "data/cache/rcsb_biological_assemblies"
STRUCTURAL_AUDIT_TERMINAL_STATUSES = frozenset({"complete", "esmfold_single_fallback"})

SASA_MAX = {
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

# These deliberately follow broad side-chain chemistry classes rather than a
# force-field.  Aromatic is retained as an overlapping annotation because the
# Gunasekaran and Nussinov comparison reports aromatic--aromatic contacts
# separately from the hydrophobic interaction class.
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP", "PRO"})
POLAR_RESIDUES = frozenset({"SER", "THR", "ASN", "GLN", "CYS", "ASP", "GLU", "LYS", "ARG", "HIS"})
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

# Heavy-atom donor/acceptor definitions are intentionally conservative.  They
# identify residue-pair hydrogen-bond opportunities from experimentally solved
# structures without relying on unobserved hydrogen coordinates.
DONOR_ATOMS = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "ASN": frozenset({"ND2"}),
    "GLN": frozenset({"NE2"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "LYS": frozenset({"NZ"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
    "TRP": frozenset({"NE1"}),
    "TYR": frozenset({"OH"}),
}
ACCEPTOR_ATOMS = {
    "ASP": frozenset({"OD1", "OD2"}),
    "ASN": frozenset({"OD1"}),
    "GLU": frozenset({"OE1", "OE2"}),
    "GLN": frozenset({"OE1"}),
    "HIS": frozenset({"ND1", "NE2"}),
    "SER": frozenset({"OG"}),
    "THR": frozenset({"OG1"}),
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    features_per_track: int = 10
    hotspots_per_protein: int = 5
    controls_per_protein: int = 5
    low_activation_quantile: float = 0.5
    sphere_radius_angstrom: float = 8.0
    contact_cutoff_angstrom: float = 4.5
    download_batch_size: int = 25
    device: str = "auto"
    enable_esmfold_fallback: bool = False
    partition: str = "test"

    @property
    def config_hash(self) -> str:
        values = asdict(self)
        # Preserve the established test-run checkpoint hash; validation is a new
        # explicitly partitioned analysis and therefore receives a distinct hash.
        if self.partition == "test":
            values.pop("partition")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ContactInteractionConfig:
    """Configuration for the post-hoc contact chemistry extension."""

    seed: int = 42
    contact_cutoff_angstrom: float = 4.5
    hbond_cutoff_angstrom: float = 3.5
    permutations: int = 10_000

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def status(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; use --device auto or --device cpu")
    return torch.device(device)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def is_completed_structure_audit(row: dict[str, object]) -> bool:
    """Whether an audit row represents a successful, non-retryable context."""
    return str(row.get("status", "")) in STRUCTURAL_AUDIT_TERMINAL_STATUSES


def json_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in result] if isinstance(result, list) else []


def select_feature_tracks(associations: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select independent RMSD and PRS tracks, retaining overlap provenance."""
    required = {"feature_id", "displacement_balanced_spearman", "prs_balanced_spearman"}
    if missing := required - set(associations):
        raise ValueError(f"association table missing columns: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    for target, effect, fdr in (
        ("rmsd_displacement", "displacement_balanced_spearman", "displacement_fdr"),
        ("prs", "prs_balanced_spearman", "prs_fdr"),
    ):
        ranked = associations.copy()
        ranked["absolute_effect"] = ranked[effect].abs()
        ranked["_fdr"] = ranked[fdr] if fdr in ranked else 1.0
        ranked = ranked.sort_values(
            ["absolute_effect", "_fdr", "feature_id"], ascending=[False, True, True]
        ).head(count)
        ranked = ranked.assign(selection_track=target, selection_rank=np.arange(1, len(ranked) + 1))
        rows.append(
            ranked[["feature_id", "selection_track", "selection_rank", "absolute_effect", "_fdr"]]
        )
    selected = pd.concat(rows, ignore_index=True)
    grouped = (
        selected.groupby("feature_id", as_index=False)
        .agg(
            selection_tracks=("selection_track", lambda values: ";".join(sorted(values))),
            selection_ranks=("selection_rank", lambda values: ";".join(map(str, sorted(values)))),
            selection_absolute_effects=(
                "absolute_effect",
                lambda values: ";".join(f"{float(value):.8g}" for value in values),
            ),
            selection_fdrs=(
                "_fdr",
                lambda values: ";".join(f"{float(value):.8g}" for value in values),
            ),
        )
        .sort_values("feature_id")
        .reset_index(drop=True)
    )
    return grouped


def load_selected_features(
    association_path: Path, features_per_track: int, selected_features_path: Path | None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze an explicit feature set, or reproduce the established track selection."""
    if selected_features_path is None:
        selected = select_feature_tracks(pd.read_csv(association_path), features_per_track)
        return selected, {
            "selection_source": "association_ranking",
            "association_path": str(association_path),
            "association_sha256": sha256_file(association_path),
        }
    selected = pd.read_csv(selected_features_path)
    required = {"feature_id", "selection_tracks"}
    if missing := required - set(selected):
        raise ValueError(f"selected feature file missing columns: {sorted(missing)}")
    selected = selected.copy()
    selected["feature_id"] = selected.feature_id.astype(int)
    if selected.feature_id.duplicated().any() or selected.empty:
        raise ValueError("selected feature file must contain non-empty unique feature IDs")
    return selected, {
        "selection_source": "frozen_selected_features",
        "selected_features_path": str(selected_features_path),
        "selected_features_sha256": sha256_file(selected_features_path),
    }


def stable_rng(seed: int, protein_id: str, feature_id: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{protein_id}|{feature_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def choose_positions(
    activations: np.ndarray,
    protein_id: str,
    feature_id: int,
    config: Config,
    sequence: str | None = None,
) -> list[dict[str, object]]:
    """Return hotspots and low-activation controls matched on residue and position."""
    if activations.ndim != 1 or len(activations) < 2:
        raise ValueError("activation vector is too short for hotspot/control selection")
    if sequence is not None and len(sequence) != len(activations):
        raise ValueError("sequence and activation vector lengths differ")
    indices = np.arange(len(activations))
    order = np.lexsort((indices, -activations))
    hotspot_count = min(config.hotspots_per_protein, max(1, len(activations) // 2))
    hotspot = order[:hotspot_count]
    cutoff = float(np.quantile(activations, config.low_activation_quantile, method="higher"))
    eligible = indices[(activations <= cutoff) & ~np.isin(indices, hotspot)]
    if len(eligible) < config.controls_per_protein:
        eligible = indices[~np.isin(indices, hotspot)]
    control_count = min(config.controls_per_protein, len(eligible))
    if control_count < 1:
        raise ValueError("activation vector has no non-hotspot residue for a control")
    rng = stable_rng(config.seed, protein_id, feature_id)
    controls: list[int] = []
    control_matches: dict[int, tuple[int, bool]] = {}
    remaining = set(int(value) for value in eligible)
    for hotspot_index in hotspot[:control_count]:
        same_amino_acid = (
            [
                value
                for value in remaining
                if sequence and sequence[value] == sequence[hotspot_index]
            ]
            if sequence
            else []
        )
        pool = same_amino_acid or list(remaining)
        if not pool:
            break
        distances = np.asarray([abs(value - int(hotspot_index)) for value in pool])
        nearest = [
            value
            for value, distance in zip(pool, distances, strict=True)
            if distance == distances.min()
        ]
        selected_control = int(rng.choice(nearest))
        controls.append(selected_control)
        remaining.remove(selected_control)
        control_matches[selected_control] = (
            int(hotspot_index),
            bool(sequence and sequence[selected_control] == sequence[hotspot_index]),
        )
    ranks = np.empty(len(activations), dtype=np.int64)
    ranks[order] = np.arange(1, len(activations) + 1)
    rows: list[dict[str, object]] = []
    for kind, positions in (("hotspot", hotspot), ("low_activation_control", controls)):
        for index in positions:
            matched, amino_acid_matched = control_matches.get(
                int(index), (int(index), True if sequence is not None else None)
            )
            rows.append(
                {
                    "residue_index": int(index),
                    "selection_kind": kind,
                    "activation": float(activations[index]),
                    "activation_rank": int(ranks[index]),
                    "activation_quantile": float(
                        (len(activations) - ranks[index] + 1) / len(activations)
                    ),
                    "matched_hotspot_residue_index": int(matched),
                    "control_amino_acid_matched": amino_acid_matched,
                    "control_sequence_distance": abs(int(index) - int(matched)),
                }
            )
    return rows


def activation_vector(
    matrix: np.ndarray,
    feature_id: int,
    model: torch.nn.Module,
    center: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(matrix), 512):
        batch = torch.from_numpy(
            matrix[start : start + 512].astype(np.float32, copy=False) - center
        ).to(device)
        with torch.inference_mode():
            encoded = model.encode(batch)[:, feature_id]
        values.append(encoded.detach().cpu().numpy())
    return np.concatenate(values).astype(np.float32, copy=False)


def require_structure_dependencies() -> tuple[Any, Any, Any]:
    try:
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except ImportError as error:  # pragma: no cover - exercised in user environment.
        raise RuntimeError("install declared structural dependencies with `uv sync`") from error
    return MMCIFParser, PDBParser, ShrakeRupley


def parse_full_structure(path: Path, structure_id: str) -> Any:
    MMCIFParser, _, _ = require_structure_dependencies()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return MMCIFParser(QUIET=True).get_structure(structure_id, handle)


def protein_residues(model: Any) -> list[Any]:
    result = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or "CA" not in residue:
                continue
            if residue.get_resname().upper() not in SASA_MAX:
                continue
            result.append(residue)
    return result


def residue_key(residue: Any) -> tuple[str, str, str]:
    insertion = str(residue.id[2]).strip()
    return str(residue.get_parent().id), str(residue.id[1]), insertion


def mapped_residue_lookup(mapped: Any, model: Any) -> dict[int, Any]:
    by_key = {residue_key(residue): residue for residue in protein_residues(model)}
    lookup: dict[int, Any] = {}
    for label, auth, insertion in zip(
        mapped.label_residue_numbers,
        mapped.auth_residue_numbers,
        mapped.insertion_codes,
        strict=True,
    ):
        key = (str(mapped.chain_id), str(auth), str(insertion or ""))
        if key in by_key:
            lookup[int(label)] = by_key[key]
    return lookup


def residue_contacts(
    center: Any, residues: Iterable[Any], sphere_radius: float, contact_cutoff: float
) -> dict[str, object]:
    center_ca = center["CA"].coord
    center_chain, center_number, _ = residue_key(center)
    center_atoms = np.vstack(
        [atom.coord for atom in center.get_atoms() if str(atom.element).upper() != "H"]
    )
    neighbors = []
    contacts = []
    for residue in residues:
        if residue is center:
            continue
        chain, number, insertion = residue_key(residue)
        if chain == center_chain:
            try:
                if abs(int(number) - int(center_number)) <= 2:
                    continue
            except ValueError:
                pass
        distance = float(np.linalg.norm(residue["CA"].coord - center_ca))
        identifier = f"{chain}:{number}{insertion}:{residue.get_resname().upper()}"
        if distance <= sphere_radius:
            neighbors.append((identifier, residue.get_resname().upper(), distance))
        if distance <= sphere_radius:
            atoms = np.vstack(
                [atom.coord for atom in residue.get_atoms() if str(atom.element).upper() != "H"]
            )
            if len(atoms) and float(cdist(center_atoms, atoms).min()) <= contact_cutoff:
                contacts.append(identifier)
    composition: dict[str, int] = {}
    for _, residue_name, _ in neighbors:
        composition[residue_name] = composition.get(residue_name, 0) + 1
    return {
        "sphere_neighbor_count": len(neighbors),
        "sphere_neighbor_composition_json": json.dumps(composition, sort_keys=True),
        "contact_density": len(contacts),
        "contact_ids_json": json.dumps(sorted(contacts)),
    }


def sidechain_atoms(residue: Any) -> list[Any]:
    """Return resolved, non-hydrogen side-chain atoms for a standard residue."""
    return [
        atom
        for atom in residue.get_atoms()
        if str(atom.element).upper() != "H" and str(atom.get_name()).upper() not in BACKBONE_ATOMS
    ]


def interaction_class_counts(center_name: str, partner_names: Iterable[str]) -> dict[str, int]:
    """Classify side-chain contact partners for one selected residue.

    Hydrophobic--hydrophobic and polar--polar/hydrophobic--polar are mutually
    exclusive broad chemistry counts. Aromatic--aromatic is a separate,
    overlapping annotation, as in the motivating ligand-binding study.
    """
    counts = {
        "polar_polar_contact_count": 0,
        "hydrophobic_hydrophobic_contact_count": 0,
        "hydrophobic_polar_contact_count": 0,
        "aromatic_aromatic_contact_count": 0,
        "polar_partner_contact_count": 0,
        "nonpolar_partner_contact_count": 0,
    }
    center_is_polar = center_name in POLAR_RESIDUES
    center_is_hydrophobic = center_name in HYDROPHOBIC_RESIDUES
    center_is_aromatic = center_name in AROMATIC_RESIDUES
    for partner_name in partner_names:
        partner_is_polar = partner_name in POLAR_RESIDUES
        partner_is_hydrophobic = partner_name in HYDROPHOBIC_RESIDUES
        if partner_is_polar:
            counts["polar_partner_contact_count"] += 1
        if partner_is_hydrophobic:
            counts["nonpolar_partner_contact_count"] += 1
        if center_is_polar and partner_is_polar:
            counts["polar_polar_contact_count"] += 1
        elif center_is_hydrophobic and partner_is_hydrophobic:
            counts["hydrophobic_hydrophobic_contact_count"] += 1
        elif (center_is_hydrophobic and partner_is_polar) or (
            center_is_polar and partner_is_hydrophobic
        ):
            counts["hydrophobic_polar_contact_count"] += 1
        if center_is_aromatic and partner_name in AROMATIC_RESIDUES:
            counts["aromatic_aromatic_contact_count"] += 1
    return counts


def donor_acceptor_atoms(residue: Any) -> tuple[list[Any], list[Any]]:
    """Return donor and acceptor heavy atoms, including backbone donors/acceptors."""
    residue_name = residue.get_resname().upper()
    donors, acceptors = [], []
    for atom in residue.get_atoms():
        if str(atom.element).upper() == "H":
            continue
        name = str(atom.get_name()).upper()
        if name == "N" and residue_name != "PRO":
            donors.append(atom)
        elif name in DONOR_ATOMS.get(residue_name, frozenset()):
            donors.append(atom)
        if name in {"O", "OXT"} or name in ACCEPTOR_ATOMS.get(residue_name, frozenset()):
            acceptors.append(atom)
    return donors, acceptors


def has_hydrogen_bond(center: Any, partner: Any, cutoff: float) -> bool:
    """Approximate a residue-pair H bond from donor/acceptor heavy-atom distance."""
    center_donors, center_acceptors = donor_acceptor_atoms(center)
    partner_donors, partner_acceptors = donor_acceptor_atoms(partner)
    for donors, acceptors in (
        (center_donors, partner_acceptors),
        (partner_donors, center_acceptors),
    ):
        if not donors or not acceptors:
            continue
        donor_coordinates = np.vstack([atom.coord for atom in donors])
        acceptor_coordinates = np.vstack([atom.coord for atom in acceptors])
        if float(cdist(donor_coordinates, acceptor_coordinates).min()) <= cutoff:
            return True
    return False


def residue_interactions(
    center: Any, residues: Iterable[Any], contact_cutoff: float, hbond_cutoff: float
) -> dict[str, object]:
    """Measure normalized chemistry of nonlocal contacts around one residue."""
    center_chain, center_number, _ = residue_key(center)
    center_all_atoms = [atom for atom in center.get_atoms() if str(atom.element).upper() != "H"]
    center_sidechain = sidechain_atoms(center)
    all_contact_partners: list[Any] = []
    sidechain_contact_partners: list[Any] = []
    hbond_partners: list[Any] = []
    if not center_all_atoms:
        return {}
    center_all_coordinates = np.vstack([atom.coord for atom in center_all_atoms])
    for partner in residues:
        if partner is center:
            continue
        chain, number, _ = residue_key(partner)
        if chain == center_chain:
            try:
                if abs(int(number) - int(center_number)) <= 2:
                    continue
            except ValueError:
                pass
        partner_all_atoms = [
            atom for atom in partner.get_atoms() if str(atom.element).upper() != "H"
        ]
        if not partner_all_atoms:
            continue
        partner_all_coordinates = np.vstack([atom.coord for atom in partner_all_atoms])
        if float(cdist(center_all_coordinates, partner_all_coordinates).min()) > contact_cutoff:
            continue
        all_contact_partners.append(partner)
        if has_hydrogen_bond(center, partner, hbond_cutoff):
            hbond_partners.append(partner)
        partner_sidechain = sidechain_atoms(partner)
        if center_sidechain and partner_sidechain:
            center_sidechain_coordinates = np.vstack([atom.coord for atom in center_sidechain])
            partner_sidechain_coordinates = np.vstack([atom.coord for atom in partner_sidechain])
            if (
                float(cdist(center_sidechain_coordinates, partner_sidechain_coordinates).min())
                <= contact_cutoff
            ):
                sidechain_contact_partners.append(partner)
    total_contacts = len(all_contact_partners)
    sidechain_contacts = len(sidechain_contact_partners)
    chemistry = interaction_class_counts(
        center.get_resname().upper(),
        [partner.get_resname().upper() for partner in sidechain_contact_partners],
    )
    result: dict[str, object] = {
        "protein_contact_count": total_contacts,
        "sidechain_contact_count": sidechain_contacts,
        "hydrogen_bond_partner_count": len(hbond_partners),
        "hydrogen_bond_contact_fraction": len(hbond_partners) / total_contacts
        if total_contacts
        else np.nan,
    }
    for name, value in chemistry.items():
        result[name] = value
        result[name.replace("_count", "_fraction")] = (
            value / sidechain_contacts if sidechain_contacts else np.nan
        )
    return result


def interaction_properties(
    path: Path,
    structure_id: str,
    mapped: Any,
    positions: set[int],
    config: ContactInteractionConfig,
) -> dict[int, dict[str, object]]:
    """Map canonical positions and calculate protein-residue contact chemistry."""
    structure = parse_full_structure(path, structure_id)
    model = next(structure.get_models())
    residues = protein_residues(model)
    lookup = mapped_residue_lookup(mapped, model)
    canonical_index = {position: index for index, position in enumerate(mapped.canonical_positions)}
    result: dict[int, dict[str, object]] = {}
    for canonical_position in positions:
        label_index = canonical_index.get(canonical_position)
        if label_index is None:
            continue
        residue = lookup.get(int(mapped.label_residue_numbers[label_index]))
        if residue is None:
            continue
        result[canonical_position] = residue_interactions(
            residue, residues, config.contact_cutoff_angstrom, config.hbond_cutoff_angstrom
        )
    return result


def sse_for_model(
    path: Path | None, model: Any
) -> tuple[dict[tuple[str, str, str], str], str, str]:
    """Use PDBx SSE records first, then the CA-based P-SEA fallback."""
    result = {residue_key(residue): "C" for residue in protein_residues(model)}
    method = "psea_geometry"
    error = ""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx

        for chain in model:
            chain_residues = [
                residue for residue in chain if residue.id[0] == " " and "CA" in residue
            ]
            chain_id = str(chain.id)
            # P-SEA fallback avoids requiring a system DSSP binary.
            atoms = [residue["CA"].coord for residue in chain_residues]
            if len(atoms) >= 3:
                array = struc.AtomArray(len(atoms))
                array.coord = np.asarray(atoms)
                array.chain_id = np.asarray([chain_id] * len(atoms))
                array.res_id = np.arange(1, len(atoms) + 1)
                array.res_name = np.asarray([residue.get_resname() for residue in chain_residues])
                array.atom_name = np.asarray(["CA"] * len(atoms))
                values = struc.annotate_sse(array)
                for residue, value in zip(chain_residues, values, strict=True):
                    result[residue_key(residue)] = {"a": "H", "b": "E", "c": "C"}.get(
                        str(value), "C"
                    )
        if path is not None:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                cif = pdbx.CIFFile.read(handle)
            annotation = pdbx.get_sse(cif)
            method = "pdbx_annotation_with_psea_fallback"
            for chain in model:
                chain_residues = [
                    residue for residue in chain if residue.id[0] == " " and "CA" in residue
                ]
                values = annotation.get(str(chain.id))
                if values is not None and len(values) == len(chain_residues):
                    for residue, value in zip(chain_residues, values, strict=True):
                        result[residue_key(residue)] = {"a": "H", "b": "E", "c": "C"}.get(
                            str(value), "C"
                        )
    except Exception as caught:
        error = str(caught)
    return result, method, error


def structural_properties(
    path: Path, structure_id: str, mapped: Any, positions: set[int], config: Config
) -> dict[int, dict[str, object]]:
    _, _, ShrakeRupley = require_structure_dependencies()
    structure = parse_full_structure(path, structure_id)
    model = next(structure.get_models())
    residues = protein_residues(model)
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(model, level="R")
    sse, sse_method, sse_error = sse_for_model(path, model)
    lookup = mapped_residue_lookup(mapped, model)
    result: dict[int, dict[str, object]] = {}
    canonical_index = {position: index for index, position in enumerate(mapped.canonical_positions)}
    for canonical_position in positions:
        label_index = canonical_index.get(canonical_position)
        if label_index is None:
            continue
        label_number = mapped.label_residue_numbers[label_index]
        residue = lookup.get(int(label_number))
        if residue is None:
            continue
        name = residue.get_resname().upper()
        sasa = float(getattr(residue, "sasa", np.nan))
        result[canonical_position] = {
            "structure_id": structure_id,
            "structure_chain_id": str(mapped.chain_id),
            "structure_auth_chain_id": str(mapped.chain_id),
            "secondary_structure": sse.get(residue_key(residue), "C"),
            "secondary_structure_method": sse_method,
            "secondary_structure_error": sse_error,
            "structure_context": "biological_assembly_1",
            "sasa_angstrom2": sasa,
            "relative_sasa": sasa / SASA_MAX[name]
            if name in SASA_MAX and np.isfinite(sasa)
            else np.nan,
            **residue_contacts(
                residue,
                residues,
                sphere_radius=config.sphere_radius_angstrom,
                contact_cutoff=config.contact_cutoff_angstrom,
            ),
        }
    return result


def esmfold_fallback_properties(
    sequence: str,
    positions: set[int],
    root: Path,
    device: torch.device,
    config: Config,
) -> dict[int, dict[str, object]]:
    """Predict one fallback structure only after experimental resolution fails."""
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
    except ImportError as error:  # pragma: no cover - depends on optional runtime extra.
        raise RuntimeError("ESMFold fallback requires `uv sync --extra esmfold`") from error
    _, PDBParser, ShrakeRupley = require_structure_dependencies()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    folding_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()
    tokens = {
        name: value.to(device)
        for name, value in tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).items()
    }
    with torch.inference_mode():
        outputs = folding_model(**tokens, num_recycles=0)
    atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs).detach().cpu().numpy()
    output_arrays = {
        name: value.detach().cpu().numpy()
        for name, value in outputs.items()
        if torch.is_tensor(value)
    }
    predicted = Protein(
        aatype=output_arrays["aatype"][0],
        atom_positions=atom_positions[0],
        atom_mask=output_arrays["atom37_atom_exists"][0],
        residue_index=output_arrays["residue_index"][0] + 1,
        b_factors=output_arrays["plddt"][0],
        chain_index=output_arrays.get("chain_index", [None])[0],
    )
    pdb_path = root / "esmfold_single.pdb"
    pdb_path.write_text(to_pdb(predicted), encoding="utf-8")
    model = next(PDBParser(QUIET=True).get_structure("esmfold_single", pdb_path).get_models())
    residues = protein_residues(model)
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(model, level="R")
    sse, sse_method, sse_error = sse_for_model(None, model)
    result: dict[int, dict[str, object]] = {}
    for position in positions:
        if not 1 <= position <= len(residues):
            continue
        residue = residues[position - 1]
        name = residue.get_resname().upper()
        sasa = float(getattr(residue, "sasa", np.nan))
        result[position] = {
            "structure_id": "esmfold_single",
            "structure_chain_id": str(residue.get_parent().id),
            "structure_auth_chain_id": str(residue.get_parent().id),
            "secondary_structure": sse.get(residue_key(residue), "C"),
            "secondary_structure_method": sse_method,
            "secondary_structure_error": sse_error,
            "structure_context": "predicted_monomer",
            "sasa_angstrom2": sasa,
            "relative_sasa": sasa / SASA_MAX[name]
            if name in SASA_MAX and np.isfinite(sasa)
            else np.nan,
            **residue_contacts(
                residue,
                residues,
                sphere_radius=config.sphere_radius_angstrom,
                contact_cutoff=config.contact_cutoff_angstrom,
            ),
        }
    return result


def resolve_mapped_structure(structure_id: str, path: Path, sequence: str) -> Any:
    """Map the canonical sequence directly onto the coordinate file being measured."""
    pdb_code, preferred_chains = parse_structure_id(structure_id)
    candidates = []
    for chain in parse_mmcif_ca_chains(path, structure_id):
        try:
            mapped = map_structure_to_canonical(
                chain, sequence, minimum_identity=0.9, minimum_coverage=0.5
            )
        except ValueError:
            continue
        preference = min(
            (
                index
                for index, value in enumerate(preferred_chains)
                if value in {chain.chain_id, chain.auth_chain_id}
            ),
            default=len(preferred_chains),
        )
        candidates.append(
            (preference, -mapped.sequence_identity, -mapped.canonical_coverage, mapped)
        )
    if not candidates:
        raise ValueError(f"no canonical-chain mapping for {structure_id}")
    return min(candidates, key=lambda value: value[:3])[3]


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def download_biological_assembly(pdb_code: str, root: Path) -> Path:
    """Return a cached assembly 1, downloading it atomically on a cache miss."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{pdb_code.lower()}-assembly1.cif.gz"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    url = f"https://files.rcsb.org/download/{pdb_code.upper()}-assembly1.cif.gz"
    error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                timeout=(15, 120),
                headers={"User-Agent": "protein-state-router/1.0 (research structural analysis)"},
            )
            response.raise_for_status()
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(response.content)
            os.replace(temporary, destination)
            return destination
        except requests.RequestException as caught:
            error = caught
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download biological assembly for {pdb_code}") from error


def contact_delta(a: str, b: str) -> dict[str, object]:
    a_set, b_set = set(json.loads(a)), set(json.loads(b))
    union = a_set | b_set
    return {
        "contacts_gained": len(b_set - a_set),
        "contacts_lost": len(a_set - b_set),
        "contact_jaccard": len(a_set & b_set) / len(union) if union else 1.0,
    }


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_permutation_tests(
    properties: pd.DataFrame, *, seed: int = 42, permutations: int = 10_000
) -> pd.DataFrame:
    """Test hotspot-minus-control structural differences within each protein."""
    columns = [
        "feature_id",
        "metric",
        "n_paired_proteins",
        "hotspot_minus_control_mean",
        "hotspot_minus_control_median",
        "permutation_p_two_sided",
        "permutations",
        "fdr",
    ]
    if properties.empty or "feature_id" not in properties:
        return pd.DataFrame(columns=columns)
    metrics = (
        "sasa_angstrom2",
        "relative_sasa",
        "sphere_neighbor_count",
        "contact_density",
    )
    if "structure_source" in properties:
        properties = properties.loc[properties.structure_source.eq("experimental_pdb_assembly1")]
    rows: list[dict[str, object]] = []
    for feature_id, group in properties.groupby("feature_id"):
        if "matched_hotspot_residue_index" in group:
            pair_keys = ["protein_id", "matched_hotspot_residue_index"]
            if "structure_id" in group:
                pair_keys.append("structure_id")
            hotspot = group.loc[group.selection_kind.eq("hotspot")].set_index(pair_keys)
            control = group.loc[group.selection_kind.eq("low_activation_control")].set_index(
                pair_keys
            )
            paired = hotspot[list(metrics)].join(
                control[list(metrics)], lsuffix="_hotspot", rsuffix="_control", how="inner"
            )
        else:
            hotspot = group.loc[group.selection_kind.eq("hotspot")]
            control = group.loc[group.selection_kind.eq("low_activation_control")]
            hotspot_means = hotspot.groupby("protein_id")[list(metrics)].mean()
            control_means = control.groupby("protein_id")[list(metrics)].mean()
            paired = hotspot_means.join(
                control_means, lsuffix="_hotspot", rsuffix="_control", how="inner"
            )
        for metric_index, metric in enumerate(metrics):
            pair_difference = paired[f"{metric}_hotspot"] - paired[f"{metric}_control"]
            if isinstance(pair_difference.index, pd.MultiIndex):
                pair_difference = pair_difference.groupby(level="protein_id").mean()
            difference = pair_difference.dropna().to_numpy(dtype=np.float64)
            if len(difference) == 0:
                continue
            observed = float(difference.mean())
            rng = np.random.default_rng(seed + int(feature_id) * 1009 + metric_index * 997)
            exceedances = 0
            for start in range(0, permutations, 1_000):
                count = min(1_000, permutations - start)
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(difference)))
                null_means = (signs * difference[None, :]).mean(axis=1)
                exceedances += int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "metric": metric,
                    "n_paired_proteins": len(difference),
                    "hotspot_minus_control_mean": observed,
                    "hotspot_minus_control_median": float(np.median(difference)),
                    "permutation_p_two_sided": (exceedances + 1) / (permutations + 1),
                    "permutations": permutations,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["fdr"] = bh_adjust(result.permutation_p_two_sided.to_numpy())
    return result[columns]


def paired_delta_permutation_tests(
    deltas: pd.DataFrame, *, seed: int = 42, permutations: int = 10_000
) -> pd.DataFrame:
    """Test whether conformational changes differ between hotspots and controls."""
    if deltas.empty:
        return pd.DataFrame()
    renamed = deltas.rename(
        columns={
            "delta_sasa_b_minus_a": "sasa_angstrom2",
            "delta_relative_sasa_b_minus_a": "relative_sasa",
            "delta_sphere_neighbor_count_b_minus_a": "sphere_neighbor_count",
            "delta_contact_density_b_minus_a": "contact_density",
        }
    )
    result = paired_permutation_tests(renamed, seed=seed, permutations=permutations)
    if not result.empty:
        result.insert(1, "analysis", "state_b_minus_state_a_delta")
    return result


CONTACT_INTERACTION_METRICS = (
    "protein_contact_count",
    "sidechain_contact_count",
    "hydrogen_bond_contact_fraction",
    "polar_polar_contact_fraction",
    "hydrophobic_hydrophobic_contact_fraction",
    "hydrophobic_polar_contact_fraction",
    "aromatic_aromatic_contact_fraction",
    "polar_partner_contact_fraction",
    "nonpolar_partner_contact_fraction",
)


def sign_flip_test(
    difference: np.ndarray, *, seed: int, permutations: int
) -> tuple[float, str, int]:
    """Return deterministic two-sided sign-flip inference for a paired contrast."""
    values = np.asarray(difference, dtype=np.float64)
    observed = abs(float(values.mean()))
    if len(values) <= 16:
        masks = np.arange(2 ** len(values), dtype=np.uint32)[:, None]
        bit_positions = np.arange(len(values), dtype=np.uint32)
        signs = np.where(((masks >> bit_positions) & 1) == 1, 1.0, -1.0)
        null_means = (signs * values[None, :]).mean(axis=1)
        return float(np.mean(np.abs(null_means) >= observed)), "exact_sign_flip", len(null_means)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, permutations, 1_000):
        count = min(1_000, permutations - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        null_means = (signs * values[None, :]).mean(axis=1)
        exceedances += int(np.count_nonzero(np.abs(null_means) >= observed))
    return (exceedances + 1) / (permutations + 1), "monte_carlo_sign_flip", permutations


def paired_contact_interaction_tests(
    properties: pd.DataFrame, *, seed: int = 42, permutations: int = 10_000
) -> pd.DataFrame:
    """Test hotspot-control contact chemistry separately in the two dynamic cohorts."""
    columns = [
        "cohort",
        "feature_id",
        "metric",
        "n_paired_proteins",
        "hotspot_minus_control_mean",
        "hotspot_minus_control_median",
        "permutation_p_two_sided",
        "permutation_method",
        "permutations_or_exact_signs",
        "fdr",
    ]
    required = {
        "cohort",
        "feature_id",
        "protein_id",
        "selection_kind",
        "matched_hotspot_residue_index",
        *CONTACT_INTERACTION_METRICS,
    }
    if properties.empty or required - set(properties):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (cohort, feature_id), group in properties.groupby(["cohort", "feature_id"]):
        pair_keys = ["protein_id", "matched_hotspot_residue_index"]
        hotspot = group.loc[group.selection_kind.eq("hotspot")].set_index(pair_keys)
        control = group.loc[group.selection_kind.eq("low_activation_control")].set_index(pair_keys)
        paired = hotspot[list(CONTACT_INTERACTION_METRICS)].join(
            control[list(CONTACT_INTERACTION_METRICS)],
            lsuffix="_hotspot",
            rsuffix="_control",
            how="inner",
        )
        for metric_index, metric in enumerate(CONTACT_INTERACTION_METRICS):
            difference = paired[f"{metric}_hotspot"] - paired[f"{metric}_control"]
            protein_difference = difference.groupby(level="protein_id").mean().dropna().to_numpy()
            if not len(protein_difference):
                continue
            p_value, method, draws = sign_flip_test(
                protein_difference,
                seed=seed + int(feature_id) * 1009 + metric_index * 997,
                permutations=permutations,
            )
            rows.append(
                {
                    "cohort": cohort,
                    "feature_id": int(feature_id),
                    "metric": metric,
                    "n_paired_proteins": len(protein_difference),
                    "hotspot_minus_control_mean": float(protein_difference.mean()),
                    "hotspot_minus_control_median": float(np.median(protein_difference)),
                    "permutation_p_two_sided": p_value,
                    "permutation_method": method,
                    "permutations_or_exact_signs": draws,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["fdr"] = bh_adjust(result.permutation_p_two_sided.to_numpy())
    return result[columns]


def _read_property_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_contact_interaction_html(
    summary: dict[str, object], properties: pd.DataFrame, tests: pd.DataFrame, output: Path
) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if not tests.empty:
        focus = tests.loc[
            tests.metric.isin(
                (
                    "hydrogen_bond_contact_fraction",
                    "polar_polar_contact_fraction",
                    "hydrophobic_hydrophobic_contact_fraction",
                    "hydrophobic_polar_contact_fraction",
                    "aromatic_aromatic_contact_fraction",
                )
            )
        ].copy()
        labels = focus.metric.str.removesuffix("_contact_fraction").str.replace("_", " ")
        figure, axis = plt.subplots(figsize=(12, 5))
        for cohort, group in focus.assign(metric_label=labels).groupby("cohort"):
            axis.scatter(
                group.metric_label, group.hotspot_minus_control_mean, label=cohort, alpha=0.8
            )
        axis.axhline(0, color="#697782", linewidth=1)
        axis.set(ylabel="Hotspot minus control", title="Normalized contact-chemistry contrasts")
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "contact_interaction_effects.png", dpi=180)
        plt.close(figure)
    page = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>SAE hotspot contact chemistry</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#17212b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce3e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{font-family:ui-monospace,monospace}}</style></head><body>
<h1>SAE hotspot contact-chemistry analysis</h1><p>This post-hoc analysis reuses the completed ESMFold SAE structural-role hotspot/control selections and experimental PDB assembly mappings. It evaluates only held-out dynamic proteins, split into ProMISE ligand-induced and all remaining dynamic proteins.</p>
<p><b>Mapped dynamic proteins:</b> {summary.get("mapped_dynamic_proteins", 0):,}. <b>Ligand-induced:</b> {summary.get("mapped_ligand_induced_proteins", 0):,}. <b>Other dynamic:</b> {summary.get("mapped_other_dynamic_proteins", 0):,}.</p>
<p>Protein residue contacts are nonlocal heavy-atom contacts within 4.5 Å. Chemistry fractions use sidechain-contact partners; hydrogen-bond fractions use all protein contact partners and conservative donor/acceptor heavy-atom pairs within 3.5 Å. Ligands, ions, water, and other hetero-residues are excluded.</p>
<img src=\"figures/contact_interaction_effects.png\" alt=\"Contact chemistry effects\"><h2>Paired hotspot-control tests</h2><p>Each contrast is averaged within protein before two-sided sign-flip inference. The small ligand-induced cohort uses exact enumeration; the larger cohort uses seeded Monte Carlo permutations. BH FDR spans both cohorts, all selected features, and all contact metrics.</p>{tests.to_html(index=False, float_format=lambda value: f"{value:.3g}") if not tests.empty else "<p>No tests available.</p>"}
<h2>Artifacts</h2><ul><li><a href=\"contact_interaction_properties.csv\">Contact interaction properties</a></li><li><a href=\"contact_interaction_paired_tests.csv\">Paired tests</a></li><li><a href=\"coverage_audit.csv\">Coverage audit</a></li><li><a href=\"summary.json\">Summary</a></li></ul></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def build_html(
    summary: dict[str, object],
    comparison: pd.DataFrame,
    tests: pd.DataFrame,
    delta_tests: pd.DataFrame,
    output: Path,
) -> None:
    rows = []
    grouped = comparison.groupby("feature_id") if "feature_id" in comparison else []
    for feature, group in grouped:
        hotspot = group.loc[group.selection_kind.eq("hotspot")]
        control = group.loc[group.selection_kind.eq("low_activation_control")]
        rows.append(
            {
                "feature": int(feature),
                "hotspot_sasa": hotspot.sasa_angstrom2.mean(),
                "control_sasa": control.sasa_angstrom2.mean(),
                "hotspot_contacts": hotspot.contact_density.mean(),
                "control_contacts": control.contact_density.mean(),
            }
        )
    table = pd.DataFrame(rows).sort_values("feature") if rows else pd.DataFrame()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if not table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(table.control_sasa, table.hotspot_sasa, color="#276fbf")
        axes[0].axline((0, 0), slope=1, color="#697782", linewidth=1)
        axes[0].set(
            xlabel="Control mean SASA", ylabel="Hotspot mean SASA", title="Solvent exposure"
        )
        axes[1].scatter(table.control_contacts, table.hotspot_contacts, color="#2a9d8f")
        axes[1].axline((0, 0), slope=1, color="#697782", linewidth=1)
        axes[1].set(
            xlabel="Control contact density",
            ylabel="Hotspot contact density",
            title="Local packing",
        )
        fig.tight_layout()
        fig.savefig(figures / "hotspot_vs_control.png", dpi=180)
        plt.close(fig)
    page = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>SAE feature structural roles</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#17212b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dce3e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{font-family:ui-monospace,monospace}}</style></head><body>
<h1>SAE feature structural-role analysis</h1><p>Frozen homology-grouped Seed-42 SAE residue hotspots versus seeded low-activation controls matched preferentially by amino-acid identity and then by sequence position. Coordinates were processed in temporary batches and are not retained.</p>
<p><b>Completed proteins:</b> {summary.get("completed_proteins", 0):,} / {summary.get("total_proteins", 0):,}. <b>Experimental mappings:</b> {summary.get("experimental_mappings", 0):,}. <b>Fallbacks:</b> {summary.get("esmfold_fallbacks", 0):,}.</p>
<img src=\"figures/hotspot_vs_control.png\" alt=\"Hotspot structural properties\"><h2>Feature summaries</h2>{table.to_html(index=False, float_format=lambda value: f"{value:.3g}") if not table.empty else "<p>No mapped residue properties yet.</p>"}
<h2>Paired permutation tests</h2><p>Each test first computes matched hotspot-minus-control contrasts and then averages them within protein, preserving the protein as the unit of inference. Two-sided sign-flip permutations use 10,000 draws; FDR is across all feature-measure tests. The features were selected from the same held-out association results, so these tests are exploratory and require independent confirmation.</p>{tests.to_html(index=False, float_format=lambda value: f"{value:.3g}") if not tests.empty else "<p>No permutation tests available.</p>"}
<h2>Conformational-delta tests</h2>{delta_tests.to_html(index=False, float_format=lambda value: f"{value:.3g}") if not delta_tests.empty else "<p>No paired state-delta tests available.</p>"}
<h2>Artifacts</h2><ul><li><a href=\"selected_features.csv\">selected features</a></li><li><a href=\"residue_structure_properties.csv\">residue properties</a></li><li><a href=\"paired_permutation_tests.csv\">paired permutation tests</a></li><li><a href=\"paired_delta_permutation_tests.csv\">paired conformational-delta tests</a></li><li><a href=\"dynamic_transition_deltas.csv\">dynamic transition deltas</a></li><li><a href=\"coverage_audit.csv\">coverage audit</a></li></ul></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def load_partition_catalog(
    seed_catalog: Path,
    full_catalog: Path,
    transition_catalog: Path,
    transition_summary: Path,
    partition: str,
) -> pd.DataFrame:
    """Load one immutable split while retaining its experimental structure metadata."""
    if partition not in {"val", "test"}:
        raise ValueError("structural-role analysis supports only val or test partitions")
    seed = pd.read_parquet(seed_catalog)
    full = pd.read_parquet(full_catalog)
    full_structure = full[
        [
            "protein_id",
            "pdb_codes",
            "transition_state_a_structure_id",
            "transition_state_b_structure_id",
        ]
    ].rename(
        columns={
            "pdb_codes": "pdb_codes_full",
            "transition_state_a_structure_id": "transition_state_a_structure_id_full",
            "transition_state_b_structure_id": "transition_state_b_structure_id_full",
        }
    )
    merged = seed.merge(
        full_structure,
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
    transition = pd.read_csv(transition_catalog)[["protein_id", "structure_ids_json"]]
    summary = pd.read_csv(transition_summary)[
        ["protein_id", "status", "state_a_structure_id", "state_b_structure_id"]
    ]
    merged = merged.merge(transition, on="protein_id", how="left").merge(
        summary, on="protein_id", how="left", suffixes=("", "_summary")
    )
    for column in (
        "pdb_codes",
        "transition_state_a_structure_id",
        "transition_state_b_structure_id",
    ):
        full_column = f"{column}_full"
        if full_column in merged:
            merged[column] = merged[full_column]
            merged = merged.drop(columns=[full_column])
    result = merged.loc[merged.split.eq(partition)].copy()
    if result.empty or result.protein_id.duplicated().any():
        raise ValueError(f"expected non-empty unique frozen Seed-42 {partition} proteins")
    return result


def compare_effect_tables(reference: pd.DataFrame, validation: pd.DataFrame, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare feature-level structural effects without pooling proteins across partitions."""
    keys = ["feature_id", "metric"]
    required = {*keys, "hotspot_minus_control_mean", "fdr"}
    if required - set(reference) or required - set(validation):
        raise ValueError(f"{analysis} tables lack required effect columns")
    left = reference[list(required)].rename(
        columns={
            "hotspot_minus_control_mean": "reference_effect",
            "fdr": "reference_fdr",
        }
    )
    right = validation[list(required)].rename(
        columns={
            "hotspot_minus_control_mean": "validation_effect",
            "fdr": "validation_fdr",
        }
    )
    joined = left.merge(right, on=keys, how="inner", validate="one_to_one")
    joined.insert(0, "analysis", analysis)
    if joined.empty:
        return joined, pd.DataFrame()
    joined["same_effect_direction"] = np.sign(joined.reference_effect).eq(
        np.sign(joined.validation_effect)
    )
    joined["reference_fdr_significant"] = joined.reference_fdr.lt(0.05)
    joined["validation_fdr_significant"] = joined.validation_fdr.lt(0.05)
    joined["replicated_fdr_same_direction"] = (
        joined.reference_fdr_significant
        & joined.validation_fdr_significant
        & joined.same_effect_direction
    )
    summaries: list[dict[str, object]] = []
    for metric, group in joined.groupby("metric", sort=True):
        correlation = (
            float(np.corrcoef(group.reference_effect, group.validation_effect)[0, 1])
            if len(group) >= 2
            and np.std(group.reference_effect) > 0
            and np.std(group.validation_effect) > 0
            else float("nan")
        )
        reference_significant = group.reference_fdr_significant
        summaries.append(
            {
                "analysis": analysis,
                "metric": metric,
                "n_comparable_features": int(len(group)),
                "effect_correlation_pearson": correlation,
                "direction_concordance": float(group.same_effect_direction.mean()),
                "reference_fdr_significant": int(reference_significant.sum()),
                "validation_fdr_significant": int(group.validation_fdr_significant.sum()),
                "replicated_fdr_same_direction": int(group.replicated_fdr_same_direction.sum()),
                "reference_significant_replication_rate": float(
                    group.loc[reference_significant, "replicated_fdr_same_direction"].mean()
                )
                if reference_significant.any()
                else float("nan"),
            }
        )
    return joined, pd.DataFrame(summaries)


def write_reference_comparison(reference_output: Path, output: Path) -> dict[str, object]:
    """Persist a split-independent replication report for structural and delta tests."""
    outputs: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for name in ("paired_permutation_tests", "paired_delta_permutation_tests"):
        reference_path = reference_output / f"{name}.csv"
        validation_path = output / f"{name}.csv"
        if not reference_path.is_file() or not validation_path.is_file():
            continue
        effects, summary = compare_effect_tables(
            pd.read_csv(reference_path), pd.read_csv(validation_path), name
        )
        outputs.append(effects)
        summaries.append(summary)
    effect_frame = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    summary_frame = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    atomic_frame(effect_frame, output / "reference_effect_comparison.csv")
    atomic_frame(summary_frame, output / "reference_replication_summary.csv")
    result = {
        "reference_output": str(reference_output),
        "reference_effect_comparisons": int(len(effect_frame)),
        "reference_replication_summary_rows": int(len(summary_frame)),
    }
    atomic_json(output / "reference_comparison_manifest.json", result)
    return result


def run_contact_interaction_analysis(
    catalog_path: Path,
    source_properties_path: Path,
    output: Path,
    config: ContactInteractionConfig,
    rcsb_cache: Path,
) -> dict[str, object]:
    """Extend completed structural-role results with contact-chemistry tests only."""
    catalog = pd.read_parquet(catalog_path)
    dynamic = catalog.loc[catalog.split.eq("test") & catalog.dataset_label.eq(1)].copy()
    if dynamic.empty or dynamic.protein_id.duplicated().any():
        raise ValueError("expected unique held-out dynamic proteins")
    dynamic["cohort"] = np.where(
        dynamic.protein_id.str.startswith("promise:ligand_induced:"),
        "promise_ligand_induced",
        "other_dynamic",
    )
    source = _read_property_frame(source_properties_path)
    required = {
        "protein_id",
        "feature_id",
        "selection_kind",
        "matched_hotspot_residue_index",
        "canonical_residue_number",
        "structure_id",
        "structure_source",
    }
    if missing := required - set(source):
        raise ValueError(f"structural-role property table missing columns: {sorted(missing)}")
    source = source.loc[source.structure_source.eq("experimental_pdb_assembly1")].merge(
        dynamic[["protein_id", "sequence", "cohort"]],
        on="protein_id",
        how="inner",
        validate="many_to_one",
    )
    if source.empty:
        raise ValueError("no experimentally mapped dynamic hotspot/control residues")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    if progress_path.is_file():
        existing = json.loads(progress_path.read_text())
        if existing.get("config_hash") not in {None, config.config_hash}:
            raise ValueError(
                "existing contact-interaction checkpoint has a different configuration"
            )
    property_path = output / "contact_interaction_properties.csv"
    audit_path = output / "coverage_audit.csv"
    records = read_records(property_path)
    audits = [row for row in read_records(audit_path) if str(row.get("status")) == "complete"]
    # Failed downloads/mappings are retryable: the temporary-coordinate design
    # means a sandboxed or transient network failure must not permanently mark a
    # context as complete.
    completed = {
        (str(row["protein_id"]), str(row["structure_id"]))
        for row in audits
        if str(row.get("status")) == "complete"
    }
    groups = list(source.groupby(["protein_id", "structure_id"], sort=True))
    rcsb_cache.mkdir(parents=True, exist_ok=True)
    for group_index, ((protein_id, structure_id), group) in enumerate(groups, start=1):
        key = (str(protein_id), str(structure_id))
        if key in completed:
            continue
        sequence = str(group.sequence.iloc[0])
        positions = set(group.canonical_residue_number.astype(int))
        error_text = ""
        measurements: dict[int, dict[str, object]] = {}
        try:
            pdb_code, _ = parse_structure_id(str(structure_id))
            path = download_biological_assembly(pdb_code, rcsb_cache)
            mapped = resolve_mapped_structure(str(structure_id), path, sequence)
            measurements = interaction_properties(
                path, str(structure_id), mapped, positions, config
            )
        except Exception as error:  # Keep a resumable per-structure audit trail.
            error_text = f"{type(error).__name__}: {error}"
        if measurements:
            for row in group.itertuples(index=False):
                result = measurements.get(int(row.canonical_residue_number))
                if result is not None:
                    records.append(
                        {
                            "protein_id": row.protein_id,
                            "cohort": row.cohort,
                            "feature_id": int(row.feature_id),
                            "selection_kind": row.selection_kind,
                            "matched_hotspot_residue_index": int(row.matched_hotspot_residue_index),
                            "canonical_residue_number": int(row.canonical_residue_number),
                            "structure_id": row.structure_id,
                            **result,
                        }
                    )
        audits.append(
            {
                "protein_id": protein_id,
                "structure_id": structure_id,
                "status": "complete" if measurements else "unavailable",
                "n_requested_positions": len(positions),
                "n_measured_positions": len(measurements),
                "error": error_text,
            }
        )
        if len(audits) % 20 == 0 or group_index == len(groups):
            atomic_frame(pd.DataFrame(records), property_path)
            atomic_frame(pd.DataFrame(audits), audit_path)
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "config_hash": config.config_hash,
                    "completed_structure_contexts": len(audits),
                    "total_structure_contexts": len(groups),
                    "last_protein_id": protein_id,
                    "last_structure_id": structure_id,
                },
            )
            status(
                "processing_contact_interactions",
                completed=len(audits),
                total=len(groups),
                last_protein_id=protein_id,
            )
    properties = pd.DataFrame(records)
    if not properties.empty:
        properties.to_parquet(output / "contact_interaction_properties.parquet", index=False)
    tests = paired_contact_interaction_tests(
        properties, seed=config.seed, permutations=config.permutations
    )
    atomic_frame(tests, output / "contact_interaction_paired_tests.csv")
    if not tests.empty:
        tests.to_parquet(output / "contact_interaction_paired_tests.parquet", index=False)
    audit = pd.DataFrame(audits)
    mapped_by_cohort = (
        properties.groupby("cohort").protein_id.nunique().to_dict() if not properties.empty else {}
    )
    summary = {
        "config_hash": config.config_hash,
        "configuration": asdict(config),
        "source_properties": str(source_properties_path),
        "source_properties_sha256": sha256_file(source_properties_path),
        "dynamic_test_proteins": int(len(dynamic)),
        "ligand_induced_test_proteins": int(dynamic.cohort.eq("promise_ligand_induced").sum()),
        "other_dynamic_test_proteins": int(dynamic.cohort.eq("other_dynamic").sum()),
        "mapped_dynamic_proteins": int(properties.protein_id.nunique())
        if not properties.empty
        else 0,
        "mapped_ligand_induced_proteins": int(mapped_by_cohort.get("promise_ligand_induced", 0)),
        "mapped_other_dynamic_proteins": int(mapped_by_cohort.get("other_dynamic", 0)),
        "completed_structure_contexts": len(audit),
        "total_structure_contexts": len(groups),
        "primary_analysis": "experimental_pdb_assembly1_protein_residue_contacts_only",
    }
    atomic_json(output / "summary.json", summary)
    build_contact_interaction_html(summary, properties, tests, output)
    atomic_json(progress_path, {"status": "complete", **summary})
    status("contact_interactions_complete", **summary)
    return summary


def run(
    seed_catalog: Path,
    full_catalog: Path,
    transition_catalog: Path,
    transition_summary: Path,
    association_path: Path,
    sae_root: Path,
    output: Path,
    config: Config,
    rcsb_cache: Path,
    selected_features_path: Path | None = None,
    reference_output: Path | None = None,
) -> dict[str, object]:
    device = resolve_device(config.device)
    output.mkdir(parents=True, exist_ok=True)
    rcsb_cache.mkdir(parents=True, exist_ok=True)
    selected, selection_manifest = load_selected_features(
        association_path, config.features_per_track, selected_features_path
    )
    selected.to_csv(output / "selected_features.csv", index=False)
    catalog = load_partition_catalog(
        seed_catalog, full_catalog, transition_catalog, transition_summary, config.partition
    )
    model, center, manifest = load_frozen_sae(sae_root, device)
    if manifest.get("catalog_sha256") != sha256_file(seed_catalog):
        raise ValueError("frozen SAE and frozen catalog checksums differ")
    progress_path = output / "progress.json"
    existing = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if existing and existing.get("config_hash") != config.config_hash:
        raise ValueError("existing structural-role checkpoint has a different configuration")
    property_path = output / "residue_structure_properties.csv"
    audit_path = output / "coverage_audit.csv"
    properties = read_records(property_path)
    audits = read_records(audit_path)
    # Network and mapping failures are deliberately retryable.  Only successful
    # experimental contexts (or an explicit fallback) suppress reruns.
    completed = {str(row["protein_id"]) for row in audits if is_completed_structure_audit(row)}
    for row_index, row in enumerate(catalog.itertuples(index=False), start=1):
        if row.protein_id in completed:
            continue
        # Replace stale failed audit rows when retrying this protein.  This
        # keeps progress and final counts one-row-per-protein after recovery.
        audits = [
            audit
            for audit in audits
            if str(audit.get("protein_id")) != str(row.protein_id)
            or is_completed_structure_audit(audit)
        ]
        matrix = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=len(center),
        )
        selections = []
        for feature in selected.itertuples(index=False):
            for selection in choose_positions(
                activation_vector(matrix, int(feature.feature_id), model, center, device),
                row.protein_id,
                int(feature.feature_id),
                config,
                row.sequence,
            ):
                selections.append(
                    {
                        **selection,
                        "feature_id": int(feature.feature_id),
                        "selection_tracks": feature.selection_tracks,
                    }
                )
        requested = {int(item["residue_index"]) + 1 for item in selections}
        structure_ids = (
            json_list(row.structure_ids_json)
            if row.dataset_label == 1
            else json_list(row.pdb_codes)
        )
        if row.dataset_label == 1 and not structure_ids:
            structure_ids = [
                value
                for value in (
                    row.transition_state_a_structure_id,
                    row.transition_state_b_structure_id,
                )
                if isinstance(value, str)
            ]
        states: dict[str, tuple[dict[int, dict[str, object]], str]] = {}
        errors: list[str] = []
        with tempfile.TemporaryDirectory(prefix="sae-structural-roles-") as temporary:
            fallback_root = Path(temporary)
            for batch in chunks(list(dict.fromkeys(structure_ids)), config.download_batch_size):
                parsed_ids: list[tuple[str, str]] = []
                for structure_id in batch:
                    try:
                        pdb_code, _ = parse_structure_id(structure_id)
                        parsed_ids.append((structure_id, pdb_code))
                    except Exception as error:
                        errors.append(f"{structure_id}: {error}")
                codes = list(dict.fromkeys(code for _, code in parsed_ids))
                downloads: dict[str, Path] = {}
                download_errors: dict[str, Exception] = {}
                # Retain modest concurrency for independent assemblies while retry
                # backoff protects against transient cache-miss retrieval failures.
                with ThreadPoolExecutor(max_workers=min(4, max(1, len(codes)))) as executor:
                    futures = {
                        executor.submit(download_biological_assembly, code, rcsb_cache): code
                        for code in codes
                    }
                    for future in as_completed(futures):
                        code = futures[future]
                        try:
                            downloads[code] = future.result()
                        except Exception as error:
                            download_errors[code] = error
                for structure_id, pdb_code in parsed_ids:
                    try:
                        if pdb_code in download_errors:
                            raise download_errors[pdb_code]
                        assembly_path = downloads[pdb_code]
                        mapped = resolve_mapped_structure(structure_id, assembly_path, row.sequence)
                        measurements = structural_properties(
                            assembly_path,
                            structure_id,
                            mapped,
                            requested,
                            config,
                        )
                        if not measurements:
                            raise ValueError("no selected residues map onto biological assembly")
                        states[structure_id] = (
                            measurements,
                            "experimental_pdb_assembly1",
                        )
                    except (
                        Exception
                    ) as error:  # Per-structure errors should not discard a protein checkpoint.
                        errors.append(f"{structure_id}: {error}")
            if not states and config.enable_esmfold_fallback:
                try:
                    states["esmfold_single"] = (
                        esmfold_fallback_properties(
                            row.sequence, requested, fallback_root, device, config
                        ),
                        "esmfold_single_fallback",
                    )
                except Exception as error:
                    errors.append(f"esmfold fallback: {error}")
        if states:
            for selection in selections:
                canonical_position = int(selection["residue_index"]) + 1
                for _structure_id, (measurements, structure_source) in states.items():
                    measurement = measurements.get(canonical_position)
                    if measurement is None:
                        continue
                    properties.append(
                        {
                            "protein_id": row.protein_id,
                            "sequence_sha256": row.sequence_sha256,
                            "dataset_label": int(row.dataset_label),
                            "source_dataset": row.source_dataset,
                            "canonical_residue_number": canonical_position,
                            **selection,
                            "structure_source": structure_source,
                            **measurement,
                        }
                    )
        used_fallback = any(source == "esmfold_single_fallback" for _, source in states.values())
        audits.append(
            {
                "protein_id": row.protein_id,
                "dataset_label": int(row.dataset_label),
                "status": "esmfold_single_fallback"
                if used_fallback
                else ("complete" if states else "experimental_unavailable"),
                "n_requested_structures": len(structure_ids),
                "n_mapped_structures": len(states),
                "error": " | ".join(errors[:5]),
            }
        )
        if row_index % 10 == 0 or row_index == len(catalog):
            completed_count = sum(is_completed_structure_audit(audit) for audit in audits)
            atomic_frame(pd.DataFrame(properties), property_path)
            atomic_frame(pd.DataFrame(audits), audit_path)
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "config_hash": config.config_hash,
                    "completed_proteins": completed_count,
                    "total_proteins": len(catalog),
                    "last_protein_id": row.protein_id,
                },
            )
            status(
                "processing_proteins",
                completed=completed_count,
                total=len(catalog),
                last_protein_id=row.protein_id,
            )
    property_frame = pd.DataFrame(properties)
    if not property_frame.empty:
        property_frame.to_parquet(output / "residue_structure_properties.parquet", index=False)
    # Only explicit state A/B pairs have biologically directed deltas.
    deltas: list[dict[str, object]] = []
    if not property_frame.empty:
        for protein_id, group in property_frame.groupby("protein_id"):
            catalog_row = catalog.loc[catalog.protein_id.eq(protein_id)].iloc[0]
            a_id, b_id = catalog_row.state_a_structure_id, catalog_row.state_b_structure_id
            if not isinstance(a_id, str) or not isinstance(b_id, str):
                continue
            keys = ["feature_id", "selection_kind", "canonical_residue_number"]
            a = group.loc[group.structure_id.eq(a_id)].set_index(keys)
            b = group.loc[group.structure_id.eq(b_id)].set_index(keys)
            for key in a.index.intersection(b.index):
                first, second = a.loc[key], b.loc[key]
                deltas.append(
                    {
                        "protein_id": protein_id,
                        "feature_id": key[0],
                        "selection_kind": key[1],
                        "canonical_residue_number": key[2],
                        "state_a_structure_id": a_id,
                        "state_b_structure_id": b_id,
                        "delta_sasa_b_minus_a": second.sasa_angstrom2 - first.sasa_angstrom2,
                        "delta_relative_sasa_b_minus_a": second.relative_sasa - first.relative_sasa,
                        "delta_contact_density_b_minus_a": second.contact_density
                        - first.contact_density,
                        "delta_sphere_neighbor_count_b_minus_a": second.sphere_neighbor_count
                        - first.sphere_neighbor_count,
                        "secondary_structure_changed": bool(
                            second.secondary_structure != first.secondary_structure
                        ),
                        **contact_delta(first.contact_ids_json, second.contact_ids_json),
                    }
                )
    delta_frame = pd.DataFrame(deltas)
    atomic_frame(delta_frame, output / "dynamic_transition_deltas.csv")
    if not delta_frame.empty:
        delta_frame.to_parquet(output / "dynamic_transition_deltas.parquet", index=False)
    audit_frame = pd.DataFrame(audits)
    permutation_tests = paired_permutation_tests(property_frame, seed=config.seed)
    atomic_frame(permutation_tests, output / "paired_permutation_tests.csv")
    delta_tests = paired_delta_permutation_tests(delta_frame, seed=config.seed)
    atomic_frame(delta_tests, output / "paired_delta_permutation_tests.csv")
    summary = {
        "config_hash": config.config_hash,
        "configuration": asdict(config),
        "partition": config.partition,
        **selection_manifest,
        "total_proteins": len(catalog),
        "completed_proteins": int(
            audit_frame.status.isin(STRUCTURAL_AUDIT_TERMINAL_STATUSES).sum()
        ),
        "experimental_unavailable_proteins": int(
            audit_frame.status.eq("experimental_unavailable").sum()
        ),
        "experimental_mappings": int(audit_frame.n_mapped_structures.sum()),
        "esmfold_fallbacks": int(audit_frame.status.eq("esmfold_single_fallback").sum()),
        "primary_analysis_excludes_esmfold_fallbacks": True,
        "selected_unique_features": len(selected),
        "sae_manifest_sha256": sha256_file(sae_root / "manifest.json"),
        "rcsb_cache": str(rcsb_cache),
    }
    if reference_output is not None:
        summary["reference_comparison"] = write_reference_comparison(reference_output, output)
    atomic_json(output / "summary.json", summary)
    build_html(summary, property_frame, permutation_tests, delta_tests, output)
    atomic_json(progress_path, {"status": "complete", **summary})
    status("complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--full-catalog", type=Path, default=DEFAULT_FULL_CATALOG)
    parser.add_argument("--transition-catalog", type=Path, default=DEFAULT_TRANSITION_CATALOG)
    parser.add_argument("--transition-summary", type=Path, default=DEFAULT_TRANSITION_SUMMARY)
    parser.add_argument("--associations", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--sae-root", type=Path, default=DEFAULT_SAE)
    parser.add_argument(
        "--rcsb-cache",
        type=Path,
        default=DEFAULT_RCSB_CACHE,
        help="Persistent cache for RCSB biological-assembly CIFs shared across runs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partition", choices=("test", "val"), default="test")
    parser.add_argument(
        "--selected-features",
        type=Path,
        help="Freeze a preselected feature CSV instead of recomputing association rankings.",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Completed structural-role root used only for feature-effect replication comparison.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--features-per-track", type=int, default=10)
    parser.add_argument("--enable-esmfold-fallback", action="store_true")
    parser.add_argument(
        "--contact-interaction-only",
        action="store_true",
        help="Reuse completed structural-role selections for dynamic-protein contact chemistry only.",
    )
    parser.add_argument(
        "--contact-properties",
        type=Path,
        help="Existing residue_structure_properties parquet/csv to reuse (defaults to --output).",
    )
    parser.add_argument(
        "--contact-output",
        type=Path,
        help="Contact-chemistry output root (defaults to --output/contact_interactions).",
    )
    parser.add_argument("--contact-permutations", type=int, default=10_000)
    args = parser.parse_args()
    if args.features_per_track < 1:
        parser.error("--features-per-track must be positive")
    if args.contact_permutations < 1:
        parser.error("--contact-permutations must be positive")
    if args.contact_interaction_only:
        source = args.contact_properties or args.output / "residue_structure_properties.parquet"
        if not source.is_file() and source.suffix == ".parquet":
            source = source.with_suffix(".csv")
        run_contact_interaction_analysis(
            args.catalog,
            source,
            args.contact_output or args.output / "contact_interactions",
            ContactInteractionConfig(permutations=args.contact_permutations),
            args.rcsb_cache,
        )
        return
    run(
        args.catalog,
        args.full_catalog,
        args.transition_catalog,
        args.transition_summary,
        args.associations,
        args.sae_root,
        args.output,
        Config(
            device=args.device,
            features_per_track=args.features_per_track,
            enable_esmfold_fallback=args.enable_esmfold_fallback,
            partition=args.partition,
        ),
        args.rcsb_cache,
        selected_features_path=args.selected_features,
        reference_output=args.reference_output,
    )


if __name__ == "__main__":
    main()
