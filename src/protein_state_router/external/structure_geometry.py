"""Minimal, dependency-free mmCIF geometry evidence for negative curation."""

# ruff: noqa: E701, E702
from __future__ import annotations

import gzip
import shlex
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StructureGeometry:
    structure_id: str
    chain_id: str
    residue_numbers: tuple[int, ...]
    ca_coords: np.ndarray
    residue_names: tuple[str, ...] = ()
    auth_chain_id: str | None = None
    auth_residue_numbers: tuple[str, ...] = ()
    insertion_codes: tuple[str, ...] = ()


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
}


def parse_mmcif_ca_chains(
    path: str | Path, structure_id: str | None = None
) -> list[StructureGeometry]:
    """Parse one C-alpha conformer per polymer chain from an mmCIF file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    lines = opener(path, "rt", encoding="utf-8", errors="replace").read().splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        loop_headers: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            loop_headers.append(lines[index].strip().split()[0])
            index += 1
        if not any(header.startswith("_atom_site.") for header in loop_headers):
            continue
        headers = loop_headers
        while index < len(lines):
            value = lines[index].strip()
            if not value or value == "#" or value == "loop_" or value.startswith("data_"):
                break
            parts = shlex.split(value)
            if len(parts) >= len(headers):
                rows.append(parts[: len(headers)])
            index += 1
        break
    idx = {h.rsplit(".", 1)[-1]: i for i, h in enumerate(headers)}
    required = {"label_atom_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"}
    if not required.issubset(idx):
        raise ValueError(f"mmCIF missing atom_site columns: {sorted(required - set(idx))}")
    chains: dict[tuple[str, str], dict[int, tuple[np.ndarray, str, str, str]]] = {}
    for row in rows:
        if row[idx["label_atom_id"]] != "CA":
            continue
        if "label_alt_id" in idx and row[idx["label_alt_id"]] not in {".", "?", "A", "1"}:
            continue
        label_chain = row[idx["label_asym_id"]]
        auth_chain = row[idx["auth_asym_id"]] if "auth_asym_id" in idx else label_chain
        try:
            label_residue = int(row[idx["label_seq_id"]])
            xyz = np.array([float(row[idx[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")])
        except (ValueError, KeyError):
            continue
        name = row[idx["label_comp_id"]] if "label_comp_id" in idx else "UNK"
        auth_residue = row[idx["auth_seq_id"]] if "auth_seq_id" in idx else str(label_residue)
        insertion = row[idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in idx else ""
        insertion = "" if insertion in {".", "?"} else insertion
        chains.setdefault((label_chain, auth_chain), {}).setdefault(
            label_residue, (xyz, name, auth_residue, insertion)
        )
    parsed: list[StructureGeometry] = []
    for (label_chain, auth_chain), residues in chains.items():
        nums = tuple(sorted(residues))
        parsed.append(
            StructureGeometry(
                structure_id or Path(path).stem,
                label_chain,
                nums,
                np.vstack([residues[n][0] for n in nums]),
                tuple(residues[n][1] for n in nums),
                auth_chain,
                tuple(residues[n][2] for n in nums),
                tuple(residues[n][3] for n in nums),
            )
        )
    if not parsed:
        raise ValueError(f"no CA atoms found in {path}")
    return parsed


def parse_mmcif_ca(
    path: str | Path, structure_id: str | None = None, chain_id: str | None = None
) -> StructureGeometry:
    """Parse CA atoms from a small mmCIF file (sufficient for curated evidence)."""
    chains = parse_mmcif_ca_chains(path, structure_id)
    if chain_id is None:
        return chains[0]
    for chain in chains:
        if chain.chain_id == chain_id or chain.auth_chain_id == chain_id:
            return chain
    raise ValueError(f"no CA atoms found for chain {chain_id} in {path}")


def aligned_residue_displacements(
    a: StructureGeometry, b: StructureGeometry
) -> tuple[tuple[int, ...], tuple[str, ...], np.ndarray]:
    """Return per-residue Cα displacement after one global Kabsch superposition."""
    common = tuple(sorted(set(a.residue_numbers) & set(b.residue_numbers)))
    if len(common) < 3:
        raise ValueError("at least three common residue positions are required")
    ia = {residue: index for index, residue in enumerate(a.residue_numbers)}
    ib = {residue: index for index, residue in enumerate(b.residue_numbers)}
    x = a.ca_coords[[ia[residue] for residue in common]]
    y = b.ca_coords[[ib[residue] for residue in common]]
    x = x - x.mean(0)
    y = y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    names = tuple(a.residue_names[ia[residue]] if a.residue_names else "UNK" for residue in common)
    return common, names, np.linalg.norm(x @ rotation - y, axis=1)


def aligned_rmsd(a: StructureGeometry, b: StructureGeometry) -> tuple[float, float]:
    common = set(a.residue_numbers) & set(b.residue_numbers)
    if len(common) < 3:
        return float("nan"), len(common) / max(
            1, min(len(a.residue_numbers), len(b.residue_numbers))
        )
    _, _, displacements = aligned_residue_displacements(a, b)
    rmsd = float(np.sqrt(np.mean(displacements**2)))
    return rmsd, len(common) / max(1, min(len(a.residue_numbers), len(b.residue_numbers)))


def group_geometry(
    paths: list[str | Path], chain_ids: list[str] | None = None
) -> dict[str, float | int | None]:
    """Summarize all valid pairwise Cα comparisons for one same-protein group."""
    if len(paths) < 2:
        return {
            "n_available_conformations": len(paths),
            "max_ca_rmsd": None,
            "aligned_coverage": None,
        }
    chains: list[str | None] = list(chain_ids) if chain_ids is not None else [None] * len(paths)
    structures = [
        parse_mmcif_ca(path, chain_id=chain) for path, chain in zip(paths, chains, strict=True)
    ]
    pairs = [aligned_rmsd(first, second) for first, second in combinations(structures, 2)]
    valid = [(rmsd, coverage) for rmsd, coverage in pairs if not np.isnan(rmsd)]
    if not valid:
        return {
            "n_available_conformations": len(paths),
            "max_ca_rmsd": None,
            "aligned_coverage": None,
        }
    return {
        "n_available_conformations": len(paths),
        "max_ca_rmsd": max(rmsd for rmsd, _ in valid),
        "aligned_coverage": min(coverage for _, coverage in valid),
    }
