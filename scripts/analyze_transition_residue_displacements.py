"""Measure per-residue Cα displacement for explicit-state transition pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from protein_state_router.external.structure_geometry import (
    aligned_residue_displacements,
    aligned_rmsd,
    parse_mmcif_ca,
)


def parse_structure_id(value: str) -> tuple[str, list[str | None]]:
    parts = value.strip().split("_")
    if not parts or len(parts[0]) != 4:
        raise ValueError(f"invalid PDB structure ID: {value}")
    candidates = [parts[-1]] if len(parts) > 1 else []
    if len(parts) > 2:
        candidates.append(parts[1])
    if candidates and candidates[0].rstrip("0123456789") != candidates[0]:
        candidates.append(candidates[0].rstrip("0123456789"))
    candidates.append(None)
    return parts[0].upper(), list(dict.fromkeys(candidates))


def resolve_structure(structure_id: str, structures_root: Path):
    pdb_code, chain_candidates = parse_structure_id(structure_id)
    path = structures_root / f"{pdb_code.lower()}.cif.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing mmCIF for {pdb_code}: {path}")
    errors: list[str] = []
    for chain in chain_candidates:
        try:
            return parse_mmcif_ca(path, structure_id=structure_id, chain_id=chain)
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(f"could not resolve chain for {structure_id}: {errors[-1]}")


def analyze(frame: pd.DataFrame, structures_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    residues: list[dict[str, object]] = []
    direct = frame.loc[frame.transition_priority.eq("direct_pair")]
    for row in direct.itertuples(index=False):
        first = resolve_structure(row.selected_state_a_structure_id, structures_root)
        second = resolve_structure(row.selected_state_b_structure_id, structures_root)
        positions, amino_acids, distances = aligned_residue_displacements(first, second)
        rmsd, coverage = aligned_rmsd(first, second)
        summaries.append(
            {
                "protein_id": row.protein_id,
                "state_a_structure_id": row.selected_state_a_structure_id,
                "state_b_structure_id": row.selected_state_b_structure_id,
                "resolved_state_a_chain": first.chain_id,
                "resolved_state_b_chain": second.chain_id,
                "aligned_ca_rmsd": rmsd,
                "aligned_coverage": coverage,
                "n_aligned_residues": len(positions),
                "max_residue_ca_displacement": float(distances.max()),
                "mean_residue_ca_displacement": float(distances.mean()),
            }
        )
        residues.extend(
            {
                "protein_id": row.protein_id,
                "state_a_structure_id": row.selected_state_a_structure_id,
                "state_b_structure_id": row.selected_state_b_structure_id,
                "residue_position": position,
                "residue_position_system": "mmcif_label_seq_id",
                "amino_acid_3letter": amino_acid,
                "ca_displacement_after_global_kabsch": float(distance),
            }
            for position, amino_acid, distance in zip(
                positions, amino_acids, distances, strict=True
            )
        )
    return pd.DataFrame(summaries), pd.DataFrame(residues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--residue-output", type=Path, required=True)
    args = parser.parse_args()
    summary, residues = analyze(pd.read_csv(args.candidates), args.structures_root)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.residue_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    residues.to_csv(args.residue_output, index=False)
    print(json.dumps({"pairs": len(summary), "residues": len(residues)}, sort_keys=True))


if __name__ == "__main__":
    main()
