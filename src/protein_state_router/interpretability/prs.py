"""Canonical residue mapping and perturbation-response scanning (PRS)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from protein_state_router.external.structure_geometry import THREE_TO_ONE, StructureGeometry


@dataclass(frozen=True)
class PRSConfig:
    cutoff_start: float = 9.5
    cutoff_stop: float = 12.0
    cutoff_step: float = 0.5
    n_force_directions: int = 200
    convergence_fraction: float = 0.5
    seed: int = 42
    eigenvalue_rtol: float = 1e-8

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class MappedStructure:
    structure_id: str
    chain_id: str
    canonical_positions: tuple[int, ...]
    canonical_amino_acids: tuple[str, ...]
    label_residue_numbers: tuple[int, ...]
    auth_residue_numbers: tuple[str, ...]
    insertion_codes: tuple[str, ...]
    ca_coords: np.ndarray
    sequence_identity: float
    canonical_coverage: float


@dataclass(frozen=True)
class AlignedTransition:
    positions: tuple[int, ...]
    amino_acids: tuple[str, ...]
    a_label_numbers: tuple[int, ...]
    a_auth_numbers: tuple[str, ...]
    a_insertions: tuple[str, ...]
    b_label_numbers: tuple[int, ...]
    b_auth_numbers: tuple[str, ...]
    b_insertions: tuple[str, ...]
    a_coords: np.ndarray
    b_coords: np.ndarray
    displacement_vectors: np.ndarray
    displacements: np.ndarray
    rmsd: float
    canonical_coverage: float


def _global_alignment_map(observed: str, canonical: str) -> list[tuple[int, int]]:
    """Needleman-Wunsch mapping, returning zero-based observed/canonical matches."""
    n, m = len(observed), len(canonical)
    previous = np.arange(m + 1, dtype=np.int32) * -2
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    trace[0, 1:] = 2
    trace[1:, 0] = 1
    for i, amino_acid in enumerate(observed, start=1):
        current = np.empty(m + 1, dtype=np.int32)
        current[0] = -2 * i
        for j, canonical_amino_acid in enumerate(canonical, start=1):
            diagonal = previous[j - 1] + (2 if amino_acid == canonical_amino_acid else -1)
            up = previous[j] - 2
            left = current[j - 1] - 2
            best = max(diagonal, up, left)
            current[j] = best
            trace[i, j] = 0 if best == diagonal else (1 if best == up else 2)
        previous = current
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i and j:
        direction = trace[i, j]
        if direction == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def map_structure_to_canonical(
    structure: StructureGeometry,
    canonical_sequence: str,
    minimum_identity: float = 0.9,
    minimum_coverage: float = 0.8,
) -> MappedStructure:
    observed = "".join(THREE_TO_ONE.get(name.upper(), "X") for name in structure.residue_names)
    pairs = _global_alignment_map(observed, canonical_sequence)
    matches = sum(observed[i] == canonical_sequence[j] for i, j in pairs)
    identity = matches / max(1, len(pairs))
    coverage = len(pairs) / max(1, len(canonical_sequence))
    if identity < minimum_identity or coverage < minimum_coverage:
        raise ValueError(
            f"canonical mapping failed: identity={identity:.3f}, coverage={coverage:.3f}"
        )
    observed_indices = [i for i, _ in pairs]
    canonical_positions = tuple(j + 1 for _, j in pairs)
    auth = structure.auth_residue_numbers or tuple(map(str, structure.residue_numbers))
    insertions = structure.insertion_codes or ("",) * len(structure.residue_numbers)
    return MappedStructure(
        structure.structure_id,
        structure.auth_chain_id or structure.chain_id,
        canonical_positions,
        tuple(canonical_sequence[position - 1] for position in canonical_positions),
        tuple(structure.residue_numbers[i] for i in observed_indices),
        tuple(auth[i] for i in observed_indices),
        tuple(insertions[i] for i in observed_indices),
        structure.ca_coords[observed_indices],
        identity,
        coverage,
    )


def align_transition(
    state_a: MappedStructure, state_b: MappedStructure, canonical_length: int
) -> AlignedTransition:
    common = tuple(sorted(set(state_a.canonical_positions) & set(state_b.canonical_positions)))
    if len(common) < 3:
        raise ValueError("at least three common canonical residues are required")
    ia = {position: index for index, position in enumerate(state_a.canonical_positions)}
    ib = {position: index for index, position in enumerate(state_b.canonical_positions)}
    a_indices = [ia[position] for position in common]
    b_indices = [ib[position] for position in common]
    a_coords = state_a.ca_coords[a_indices]
    b_coords = state_b.ca_coords[b_indices]
    a_centered = a_coords - a_coords.mean(axis=0)
    b_centered = b_coords - b_coords.mean(axis=0)
    u, _, vt = np.linalg.svd(a_centered.T @ b_centered)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    aligned_a = a_centered @ rotation
    vectors = b_centered - aligned_a
    distances = np.linalg.norm(vectors, axis=1)
    return AlignedTransition(
        common,
        tuple(state_a.canonical_amino_acids[ia[p]] for p in common),
        tuple(state_a.label_residue_numbers[ia[p]] for p in common),
        tuple(state_a.auth_residue_numbers[ia[p]] for p in common),
        tuple(state_a.insertion_codes[ia[p]] for p in common),
        tuple(state_b.label_residue_numbers[ib[p]] for p in common),
        tuple(state_b.auth_residue_numbers[ib[p]] for p in common),
        tuple(state_b.insertion_codes[ib[p]] for p in common),
        aligned_a,
        b_centered,
        vectors,
        distances,
        float(np.sqrt(np.mean(distances**2))),
        len(common) / max(1, canonical_length),
    )


def build_anm_hessian(coords: np.ndarray, cutoff: float) -> tuple[np.ndarray, int]:
    n_residues = len(coords)
    hessian = np.zeros((3 * n_residues, 3 * n_residues), dtype=np.float64)
    contacts = 0
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            delta = coords[j] - coords[i]
            distance_squared = float(delta @ delta)
            if distance_squared <= 0 or distance_squared > cutoff * cutoff:
                continue
            block = np.outer(delta, delta) / distance_squared
            si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            hessian[si, si] += block
            hessian[sj, sj] += block
            hessian[si, sj] -= block
            hessian[sj, si] -= block
            contacts += 1
    return hessian, contacts


def hessian_pseudoinverse(
    coords: np.ndarray, config: PRSConfig
) -> tuple[np.ndarray, float, int, int, float]:
    cutoff = config.cutoff_start
    while cutoff <= config.cutoff_stop + 1e-9:
        hessian, contacts = build_anm_hessian(coords, cutoff)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        tolerance = max(config.eigenvalue_rtol * max(float(eigenvalues[-1]), 1.0), 1e-12)
        if float(eigenvalues[0]) < -10 * tolerance:
            raise ValueError("ANM Hessian has a meaningful negative eigenvalue")
        zero_modes = int(np.count_nonzero(eigenvalues <= tolerance))
        if zero_modes == 6:
            nonzero = eigenvalues > tolerance
            covariance = (eigenvectors[:, nonzero] / eigenvalues[nonzero]) @ eigenvectors[
                :, nonzero
            ].T
            return covariance, cutoff, contacts, zero_modes, tolerance
        cutoff += config.cutoff_step
    raise ValueError(
        f"ANM network remained disconnected: {zero_modes} zero modes at {config.cutoff_stop} A"
    )


def perturbation_response_scan(
    coords: np.ndarray,
    transition_vectors: np.ndarray,
    sequence_hash: str,
    config: PRSConfig,
) -> tuple[list[dict[str, float | bool]], dict[str, float | int | str]]:
    target = np.asarray(transition_vectors, dtype=np.float64).reshape(-1)
    target_norm = float(np.linalg.norm(target))
    if target_norm <= 1e-12:
        raise ValueError("zero_transition")
    covariance, cutoff, contacts, zero_modes, tolerance = hessian_pseudoinverse(coords, config)
    seed_material = hashlib.sha256(f"{config.seed}:{sequence_hash}".encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_material, "big"))
    if config.n_force_directions < 20 or not 0 < config.convergence_fraction < 1:
        raise ValueError("PRS direction count and convergence fraction are invalid")
    convergence_count = max(10, round(config.n_force_directions * config.convergence_fraction))
    rows: list[dict[str, float | bool]] = []
    for residue_index in range(len(coords)):
        directions = rng.normal(size=(config.n_force_directions, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        responses = covariance[:, 3 * residue_index : 3 * residue_index + 3] @ directions.T
        norms = np.linalg.norm(responses, axis=0)
        valid = norms > 1e-12
        overlaps = np.zeros(config.n_force_directions, dtype=np.float64)
        overlaps[valid] = target @ responses[:, valid] / (target_norm * norms[valid])
        full_max = float(overlaps.max())
        partial_max = float(overlaps[:convergence_count].max())
        rows.append(
            {
                "prs_max_overlap": full_max,
                "prs_mean_overlap": float(overlaps.mean()),
                "prs_max_abs_overlap": float(np.abs(overlaps).max()),
                "prs_mean_abs_overlap": float(np.abs(overlaps).mean()),
                "prs_partial_direction_max_overlap": partial_max,
                "prs_direction_convergence_abs_delta": abs(full_max - partial_max),
            }
        )
    order = np.argsort([-float(row["prs_max_overlap"]) for row in rows])
    top5, top10 = (
        set(order[: max(1, int(np.ceil(len(rows) * 0.05)))]),
        set(order[: max(1, int(np.ceil(len(rows) * 0.10)))]),
    )
    for index, row in enumerate(rows):
        row["prs_top_5_percent"] = index in top5
        row["prs_top_10_percent"] = index in top10
    return rows, {
        "prs_cutoff_angstrom": cutoff,
        "prs_contact_count": contacts,
        "prs_zero_modes": zero_modes,
        "prs_eigenvalue_tolerance": tolerance,
        "prs_config_hash": config.config_hash,
        "prs_force_directions": config.n_force_directions,
        "prs_convergence_directions": convergence_count,
        "prs_max_direction_convergence_abs_delta": float(
            max(row["prs_direction_convergence_abs_delta"] for row in rows)
        ),
        "prs_mean_direction_convergence_abs_delta": float(
            np.mean([row["prs_direction_convergence_abs_delta"] for row in rows])
        ),
    }
