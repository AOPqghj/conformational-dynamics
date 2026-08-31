"""Exploratory Hinge Atlas test for frozen ESMFold SAE features.

The original Hinge Atlas download is no longer live.  This runner therefore
uses the five manually annotated examples supplied with this project, verifies
their PDB residue identities, and records its full provenance in the output
manifest.  It is an external descriptive validation, not a model-selection
step.
"""

# ruff: noqa: E402 - script execution needs the repository root before local imports.

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from protein_state_router.representations.esmfold_runner import ESMFoldTrunkExtractor

from interpretability.analyze_sae_transition_residue_associations import (
    bh_adjust,
    load_frozen_sae,
    sha256_file,
)
from interpretability.contracts import load_residue_matrix

CATALOG = Path("data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet")
SAE_ROOT = Path("ml/results/homology35_rerun/frozen_saes/esmfold_matrix_topk64_seed42")
ASSOCIATIONS = Path(
    "interpretability/results/homology35_rerun/sae_transition_associations/"
    "sae_feature_associations.parquet"
)
OUTPUT = Path("interpretability/results/homology35_rerun/hinge_atlas_sae")
SOURCE = "Flores et al. 2007 Hinge Atlas; user-provided manually annotated note subset"
SOURCE_URL = "https://doi.org/10.1186/1471-2105-8-167"
PDB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
AA3_TO_1 = {
    name.upper(): seq1(name).upper()
    for name in (
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
    ).split()
}


@dataclass(frozen=True)
class HingeCase:
    name: str
    reference_pdb: str
    alternate_pdb: str
    ranges: tuple[tuple[int, int], ...]


CASES = (
    HingeCase("ribose_binding_protein", "1BA2", "2DRI", ((101, 104), (235, 239), (266, 271))),
    HingeCase("calmodulin", "1CFD", "1CLL", ((74, 82),)),
    HingeCase("adenylate_kinase", "4AKE", "1AKE", ((30, 37), (115, 123))),
    HingeCase("lactoferrin", "1LFH", "1LFG", ((90, 93), (248, 253))),
    HingeCase("maltodextrin_binding_protein", "1OMP", "1ANF", ((110, 113), (259, 263))),
)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    control_draws: int = 20
    exclusion_radius: int = 5
    device: str = "cpu"
    allow_external_esmfold: bool = False
    max_external_identity: float = 0.98


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _ranges(case: HingeCase) -> list[int]:
    return [position for start, stop in case.ranges for position in range(start, stop + 1)]


def _pdb_codes(value: object) -> set[str]:
    if not isinstance(value, str) or not value:
        return set()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = value.split(",")
    return {str(item).upper().split("_")[0] for item in decoded if str(item).strip()}


def _alignment(source: str, target: str) -> tuple[dict[int, int], float]:
    """Map source indexes to target indexes using a global, identity-aware alignment."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -4.0
    aligner.extend_gap_score = -0.5
    result = aligner.align(source, target)[0]
    mapping: dict[int, int] = {}
    matched = aligned = 0
    for (source_start, source_stop), (target_start, target_stop) in zip(
        result.aligned[0], result.aligned[1], strict=True
    ):
        for source_index, target_index in zip(
            range(int(source_start), int(source_stop)),
            range(int(target_start), int(target_stop)),
            strict=True,
        ):
            mapping[source_index] = target_index
            matched += source[source_index] == target[target_index]
            aligned += 1
    return mapping, matched / aligned if aligned else 0.0


def _chain_records(cif_path: Path) -> list[tuple[str, str, dict[int, tuple[int, str]]]]:
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, cif_path)
    records: list[tuple[str, str, dict[int, tuple[int, str]]]] = []
    for chain in next(structure.get_models()).get_chains():
        sequence: list[str] = []
        position_map: dict[int, tuple[int, str]] = {}
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            amino_acid = AA3_TO_1.get(residue.resname.upper())
            if amino_acid is None:
                continue
            sequence_index = len(sequence)
            sequence.append(amino_acid)
            position_map[int(residue.id[1])] = (sequence_index, amino_acid)
        if sequence:
            records.append((str(chain.id), "".join(sequence), position_map))
    return records


def _select_chain(
    cif_path: Path, positions: Iterable[int]
) -> tuple[str, str, dict[int, tuple[int, str]]]:
    required = set(positions)
    choices = [record for record in _chain_records(cif_path) if required.issubset(record[2])]
    if not choices:
        raise ValueError(f"no PDB chain contains every annotated residue in {cif_path.name}")
    return max(choices, key=lambda item: len(item[1]))


def _fetch_pdb(pdb_id: str, cache: Path) -> Path:
    destination = cache / f"{pdb_id.upper()}.cif"
    if destination.is_file():
        return destination
    response = requests.get(PDB_URL.format(pdb_id=pdb_id.upper()), timeout=60)
    response.raise_for_status()
    cache.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def _canonical_candidate(
    catalog: pd.DataFrame, case: HingeCase, pdb_sequence: str
) -> tuple[pd.Series | None, dict[int, int], float]:
    test = catalog.loc[catalog.split.eq("test")].copy()
    direct = test.loc[
        test.pdb_codes.map(
            lambda value: bool({case.reference_pdb, case.alternate_pdb} & _pdb_codes(value))
        )
    ]
    candidates = (
        direct
        if len(direct)
        else test.loc[
            test.sequence_length.between(max(1, len(pdb_sequence) - 80), len(pdb_sequence) + 80)
        ]
    )
    best: tuple[float, pd.Series, dict[int, int]] | None = None
    for _, row in candidates.iterrows():
        mapping, identity = _alignment(pdb_sequence, str(row.sequence))
        if best is None or identity > best[0]:
            best = (identity, row, mapping)
    if best is None or best[0] < 0.98:
        return None, {}, 0.0
    return best[1], best[2], best[0]


def _overlap_split(catalog: pd.DataFrame, sequence: str, threshold: float) -> str | None:
    """Return a non-test catalog split sharing this sequence, if any."""
    non_test = catalog.loc[~catalog.split.eq("test")]
    best_split: str | None = None
    best_identity = 0.0
    for row in non_test.itertuples(index=False):
        value = str(row.sequence)
        if abs(len(value) - len(sequence)) > 80:
            continue
        _, identity = _alignment(sequence, str(value))
        if identity >= threshold and identity > best_identity:
            best_identity = identity
            best_split = str(row.split)
    return best_split


def _external_matrix(case: HingeCase, sequence: str, cache: Path, config: Config) -> np.ndarray:
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    path = cache / f"{case.name}_{digest}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return archive["values"].astype(np.float32, copy=False)
    if not config.allow_external_esmfold:
        raise RuntimeError("external ESMFold is disabled; rerun with --allow-external-esmfold")
    extractor = ESMFoldTrunkExtractor(device=config.device)
    embedding = extractor.extract(f"hinge_atlas:{case.name}", sequence)
    assert embedding.single is not None
    values = embedding.single.values.cpu().numpy().astype(np.float32, copy=False)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=values,
        sequence=np.asarray(sequence),
        model_id=np.asarray("facebook/esmfold_v1"),
    )
    return values


def _controls(
    sequence: str,
    hinges: list[int],
    *,
    draws: int,
    radius: int,
    seed: int,
) -> tuple[list[list[int]], list[list[str]]]:
    forbidden = {
        position
        for hinge in hinges
        for position in range(max(0, hinge - radius), min(len(sequence), hinge + radius + 1))
    }
    available = [position for position in range(len(sequence)) if position not in forbidden]
    if len(available) < len(hinges):
        raise ValueError("insufficient non-hinge residues for matched controls")
    rng = np.random.default_rng(seed)
    selections: list[list[int]] = []
    labels: list[list[str]] = []
    for _ in range(draws):
        unused = set(available)
        selected: list[int] = []
        draw_labels: list[str] = []
        for hinge in hinges:
            same = [position for position in unused if sequence[position] == sequence[hinge]]
            pool = same or list(unused)
            chosen = int(rng.choice(pool))
            unused.remove(chosen)
            selected.append(chosen)
            draw_labels.append("amino_acid_matched" if same else "fallback_any_residue")
        selections.append(selected)
        labels.append(draw_labels)
    return selections, labels


def _sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (len(null) + 1))


def _latents(
    matrix: np.ndarray, model: torch.nn.Module, center: np.ndarray, device: torch.device
) -> np.ndarray:
    centered = torch.from_numpy(matrix.astype(np.float32, copy=False) - center).to(device)
    with torch.inference_mode():
        _, values = model(centered)
    return values.cpu().numpy().astype(np.float32, copy=False)


def run(
    *,
    catalog_path: Path = CATALOG,
    sae_root: Path = SAE_ROOT,
    associations_path: Path = ASSOCIATIONS,
    output: Path = OUTPUT,
    config: Config | None = None,
) -> dict[str, object]:
    """Run the five-case external hinge-residue SAE comparison."""
    config = config or Config()
    if config.device != "cpu":
        raise ValueError("this exploratory runner defaults to CPU; pass --device cpu")
    torch.set_num_threads(2)
    catalog = pd.read_parquet(catalog_path)
    required = {"protein_id", "sequence", "sequence_length", "split", "embedding_path", "pdb_codes"}
    if required - set(catalog) or catalog.protein_id.duplicated().any():
        raise ValueError("expected a unique homology-aware catalog with embedding paths")
    device = torch.device(config.device)
    model, center, sae_manifest = load_frozen_sae(sae_root, device)
    output.mkdir(parents=True, exist_ok=True)
    pdb_cache = output / ".cache" / "pdb"
    external_cache = output / ".cache" / "external_esmfold"
    manifest_rows: list[dict[str, object]] = []
    protein_effects: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for case_index, case in enumerate(CASES):
        base = {
            "case": case.name,
            "reference_pdb": case.reference_pdb,
            "alternate_pdb": case.alternate_pdb,
            "annotated_pdb_residue_numbers": json.dumps(_ranges(case)),
            "annotation_source": SOURCE,
            "source_url": SOURCE_URL,
        }
        try:
            chain_id, pdb_sequence, pdb_positions = _select_chain(
                _fetch_pdb(case.reference_pdb, pdb_cache), _ranges(case)
            )
            candidate, alignment, identity = _canonical_candidate(catalog, case, pdb_sequence)
            source = ""
            if candidate is not None:
                sequence = str(candidate.sequence)
                matrix = load_residue_matrix(
                    Path(candidate.embedding_path),
                    protein_id=candidate.protein_id,
                    sequence=sequence,
                    sequence_sha256=candidate.sequence_sha256,
                    sequence_length=int(candidate.sequence_length),
                    expected_width=len(center),
                )
                source = "canonical_test"
                protein_id = str(candidate.protein_id)
            else:
                overlap = _overlap_split(catalog, pdb_sequence, config.max_external_identity)
                if overlap is not None:
                    manifest_rows.append(
                        {**base, "status": f"excluded_{overlap}_overlap", "pdb_chain": chain_id}
                    )
                    continue
                sequence = pdb_sequence
                matrix = _external_matrix(case, sequence, external_cache, config)
                source = "external_esmfold"
                protein_id = f"hinge_atlas:{case.name}"
                alignment, identity = _alignment(pdb_sequence, sequence)
            hinge_positions: list[int] = []
            for residue_number in _ranges(case):
                pdb_index, residue = pdb_positions[residue_number]
                target_index = alignment.get(pdb_index)
                if target_index is None or sequence[target_index] != residue:
                    raise ValueError(f"annotation mismatch at PDB residue {residue_number}")
                hinge_positions.append(target_index)
            hinge_positions = sorted(set(hinge_positions))
            if not hinge_positions:
                raise ValueError("no hinge residues mapped to the embedding sequence")
            latents = _latents(matrix, model, center, device)
            draws, labels = _controls(
                sequence,
                hinge_positions,
                draws=config.control_draws,
                radius=config.exclusion_radius,
                seed=config.seed + case_index,
            )
            hinge_mean = latents[hinge_positions].mean(axis=0)
            effect_draws = []
            for draw_index, (positions, match_labels) in enumerate(zip(draws, labels, strict=True)):
                effect_draws.append(hinge_mean - latents[positions].mean(axis=0))
                for hinge_position, control_position, match_level in zip(
                    hinge_positions, positions, match_labels, strict=True
                ):
                    control_rows.append(
                        {
                            "case": case.name,
                            "protein_id": protein_id,
                            "draw": draw_index,
                            "hinge_position_0based": hinge_position,
                            "control_position_0based": control_position,
                            "hinge_residue": sequence[hinge_position],
                            "control_residue": sequence[control_position],
                            "match_level": match_level,
                        }
                    )
            effects = np.asarray(effect_draws).mean(axis=0)
            protein_effects.extend(
                {
                    "case": case.name,
                    "protein_id": protein_id,
                    "feature_id": feature,
                    "mean_hinge_activation": float(hinge_mean[feature]),
                    "mean_control_activation": float(hinge_mean[feature] - effects[feature]),
                    "mean_hinge_minus_control": float(effects[feature]),
                }
                for feature in range(latents.shape[1])
            )
            manifest_rows.append(
                {
                    **base,
                    "status": "included",
                    "protein_id": protein_id,
                    "source": source,
                    "pdb_chain": chain_id,
                    "sequence_length": len(sequence),
                    "sequence_identity_to_pdb": identity,
                    "hinge_positions_0based": json.dumps(hinge_positions),
                }
            )
        except Exception as error:  # The manifest is the audit record for unavailable cases.
            manifest_rows.append(
                {**base, "status": "excluded_error", "error": f"{type(error).__name__}: {error}"}
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output / "hinge_manifest.csv", index=False)
    manifest.to_parquet(output / "hinge_manifest.parquet", index=False)
    controls = pd.DataFrame(control_rows)
    controls.to_parquet(output / "hinge_control_residues.parquet", index=False)
    effects = pd.DataFrame(protein_effects)
    if effects.empty:
        raise RuntimeError("no Hinge Atlas cases were eligible; inspect hinge_manifest.csv")
    grouped = effects.groupby("feature_id", sort=True)
    summary = grouped.agg(
        proteins=("protein_id", "nunique"),
        mean_hinge_activation=("mean_hinge_activation", "mean"),
        mean_control_activation=("mean_control_activation", "mean"),
        mean_hinge_minus_control=("mean_hinge_minus_control", "mean"),
        median_hinge_minus_control=("mean_hinge_minus_control", "median"),
    ).reset_index()
    p_values = (
        grouped["mean_hinge_minus_control"]
        .apply(lambda values: _sign_flip_p(values.to_numpy()))
        .to_numpy(dtype=float)
    )
    summary["protein_sign_flip_p"] = p_values
    summary["protein_sign_flip_fdr"] = bh_adjust(p_values)
    summary["hinge_effect_rank"] = summary.mean_hinge_minus_control.rank(
        ascending=False, method="min"
    ).astype(int)
    if associations_path.is_file():
        associations = pd.read_parquet(associations_path)
        columns = [
            "feature_id",
            "displacement_balanced_spearman",
            "displacement_fdr",
            "prs_balanced_spearman",
            "prs_fdr",
        ]
        summary = summary.merge(associations[columns], on="feature_id", how="left")
        summary["transition_associated_both_fdr_005"] = summary.displacement_fdr.le(
            0.05
        ) & summary.prs_fdr.le(0.05)
    summary.to_csv(output / "hinge_feature_effects.csv", index=False)
    summary.to_parquet(output / "hinge_feature_effects.parquet", index=False)
    overlap = summary.sort_values("hinge_effect_rank").head(50)
    overlap.to_csv(output / "transition_feature_overlap.csv", index=False)
    result = {
        "status": "completed",
        "configuration": asdict(config),
        "included_cases": int(manifest.status.eq("included").sum()),
        "excluded_cases": int((~manifest.status.eq("included")).sum()),
        "features": int(len(summary)),
        "top_hinge_feature": int(summary.sort_values("hinge_effect_rank").iloc[0].feature_id),
        "sae_manifest_sha256": sha256_file(sae_root / "manifest.json"),
        "association_table_sha256": sha256_file(associations_path)
        if associations_path.is_file()
        else None,
    }
    _json(output / "summary.json", result)
    _json(output / "run_manifest.json", {**result, "source": SOURCE, "source_url": SOURCE_URL})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--sae-root", type=Path, default=SAE_ROOT)
    parser.add_argument("--associations", type=Path, default=ASSOCIATIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--control-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-external-esmfold", action="store_true")
    args = parser.parse_args()
    if args.control_draws < 1:
        parser.error("--control-draws must be positive")
    print(
        json.dumps(
            run(
                catalog_path=args.catalog,
                sae_root=args.sae_root,
                associations_path=args.associations,
                output=args.output,
                config=Config(
                    seed=args.seed,
                    control_draws=args.control_draws,
                    device=args.device,
                    allow_external_esmfold=args.allow_external_esmfold,
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
