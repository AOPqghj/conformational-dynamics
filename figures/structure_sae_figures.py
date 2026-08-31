#!/usr/bin/env python3
# ruff: noqa: E402 - direct execution needs repository imports after path setup.

"""Render SAE hotspots and low-activation controls with ChimeraX.

The selection and structure-mapping steps deliberately reuse the frozen
test-set interpretability workflow.  Each example is a randomly ordered,
dynamic Seed-42 test protein with one transition-associated SAE feature, its
strongest residue activation, and a residue with below-mean activation for
the same feature.  ChimeraX renders the mapped experimental assembly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from interpretability.analyze_sae_feature_structural_roles import (
    DEFAULT_CATALOG,
    DEFAULT_FULL_CATALOG,
    DEFAULT_TRANSITION_CATALOG,
    DEFAULT_TRANSITION_SUMMARY,
    Config,
    activation_vector,
    choose_positions,
    download_biological_assembly,
    load_test_catalog,
    resolve_device,
    resolve_mapped_structure,
    select_feature_tracks,
)
from interpretability.analyze_sae_transition_residue_associations import load_frozen_sae
from interpretability.contracts import load_residue_matrix
from PIL import Image, ImageDraw
from scripts.analyze_transition_residue_displacements import parse_structure_id

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "writeup/figures/sae_region_candidates"
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "writeup/figures/chimerax_structure_cache"
DEFAULT_CHIMERAX = Path("/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX")
DEFAULT_ASSOCIATIONS = REPOSITORY_ROOT / (
    "interpretability/results/homology35_rerun/sae_transition_associations/"
    "sae_feature_associations.csv"
)
DEFAULT_SAE = (
    REPOSITORY_ROOT / "ml/results/homology35_rerun/frozen_saes/esmfold_matrix_topk64_seed42"
)
# Features whose structural-role test found significantly higher relative SASA
# at SAE hotspots than at controls (10,000 paired permutations, FDR < 0.05).
HIGHER_SASA_FEATURES = frozenset({375, 715, 1212, 1219, 1667, 1740, 2545, 2763, 2961, 3090, 3582})


@dataclass(frozen=True)
class RenderConfig:
    radius_angstrom: float = 8.0
    image_width: int = 1200
    image_height: int = 1000
    supersample: int = 3


def mapped_chimerax_residue(mapped: object, canonical_position: int) -> tuple[str, str]:
    """Map a 1-based canonical position to a ChimeraX chain/residue selector."""
    canonical_positions = np.asarray(mapped.canonical_positions)
    hits = np.flatnonzero(canonical_positions == canonical_position)
    if len(hits) != 1:
        raise ValueError(f"canonical residue {canonical_position} does not map uniquely")
    index = int(hits[0])
    insertion = str(mapped.insertion_codes[index] or "").strip()
    return str(mapped.chain_id), f"{mapped.auth_residue_numbers[index]}{insertion}"


def cache_structure(structure_id: str, cache_dir: Path) -> Path:
    """Download and cache the same biological assembly used by the analysis."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_code, _ = parse_structure_id(structure_id)
    cif_path = cache_dir / f"{pdb_code.lower()}-assembly1.cif"
    if cif_path.is_file() and cif_path.stat().st_size > 0:
        return cif_path
    gz_path = download_biological_assembly(pdb_code, cache_dir)
    with gzip.open(gz_path, "rb") as source, cif_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    if not cif_path.is_file() or cif_path.stat().st_size == 0:
        raise RuntimeError(f"empty assembly cache file for {structure_id}")
    return cif_path


def choose_structure_id(row: object) -> str:
    """Prefer the experimental transition-state-A structure, then any mapped structure."""
    for name in ("state_a_structure_id", "transition_state_a_structure_id"):
        value = getattr(row, name, None)
        if isinstance(value, str) and value.strip():
            return value
    raw = getattr(row, "structure_ids_json", None)
    if isinstance(raw, str) and raw:
        values = json.loads(raw)
        if values:
            return str(values[0])
    raise ValueError(f"no usable structure for {row.protein_id}")


def choose_second_structure_id(row: object) -> str:
    """Return the mapped state-B structure for a paired-conformation view."""
    for name in ("state_b_structure_id", "transition_state_b_structure_id"):
        value = getattr(row, name, None)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"no state-B structure for {row.protein_id}")


def build_render_examples(
    n_examples: int,
    seed: int,
    cache_dir: Path,
    device_name: str,
    require_two_conformations: bool = False,
) -> pd.DataFrame:
    """Build random frozen-test hotspot/control examples using analysis functions."""
    if n_examples < 1:
        raise ValueError("n_examples must be positive")
    rng = np.random.default_rng(seed)
    config = Config(
        seed=seed,
        features_per_track=10,
        hotspots_per_protein=5,
        controls_per_protein=5,
        sphere_radius_angstrom=8.0,
        device=device_name,
    )
    catalog = load_test_catalog(
        DEFAULT_CATALOG,
        DEFAULT_FULL_CATALOG,
        DEFAULT_TRANSITION_CATALOG,
        DEFAULT_TRANSITION_SUMMARY,
    )
    catalog = catalog.loc[catalog.dataset_label.eq(1)].copy()
    catalog = catalog.iloc[rng.permutation(len(catalog))].reset_index(drop=True)

    selected = select_feature_tracks(pd.read_csv(DEFAULT_ASSOCIATIONS), config.features_per_track)
    feature_ids = [
        int(feature_id)
        for feature_id in selected.feature_id
        if int(feature_id) in HIGHER_SASA_FEATURES
    ]
    if not feature_ids:
        raise ValueError("no higher-SASA structural features overlap association tracks")
    device = resolve_device(device_name)
    sae, center, _ = load_frozen_sae(DEFAULT_SAE, device)
    examples: list[dict[str, object]] = []

    for row in catalog.itertuples(index=False):
        if len(examples) >= n_examples:
            break
        try:
            feature_id = int(rng.choice(feature_ids))
            matrix = load_residue_matrix(
                Path(row.embedding_path),
                protein_id=row.protein_id,
                sequence=row.sequence,
                sequence_sha256=row.sequence_sha256,
                sequence_length=int(row.sequence_length),
            )
            activations = activation_vector(matrix, feature_id, sae, center, device)
            selections = pd.DataFrame(
                choose_positions(activations, row.protein_id, feature_id, config, row.sequence)
            )
            hotspots = selections.loc[selections.selection_kind.eq("hotspot")].sort_values(
                "activation_rank"
            )
            if hotspots.empty:
                raise ValueError("no SAE hotspot was selected")
            hotspot = hotspots.iloc[0]
            hotspot_index = int(hotspot.residue_index)
            mean_activation = float(np.mean(activations))
            indices = np.arange(len(activations))
            eligible = indices[(activations < mean_activation) & (indices != hotspot_index)]
            if len(eligible) == 0:
                raise ValueError("feature has no below-mean activation control")
            # Prefer a distant, strongly inactive residue so the two markers
            # remain visually separable on the whole-fold view.
            min_separation = max(10, len(activations) // 20)
            separated = eligible[np.abs(eligible - hotspot_index) >= min_separation]
            pool = separated if len(separated) else eligible
            control_index = int(pool[np.argmin(activations[pool])])
            control = pd.Series(
                {
                    "residue_index": control_index,
                    "activation": float(activations[control_index]),
                }
            )
            structure_id = choose_structure_id(row)
            structure_path = cache_structure(structure_id, cache_dir)
            mapped = resolve_mapped_structure(structure_id, structure_path, row.sequence)
            state_b_id = None
            state_b_path = None
            mapped_b = None
            if require_two_conformations:
                state_b_id = choose_second_structure_id(row)
                state_b_path = cache_structure(state_b_id, cache_dir)
                mapped_b = resolve_mapped_structure(state_b_id, state_b_path, row.sequence)
            hotspot_position = hotspot_index + 1
            control_position = control_index + 1
            hotspot_chain, hotspot_residue = mapped_chimerax_residue(mapped, hotspot_position)
            control_chain, control_residue = mapped_chimerax_residue(mapped, control_position)
            examples.append(
                {
                    "protein_id": row.protein_id,
                    "feature_id": feature_id,
                    "structure_id": structure_id,
                    "structure_path": str(structure_path),
                    "transition_position": hotspot_position,
                    "control_position": control_position,
                    "feature_mean_activation": mean_activation,
                    "transition_chain": hotspot_chain,
                    "transition_residue": hotspot_residue,
                    "control_chain": control_chain,
                    "control_residue": control_residue,
                    "transition_activation": float(hotspot.activation),
                    "control_activation": float(control.activation),
                    "transition_activation_rank": int(hotspot.activation_rank),
                    "control_below_feature_mean": True,
                    "control_sequence_distance": abs(control_index - hotspot_index),
                    "state_b_structure_id": state_b_id,
                    "state_b_structure_path": str(state_b_path) if state_b_path else None,
                    "state_b_mapped_chain": str(mapped_b.chain_id)
                    if mapped_b is not None
                    else None,
                    "state_b_hotspot_residue": (
                        mapped_chimerax_residue(mapped_b, hotspot_position)[1]
                        if mapped_b is not None
                        else None
                    ),
                    "state_b_control_residue": (
                        mapped_chimerax_residue(mapped_b, control_position)[1]
                        if mapped_b is not None
                        else None
                    ),
                }
            )
            print(f"selected {len(examples)}/{n_examples}: {row.protein_id}", flush=True)
        except Exception as error:
            print(f"skip {row.protein_id}: {error}", flush=True)

    result = pd.DataFrame(examples)
    if len(result) != n_examples:
        raise RuntimeError(
            f"only mapped {len(result)} of {n_examples} requested random test proteins"
        )
    return result


def resolve_chimerax(path: Path | None) -> Path:
    """Return a runnable ChimeraX executable, with a useful error if it is absent."""
    candidates = [path] if path is not None else [DEFAULT_CHIMERAX]
    executable = shutil.which("ChimeraX")
    if executable:
        candidates.append(Path(executable))
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"ChimeraX executable not found ({searched}); pass --chimerax PATH")


def cxc_quote(value: str | Path) -> str:
    """Quote a filesystem path for a ChimeraX command file."""
    return '"' + str(value).replace("\\", "/").replace('"', '\\"') + '"'


def chimerax_residue_spec(chain: str, residue: str) -> str:
    if not chain or not residue:
        raise ValueError("ChimeraX residue mapping requires non-empty chain and residue")
    return f"/{chain}:{residue}"


def region_commands(
    structure_path: str,
    chain: str,
    residue: str,
    region_color: str,
    output_path: Path,
    config: RenderConfig,
) -> str:
    """Build commands for one mapped neighborhood without launching ChimeraX."""
    center = chimerax_residue_spec(chain, residue)
    structure = Path(structure_path).resolve()
    output = output_path.resolve()
    return f"""open {cxc_quote(structure)}
hide atoms
hide surfaces
cartoon protein
color #b3b3b3 cartoons
select {center}
select zone sel {config.radius_angstrom:g} protein extend true residues true
show sel atoms
style sel stick
color sel {region_color}
surface sel color {region_color} transparency 40
select {center}
show sel atoms
style sel ball
color sel magenta
view sel pad 0.35
set bgColor white
select clear
save {cxc_quote(output)} width {config.image_width} height {config.image_height} supersample {config.supersample}
close all
"""


def run_chimerax_batch(chimerax: Path, commands: str, outputs: list[Path]) -> None:
    """Run one ChimeraX process for a complete batch of image commands."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cxc", delete=False) as handle:
        handle.write(commands + "\nexit\n")
        script_path = Path(handle.name)
    try:
        completed = subprocess.run(
            [str(chimerax), "--exit", "--script", str(script_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        script_path.unlink(missing_ok=True)
    missing = [path for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if completed.returncode != 0 or missing:
        detail = completed.stdout[-4000:]
        raise RuntimeError(f"ChimeraX batch failed; missing={missing[:3]}: {detail}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def make_pair(
    transition_path: Path, control_path: Path, output_path: Path, protein_id: str, feature_id: int
) -> None:
    with (
        Image.open(transition_path) as transition_source,
        Image.open(control_path) as control_source,
    ):
        transition = transition_source.convert("RGB")
        control = control_source.convert("RGB")
    header_height = 100
    canvas = Image.new(
        "RGB",
        (transition.width + control.width, max(transition.height, control.height) + header_height),
        "white",
    )
    canvas.paste(transition, (0, header_height))
    canvas.paste(control, (transition.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"SAE hotspot | feature {feature_id}", fill="black")
    draw.text((transition.width + 30, 20), "Matched low-activation control", fill="black")
    draw.text((30, 55), protein_id, fill="black")
    canvas.save(output_path, "PDF", resolution=300.0)


def whole_protein_commands(
    structure_path: str,
    hotspot_chain: str,
    hotspot_residue: str,
    control_chain: str,
    control_residue: str,
    output_path: Path,
    session_path: Path,
    title: str,
    config: RenderConfig,
) -> str:
    """Build one full-fold scene with hotspot and control highlighted together."""
    structure = Path(structure_path).resolve()
    output = output_path.resolve()
    hotspot = chimerax_residue_spec(hotspot_chain, hotspot_residue)
    control = chimerax_residue_spec(control_chain, control_residue)
    radius = 6.0
    return f"""open {cxc_quote(structure)}
rename #1 {cxc_quote(title)}
hide atoms
hide surfaces
cartoon protein
color #bdbdbd cartoons
select {hotspot}
select zone sel {radius:g} protein extend true residues true
show sel atoms
style sel stick
color sel red
select {control}
select zone sel {radius:g} protein extend true residues true
show sel atoms
style sel stick
color sel dodgerblue
select clear
set bgColor white
view
save {cxc_quote(output)} width {config.image_width} height {config.image_height} supersample {config.supersample}
save {cxc_quote(session_path)}
close all
"""


def make_whole_protein_page(
    image_path: Path,
    output_path: Path,
    protein_id: str,
    feature_id: int,
    hotspot_position: int,
    control_position: int,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    header_height = 92
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 14),
        f"Whole-protein SAE localization | feature {feature_id} | {protein_id}",
        fill="black",
    )
    draw.text(
        (24, 48),
        f"Hotspot residue {hotspot_position} (red)   Below-mean control {control_position} (blue)",
        fill="black",
    )
    canvas.save(output_path, "PDF", resolution=300.0)


def render_whole_protein_examples(
    examples: pd.DataFrame,
    output_dir: Path,
    chimerax: Path,
    config: RenderConfig,
    index_offset: int = 0,
) -> pd.DataFrame:
    """Render all whole-protein examples in one ChimeraX process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sae-whole-protein-") as temporary:
        staging = Path(temporary)
        jobs: list[tuple[object, Path, Path, Path]] = []
        commands: list[str] = []
        for index, row in enumerate(examples.itertuples(index=False), start=index_offset):
            prefix = f"{index:03d}_{safe_name(row.protein_id)}_feature{row.feature_id}"
            png = staging / f"{prefix}.png"
            pdf = output_dir / f"{prefix}_whole_protein.pdf"
            session = output_dir / f"{prefix}_whole_protein.cxs"
            commands.append(
                whole_protein_commands(
                    row.structure_path,
                    row.transition_chain,
                    row.transition_residue,
                    row.control_chain,
                    row.control_residue,
                    png,
                    session,
                    pdf.stem,
                    config,
                )
            )
            jobs.append((row, png, pdf, session))
        run_chimerax_batch(
            chimerax,
            "\n".join(commands),
            [path for _, png, _, session in jobs for path in (png, session)],
        )
        for index, (row, png, pdf, session) in enumerate(jobs):
            make_whole_protein_page(
                png,
                pdf,
                row.protein_id,
                int(row.feature_id),
                int(row.transition_position),
                int(row.control_position),
            )
            rendered.append(
                {**row._asdict(), "whole_protein_pdf": str(pdf), "chimerax_session": str(session)}
            )
            print(f"rendered whole-protein {index + 1}/{len(examples)}: {pdf.name}", flush=True)
    result = pd.DataFrame(rendered)
    manifest_path = output_dir / "whole_protein_examples.csv"
    if index_offset and manifest_path.is_file():
        result = pd.concat([pd.read_csv(manifest_path), result], ignore_index=True)
    result.to_csv(manifest_path, index=False)
    return result


def two_conformation_commands(
    row: object,
    hotspot_chain_b: str,
    output_a: Path,
    output_b: Path,
    session_path: Path,
    title: str,
    config: RenderConfig,
) -> str:
    """Render paired state-A/state-B folds and preserve both in a session."""
    radius = 6.0
    hotspot_a = chimerax_residue_spec(row.transition_chain, row.transition_residue)
    control_a = chimerax_residue_spec(row.control_chain, row.control_residue)
    hotspot_b = chimerax_residue_spec(hotspot_chain_b, row.state_b_hotspot_residue)
    control_b = chimerax_residue_spec(hotspot_chain_b, row.state_b_control_residue)

    def scene(structure: str, hotspot: str, control: str, output: Path, rename: str) -> str:
        return f"""open {cxc_quote(structure)}
rename #1 {cxc_quote(rename)}
hide atoms
hide surfaces
cartoon protein
color #bdbdbd cartoons
select {hotspot}
select zone sel {radius:g} protein extend true residues true
show sel atoms
style sel stick
color sel red
select {control}
select zone sel {radius:g} protein extend true residues true
show sel atoms
style sel stick
color sel dodgerblue
select clear
set bgColor white
view
save {cxc_quote(output)} width {config.image_width} height {config.image_height} supersample {config.supersample}
"""

    return (
        scene(row.structure_path, hotspot_a, control_a, output_a, f"{title} state A")
        + scene(row.state_b_structure_path, hotspot_b, control_b, output_b, f"{title} state B")
        + f"view all\nsave {cxc_quote(session_path)}\nclose all\n"
    )


def make_two_conformation_page(
    image_a: Path,
    image_b: Path,
    output_path: Path,
    protein_id: str,
    feature_id: int,
    hotspot_position: int,
    control_position: int,
) -> None:
    with Image.open(image_a) as source_a, Image.open(image_b) as source_b:
        state_a = source_a.convert("RGB")
        state_b = source_b.convert("RGB")
    header_height = 92
    canvas = Image.new(
        "RGB",
        (state_a.width + state_b.width, max(state_a.height, state_b.height) + header_height),
        "white",
    )
    canvas.paste(state_a, (0, header_height))
    canvas.paste(state_b, (state_a.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 14), f"State A | feature {feature_id} | {protein_id}", fill="black")
    draw.text((state_a.width + 24, 14), f"State B | feature {feature_id}", fill="black")
    draw.text(
        (24, 48),
        f"Hotspot {hotspot_position} (red)   Below-mean control {control_position} (blue)",
        fill="black",
    )
    canvas.save(output_path, "PDF", resolution=300.0)


def render_two_conformation_examples(
    examples: pd.DataFrame,
    output_dir: Path,
    chimerax: Path,
    config: RenderConfig,
) -> pd.DataFrame:
    """Render paired conformations in one ChimeraX batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sae-two-conformation-") as temporary:
        staging = Path(temporary)
        jobs: list[tuple[object, Path, Path, Path, Path]] = []
        commands: list[str] = []
        for index, row in enumerate(examples.itertuples(index=False)):
            prefix = f"{index:03d}_{safe_name(row.protein_id)}_feature{row.feature_id}"
            image_a = staging / f"{prefix}_state_a.png"
            image_b = staging / f"{prefix}_state_b.png"
            pdf = output_dir / f"{prefix}_two_conformations.pdf"
            session = output_dir / f"{prefix}_two_conformations.cxs"
            commands.append(
                two_conformation_commands(
                    row,
                    row.state_b_mapped_chain,
                    image_a,
                    image_b,
                    session,
                    pdf.stem,
                    config,
                )
            )
            jobs.append((row, image_a, image_b, pdf, session))
        run_chimerax_batch(
            chimerax,
            "\n".join(commands),
            [
                path
                for _, image_a, image_b, _, session in jobs
                for path in (image_a, image_b, session)
            ],
        )
        for index, (row, image_a, image_b, pdf, session) in enumerate(jobs):
            make_two_conformation_page(
                image_a,
                image_b,
                pdf,
                row.protein_id,
                int(row.feature_id),
                int(row.transition_position),
                int(row.control_position),
            )
            rendered.append(
                {
                    **row._asdict(),
                    "two_conformation_pdf": str(pdf),
                    "chimerax_session": str(session),
                }
            )
            print(f"rendered two-conformation {index + 1}/{len(examples)}: {pdf.name}", flush=True)
    result = pd.DataFrame(rendered)
    result.to_csv(output_dir / "two_conformation_examples.csv", index=False)
    return result


def render_examples(
    examples: pd.DataFrame, output_dir: Path, chimerax: Path, config: RenderConfig
) -> pd.DataFrame:
    """Render paired hotspot/control panels and save their provenance table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="sae-figure-panels-") as temporary:
        staging = Path(temporary)
        jobs: list[tuple[object, Path, Path, Path]] = []
        commands: list[str] = []
        for index, row in enumerate(examples.itertuples(index=False)):
            prefix = f"{index:03d}_{safe_name(row.protein_id)}_feature{row.feature_id}"
            hotspot_png = staging / f"{prefix}_hotspot.png"
            control_png = staging / f"{prefix}_control.png"
            pair_pdf = output_dir / f"{prefix}_pair.pdf"
            commands.append(
                region_commands(
                    row.structure_path,
                    row.transition_chain,
                    row.transition_residue,
                    "dodgerblue",
                    hotspot_png,
                    config,
                )
            )
            commands.append(
                region_commands(
                    row.structure_path,
                    row.control_chain,
                    row.control_residue,
                    "orange",
                    control_png,
                    config,
                )
            )
            jobs.append((row, hotspot_png, control_png, pair_pdf))

        run_chimerax_batch(
            chimerax,
            "\n".join(commands),
            [path for _, hotspot, control, _ in jobs for path in (hotspot, control)],
        )
        for index, (row, hotspot_png, control_png, pair_pdf) in enumerate(jobs):
            make_pair(hotspot_png, control_png, pair_pdf, row.protein_id, int(row.feature_id))
            rendered.append({**row._asdict(), "pair_pdf": str(pair_pdf)})
            print(f"rendered {index + 1}/{len(examples)}: {pair_pdf.name}", flush=True)
    result = pd.DataFrame(rendered)
    result.to_csv(output_dir / "rendered_examples.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chimerax", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--image-width", type=int, default=1200)
    parser.add_argument("--image-height", type=int, default=1000)
    parser.add_argument(
        "--whole-protein",
        action="store_true",
        help="Also render full-fold hotspot/control views; existing close-up PDFs are preserved.",
    )
    parser.add_argument(
        "--whole-only",
        action="store_true",
        help="Skip close-up rendering and only render the full-fold views.",
    )
    parser.add_argument(
        "--whole-output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "writeup/figures/sae_region_candidates_whole_protein",
    )
    parser.add_argument(
        "--both-conformations",
        action="store_true",
        help="Render state-A/state-B whole-fold comparisons in a separate output folder.",
    )
    parser.add_argument(
        "--both-output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "writeup/figures/sae_region_candidates_two_conformations",
    )
    args = parser.parse_args()
    if (
        args.n_examples < 1
        or args.index_offset < 0
        or args.radius <= 0
        or args.image_width < 1
        or args.image_height < 1
    ):
        parser.error("example count and image dimensions must be positive, and radius must be > 0")
    chimerax = resolve_chimerax(args.chimerax)
    examples = build_render_examples(
        args.n_examples,
        args.seed,
        args.cache_dir,
        args.device,
        require_two_conformations=args.both_conformations,
    )
    if args.both_conformations:
        rendered = render_two_conformation_examples(
            examples,
            args.both_output_dir,
            chimerax,
            RenderConfig(args.radius, args.image_width, args.image_height),
        )
        print(
            f"Rendered {len(rendered)} two-conformation figures in {args.both_output_dir.resolve()}",
            flush=True,
        )
        return
    if args.whole_only and not args.whole_protein:
        parser.error("--whole-only requires --whole-protein")
    if not args.whole_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        examples.to_csv(args.output_dir / "selected_examples.csv", index=False)
        rendered = render_examples(
            examples,
            args.output_dir,
            chimerax,
            RenderConfig(args.radius, args.image_width, args.image_height),
            index_offset=args.index_offset,
        )
        print(f"Rendered {len(rendered)} paired figures in {args.output_dir.resolve()}", flush=True)
    if args.whole_protein:
        whole = render_whole_protein_examples(
            examples,
            args.whole_output_dir,
            chimerax,
            RenderConfig(args.radius, args.image_width, args.image_height),
        )
        print(
            f"Rendered {len(whole)} whole-protein figures in {args.whole_output_dir.resolve()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
