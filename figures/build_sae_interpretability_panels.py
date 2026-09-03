"""Render compact SAE transition and cross-representation structural panels.

Panel A shows ESMFold SAE feature associations with residue displacement and
PRS. Panel B is intentionally reserved. Panel C matches separately trained
ESMFold and BioEMU SAE features by their sparse activation profiles across the
same held-out residues, then compares their individual structural effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ESMFOLD_ASSOCIATIONS = (
    ROOT
    / "interpretability/results/homology35_rerun/sae_transition_associations/sae_feature_associations.csv"
)
DEFAULT_ESMFOLD_TRANSITIONS = DEFAULT_ESMFOLD_ASSOCIATIONS.parent
DEFAULT_BIOEMU_TRANSITIONS = (
    ROOT / "interpretability/results/homology35_bioemu_8572_sae_transition_associations_perm10000"
)
DEFAULT_ESMFOLD_STRUCTURAL = ROOT / "interpretability/results/homology35_rerun/sae_structural_roles"
DEFAULT_BIOEMU_STRUCTURAL = (
    ROOT / "interpretability/results/homology35_bioemu_8572_sae_structural_roles"
)
DEFAULT_OUTPUT = ROOT / "AA-upgraded-neurips-workshop/figures"
DEFAULT_PANEL_B_IMAGE = (
    ROOT / "AA-upgraded-neurips-workshop/figures/sae_region_candidates_two_conformations/"
    "TAGGED_014_pathpre_MS_MS_pair_209_feature2961_two_conformations_cropped.png"
)
FDR_ALPHA = 0.05
MIN_FEATURE_COSINE = 0.205
MATCH_KEYS = ("protein_id", "sequence_sha256", "canonical_residue_number")
TRACK_COLORS = {
    "prs": "#2878b5",
    "rmsd_displacement": "#d95f02",
    "prs;rmsd_displacement": "#7a5195",
}
TRACK_LABELS = {
    "prs": "PRS-selected",
    "rmsd_displacement": "Displacement-selected",
    "prs;rmsd_displacement": "Both-selected",
}
METRICS = {
    "sasa_angstrom2": {"label": "SASA", "marker": "o", "unit": "Å²"},
    "contact_density": {
        "label": "Direct-contact count",
        "marker": "s",
        "unit": "residues",
    },
}


def _load_associations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "feature_id",
        "displacement_balanced_spearman",
        "prs_balanced_spearman",
        "displacement_fdr",
        "prs_fdr",
    }
    if missing := required - set(frame):
        raise ValueError(f"association table missing columns: {sorted(missing)}")
    if len(frame) != 4096 or frame.feature_id.nunique() != 4096:
        raise ValueError("expected exactly 4,096 unique SAE features")
    return frame


def _selected_structural_effects(directory: Path, model: str) -> pd.DataFrame:
    selected = pd.read_csv(directory / "selected_features.csv")
    effects = pd.read_csv(directory / "paired_permutation_tests.csv")
    required_selected = {"feature_id", "selection_tracks"}
    required_effects = {"feature_id", "metric", "hotspot_minus_control_mean"}
    if missing := required_selected - set(selected):
        raise ValueError(f"{model} selected-feature table missing columns: {sorted(missing)}")
    if missing := required_effects - set(effects):
        raise ValueError(f"{model} structural-effects table missing columns: {sorted(missing)}")
    result = selected.merge(effects, on="feature_id", validate="one_to_many")
    result = result.loc[result.metric.isin(METRICS)].copy()
    expected = selected.feature_id.nunique() * len(METRICS)
    if len(result) != expected:
        raise ValueError(
            f"{model} structural results lack SASA or direct-contact count for selected features"
        )
    return result


def _load_sparse_activations(directory: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    residue_index = pd.read_parquet(directory / "residue_index.parquet")
    archive = np.load(directory / "sparse_activations.npz")
    feature_indices = archive["feature_indices"]
    activation_values = archive["activation_values"]
    if (
        len(residue_index) != len(feature_indices)
        or feature_indices.shape != activation_values.shape
    ):
        raise ValueError(f"invalid sparse activation archive: {directory}")
    if feature_indices.shape[1] != 64:
        raise ValueError("expected top-64 sparse SAE activations")
    return residue_index, feature_indices, activation_values


def _selected_activation_matrix(
    feature_indices: np.ndarray,
    activation_values: np.ndarray,
    rows: np.ndarray,
    feature_ids: np.ndarray,
) -> coo_matrix:
    feature_lookup = np.full(4096, -1, dtype=np.int32)
    feature_lookup[feature_ids] = np.arange(len(feature_ids), dtype=np.int32)
    selected_indices = feature_lookup[feature_indices[rows]]
    selected_values = activation_values[rows]
    source_rows = np.repeat(np.arange(len(rows), dtype=np.int32), feature_indices.shape[1])
    selected_indices = selected_indices.ravel()
    selected_values = selected_values.ravel()
    keep = selected_indices >= 0
    return coo_matrix(
        (selected_values[keep], (source_rows[keep], selected_indices[keep])),
        shape=(len(rows), len(feature_ids)),
    ).tocsr()


def _match_selected_features(
    esmfold_transition_directory: Path,
    bioemu_transition_directory: Path,
    esmfold_features: np.ndarray,
    bioemu_features: np.ndarray,
) -> pd.DataFrame:
    esm_index, esm_ids, esm_values = _load_sparse_activations(esmfold_transition_directory)
    bio_index, bio_ids, bio_values = _load_sparse_activations(bioemu_transition_directory)
    for name, index in (("ESMFold", esm_index), ("BioEMU", bio_index)):
        if missing := set(MATCH_KEYS) - set(index):
            raise ValueError(f"{name} residue index missing identity keys: {sorted(missing)}")
    shared = esm_index.reset_index(names="esmfold_row").merge(
        bio_index.reset_index(names="bioemu_row"),
        on=list(MATCH_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(shared) < 100_000:
        raise ValueError(f"too few shared test residues for feature matching: {len(shared)}")
    esm_matrix = _selected_activation_matrix(
        esm_ids, esm_values, shared.esmfold_row.to_numpy(), esmfold_features
    )
    bio_matrix = _selected_activation_matrix(
        bio_ids, bio_values, shared.bioemu_row.to_numpy(), bioemu_features
    )
    esm_norm = np.sqrt(esm_matrix.multiply(esm_matrix).sum(axis=0)).A1
    bio_norm = np.sqrt(bio_matrix.multiply(bio_matrix).sum(axis=0)).A1
    similarities = (esm_matrix.T @ bio_matrix).toarray()
    similarities /= np.maximum(esm_norm[:, None] * bio_norm[None, :], 1e-12)
    row_ids, column_ids = linear_sum_assignment(-similarities)
    matches = pd.DataFrame(
        {
            "esmfold_feature_id": esmfold_features[row_ids].astype(int),
            "bioemu_feature_id": bioemu_features[column_ids].astype(int),
            "activation_cosine": similarities[row_ids, column_ids],
            "n_shared_residues": len(shared),
        }
    )
    return matches.loc[matches.activation_cosine.ge(MIN_FEATURE_COSINE)].copy()


def _comparison_table(
    esmfold_structural_directory: Path,
    bioemu_structural_directory: Path,
    esmfold_transition_directory: Path,
    bioemu_transition_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    esmfold = _selected_structural_effects(esmfold_structural_directory, "ESMFold")
    bioemu = _selected_structural_effects(bioemu_structural_directory, "BioEMU")
    matches = _match_selected_features(
        esmfold_transition_directory,
        bioemu_transition_directory,
        esmfold.feature_id.drop_duplicates().to_numpy(dtype=int),
        bioemu.feature_id.drop_duplicates().to_numpy(dtype=int),
    )
    if len(matches) < 8:
        raise ValueError("fewer than eight adequately matched SAE features")
    esmfold = esmfold.rename(
        columns={
            "feature_id": "esmfold_feature_id",
            "selection_tracks": "esmfold_selection_tracks",
            "hotspot_minus_control_mean": "effect_esmfold",
        }
    )
    bioemu = bioemu.rename(
        columns={
            "feature_id": "bioemu_feature_id",
            "selection_tracks": "bioemu_selection_tracks",
            "hotspot_minus_control_mean": "effect_bioemu",
        }
    )
    table = matches.merge(
        esmfold[["esmfold_feature_id", "esmfold_selection_tracks", "metric", "effect_esmfold"]],
        on="esmfold_feature_id",
        validate="one_to_many",
    ).merge(
        bioemu[["bioemu_feature_id", "bioemu_selection_tracks", "metric", "effect_bioemu"]],
        on=["bioemu_feature_id", "metric"],
        validate="one_to_one",
    )
    return table, matches


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    maximum = float(np.max(np.abs(values)))
    return (
        (-1.0, 1.0)
        if not np.isfinite(maximum) or maximum <= 0
        else (-1.15 * maximum, 1.15 * maximum)
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _format_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8.0)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)


def _render_association_panel(axis: plt.Axes, associations: pd.DataFrame) -> dict[str, int]:
    displacement = associations.displacement_fdr.lt(FDR_ALPHA)
    prs = associations.prs_fdr.lt(FDR_ALPHA)
    classes = {
        "Neither": ~(displacement | prs),
        "Displacement only": displacement & ~prs,
        "PRS only": ~displacement & prs,
        "Both": displacement & prs,
    }
    styles = {
        "Neither": {"color": "#b8b8b8", "size": 5, "alpha": 0.35},
        "Displacement only": {"color": "#4c78a8", "size": 8, "alpha": 0.7},
        "PRS only": {"color": "#f28e2b", "size": 8, "alpha": 0.7},
        "Both": {"color": "#d62728", "size": 19, "alpha": 0.9},
    }
    for name in classes:
        style = styles[name]
        axis.scatter(
            associations.loc[classes[name], "displacement_balanced_spearman"],
            associations.loc[classes[name], "prs_balanced_spearman"],
            c=style["color"],
            s=style["size"],
            alpha=style["alpha"],
            linewidths=0.35 if name == "Both" else 0,
            edgecolors="#202020" if name == "Both" else None,
            label=f"{name} (n={int(classes[name].sum())})",
        )
    axis.axhline(0, color="#7f7f7f", linewidth=0.65, linestyle="--", zorder=0)
    axis.axvline(0, color="#7f7f7f", linewidth=0.65, linestyle="--", zorder=0)
    axis.set(
        xlabel="Displacement association (balanced Spearman ρ)",
        ylabel="PRS association (balanced Spearman ρ)",
    )
    axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.5,
        markerscale=0.7,
        borderpad=0.1,
        handletextpad=0.3,
        labelspacing=0.2,
        ncol=2,
    )
    return {name: int(mask.sum()) for name, mask in classes.items()}


def _render_structural_panel(axis: plt.Axes, table: pd.DataFrame) -> plt.Axes:
    sasa = table.loc[table.metric.eq("sasa_angstrom2")]
    direct_contacts = table.loc[table.metric.eq("contact_density")]
    if len(sasa) != len(direct_contacts) or sasa.empty:
        raise ValueError("structural feature matches must have both metrics")
    sasa_limits = _symmetric_limits(np.concatenate((sasa.effect_esmfold, sasa.effect_bioemu)))
    axis.set(
        xlim=sasa_limits,
        ylim=sasa_limits,
        xlabel="ESMFold SASA effect (Å²)",
        ylabel="BioEMU SASA effect (Å²)",
    )
    for row in sasa.itertuples(index=False):
        axis.scatter(
            row.effect_esmfold,
            row.effect_bioemu,
            color=TRACK_COLORS[row.esmfold_selection_tracks],
            marker="o",
            s=19,
            edgecolors="#222222",
            linewidths=0.25,
            alpha=0.88,
            zorder=3,
        )
    axis.plot(sasa_limits, sasa_limits, color="#737373", linewidth=0.75, linestyle="--", zorder=1)
    axis.axhline(0, color="#9a9a9a", linewidth=0.55, zorder=0)
    axis.axvline(0, color="#9a9a9a", linewidth=0.55, zorder=0)
    density_axis = axis.figure.add_axes(
        axis.get_position(), frameon=False, label="direct_contact_effects"
    )
    density_axis.patch.set_alpha(0)
    density_limits = _symmetric_limits(
        np.concatenate((direct_contacts.effect_esmfold, direct_contacts.effect_bioemu))
    )
    density_axis.set(
        xlim=density_limits,
        ylim=density_limits,
        xlabel="ESMFold direct-contact count",
        ylabel="BioEMU direct-contact count",
    )
    for row in direct_contacts.itertuples(index=False):
        density_axis.scatter(
            row.effect_esmfold,
            row.effect_bioemu,
            color=TRACK_COLORS[row.esmfold_selection_tracks],
            marker="s",
            s=19,
            edgecolors="#222222",
            linewidths=0.25,
            alpha=0.88,
            zorder=3,
        )
    density_axis.plot(
        density_limits, density_limits, color="#737373", linewidth=0.75, linestyle="--", zorder=1
    )
    density_axis.axhline(0, color="#9a9a9a", linewidth=0.55, zorder=0)
    density_axis.axvline(0, color="#9a9a9a", linewidth=0.55, zorder=0)
    density_axis.xaxis.set_ticks_position("top")
    density_axis.yaxis.set_ticks_position("right")
    density_axis.xaxis.set_label_position("top")
    density_axis.yaxis.set_label_position("right")
    density_axis.tick_params(
        axis="both", which="both", labelsize=7.5, top=True, right=True, bottom=False, left=False
    )
    density_axis.xaxis.label.set_size(8.0)
    density_axis.yaxis.label.set_size(8.0)
    density_axis.spines["bottom"].set_visible(False)
    density_axis.spines["left"].set_visible(False)
    legend = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=3.5,
            label=TRACK_LABELS[track],
        )
        for track, color in TRACK_COLORS.items()
    ] + [
        Line2D([], [], color="#333333", marker="o", linestyle="None", markersize=3.5, label="SASA"),
        Line2D(
            [],
            [],
            color="#333333",
            marker="s",
            linestyle="None",
            markersize=3.5,
            label="Direct-contact count",
        ),
    ]
    axis.legend(
        handles=legend,
        loc="lower right",
        frameon=False,
        fontsize=6.2,
        ncol=2,
        borderpad=0.1,
        handletextpad=0.3,
        labelspacing=0.2,
    )
    return density_axis


def render(
    associations_path: Path,
    esmfold_structural_directory: Path,
    bioemu_structural_directory: Path,
    esmfold_transition_directory: Path,
    bioemu_transition_directory: Path,
    output_directory: Path,
    panel_b_image: Path = DEFAULT_PANEL_B_IMAGE,
    panel_labels: tuple[str, str, str] = ("A", "B", "C"),
) -> dict[str, object]:
    associations = _load_associations(associations_path)
    comparison, matches = _comparison_table(
        esmfold_structural_directory,
        bioemu_structural_directory,
        esmfold_transition_directory,
        bioemu_transition_directory,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12.4, 8.2))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.0, 2.0), height_ratios=(1.0, 1.0), hspace=0.30, wspace=0.30
    )
    association_axis, blank_axis, structural_axis = (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    for axis, label in zip(
        (association_axis, blank_axis, structural_axis), panel_labels, strict=True
    ):
        _panel_label(axis, label)
    _format_axis(association_axis)
    _format_axis(structural_axis)
    association_counts = _render_association_panel(association_axis, associations)
    blank_axis.set_xticks([])
    blank_axis.set_yticks([])
    blank_axis.grid(False)
    blank_axis.tick_params(bottom=False, left=False)
    if not panel_b_image.is_file():
        raise FileNotFoundError(f"Panel B image not found: {panel_b_image}")
    blank_axis.imshow(plt.imread(panel_b_image), aspect="equal")
    blank_axis.set_aspect("equal", adjustable="box")
    density_axis = _render_structural_panel(structural_axis, comparison)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.94, hspace=0.22, wspace=0.34)
    density_axis.set_position(structural_axis.get_position())
    stem = output_directory / "small_sae_interpretability_panels"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "panel_a": {
            "association_table": str(associations_path.relative_to(ROOT)),
            "features": int(len(associations)),
            "fdr_alpha": FDR_ALPHA,
            "class_counts": association_counts,
        },
        "panel_b": {
            "status": "two_conformation_structure_example",
            "source_pdf": str(panel_b_image.with_suffix(".pdf").relative_to(ROOT)),
            "rasterized_source": str(panel_b_image.relative_to(ROOT)),
        },
        "panel_c": {
            "estimand": "individual matched-feature hotspot-minus-control structural effects",
            "match_method": "one-to-one maximum-cosine assignment on shared sparse test-residue activations",
            "minimum_activation_cosine": MIN_FEATURE_COSINE,
            "metrics": list(METRICS),
            "n_matched_features": len(matches),
            "matches": matches.to_dict(orient="records"),
            "points": comparison.to_dict(orient="records"),
        },
        "outputs": [str(stem.with_suffix(suffix).relative_to(ROOT)) for suffix in (".pdf", ".svg")],
    }
    (output_directory / "small_sae_interpretability_panels.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esmfold-associations", type=Path, default=DEFAULT_ESMFOLD_ASSOCIATIONS)
    parser.add_argument("--esmfold-structural", type=Path, default=DEFAULT_ESMFOLD_STRUCTURAL)
    parser.add_argument("--bioemu-structural", type=Path, default=DEFAULT_BIOEMU_STRUCTURAL)
    parser.add_argument("--esmfold-transitions", type=Path, default=DEFAULT_ESMFOLD_TRANSITIONS)
    parser.add_argument("--bioemu-transitions", type=Path, default=DEFAULT_BIOEMU_TRANSITIONS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-b-image", type=Path, default=DEFAULT_PANEL_B_IMAGE)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.esmfold_associations.resolve(),
                args.esmfold_structural.resolve(),
                args.bioemu_structural.resolve(),
                args.esmfold_transitions.resolve(),
                args.bioemu_transitions.resolve(),
                args.output_directory.resolve(),
                args.panel_b_image.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
