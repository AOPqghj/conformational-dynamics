"""Render the six manuscript panels directly on one vector Matplotlib canvas.

This intentionally does *not* composite, crop, or rasterize the two existing
PDF pages.  Every plot is drawn from its original data and axes code into one
GridSpec, which keeps all panels sharp and gives the rows fixed shared heights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FIGURE_ROOT))

import build_frozen_model_routing_curves as frozen  # noqa: E402
import build_sae_interpretability_panels as sae  # noqa: E402
from protein_state_router.experiments.benchmark import (  # noqa: E402
    BenchmarkConfig,
    run_benchmark,
    sequence_feature_matrix,
)

OUTPUT = FIGURE_ROOT / "small_combined_frozen_sae_panels.pdf"
COMMON_TEST_COVARIATE_ROOT = (
    ROOT / "ml/results/homology35_rerun/common_test_covariate_models"
)


def _label(axis: plt.Axes, label: str, *, y: float = 1.04) -> None:
    axis.text(
        0.0, y, label, transform=axis.transAxes, fontsize=10.5,
        fontweight="bold", ha="left", va="bottom", clip_on=False,
    )


def _format(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8.0)
    axis.xaxis.label.set_size(8.5)
    axis.yaxis.label.set_size(8.5)


def _common_test_covariates(
    models: list[frozen.ModelPredictions],
) -> list[frozen.ModelPredictions]:
    """Fit/load covariate baselines on the frozen models' exact split.

    The previous figure pooled covariate predictions across repeated splits,
    which changed both test membership and per-protein weighting.  These two
    baselines instead use the catalog's locked train/validation/test column and
    are required to have the same held-out proteins and labels as the frozen
    embedding models before Panel A is rendered.
    """
    catalog = pd.read_parquet(frozen.DEFAULT_CATALOG).copy()
    required = {"protein_id", "dataset_label", "split", "alphafold_mean_plddt"}
    if missing := required - set(catalog):
        raise ValueError(f"frozen catalog missing columns: {sorted(missing)}")
    if set(catalog.split) != {"train", "val", "test"}:
        raise ValueError("frozen catalog must retain train, val, and test partitions")
    sequence, names = sequence_feature_matrix(catalog)
    train = catalog.split.eq("train").to_numpy()
    plddt = catalog.alphafold_mean_plddt.to_numpy(dtype=np.float32)
    training_median = float(np.nanmedian(plddt[train]))
    if not np.isfinite(training_median):
        raise ValueError("frozen training split has no finite pLDDT values")
    plddt = np.where(np.isfinite(plddt), plddt, training_median).astype(np.float32)
    features = np.column_stack((sequence, plddt))
    feature_names = (*names, "alphafold_mean_plddt_train_median_imputed")
    definitions = (
        ("Covariate-only linear", "#66a61e", "linear", "linear_covariates"),
        ("Covariate-only tree", "#e6ab02", "tree", "tree_covariates"),
    )
    outputs: list[frozen.ModelPredictions] = []
    for label, color, family, directory_name in definitions:
        directory = COMMON_TEST_COVARIATE_ROOT / directory_name
        prediction_path = directory / "test_predictions.parquet"
        selection_path = directory / "validation_selection.json"
        if not (prediction_path.is_file() and selection_path.is_file()):
            run_benchmark(
                frozen.DEFAULT_CATALOG,
                directory,
                BenchmarkConfig(
                    family=family,
                    random_seed=42,
                    search="standard",
                    cpu_threads=1,
                    save_model=False,
                ),
                features=features,
                feature_names=feature_names,
                dataset_reference=str(frozen.DEFAULT_CATALOG.relative_to(ROOT)),
            )
        outputs.append(frozen.load_predictions(label, color, directory))
    frozen.verify_shared_test_set([*models, *outputs])
    return outputs


def _frozen_inputs() -> tuple[list[frozen.ModelPredictions], list[frozen.ModelPredictions], object, frozen.ModelPredictions, object]:
    results = frozen.DEFAULT_RESULTS
    confounders = frozen.DEFAULT_CONFOUNDER_RESULTS
    cnn_directory = frozen.choose_best_cnn(results)
    models = [
        frozen.load_predictions("Logistic regression", "#1b9e77", results / "pooled_frozen_models/esmfold_single_linear"),
        frozen.load_predictions("Histogram gradient tree", "#d95f02", results / "pooled_frozen_models/esmfold_single_tree"),
        frozen.load_predictions("Full-matrix CNN", "#7570b3", cnn_directory),
    ]
    covariates = _common_test_covariates(models)
    return models, covariates, frozen.load_residual_panel(confounders), models[-1], frozen.load_catalog(frozen.DEFAULT_CATALOG)


def _draw_frozen(roc: plt.Axes, residual: plt.Axes, plddt: plt.Axes) -> None:
    models, covariates, residuals, cnn, catalog = _frozen_inputs()
    frozen.verify_shared_test_set([*models, *covariates])
    for axis, letter in ((roc, "A"), (residual, "B"), (plddt, "C")):
        _label(axis, letter, y=1.10 if letter in {"B", "C"} else 1.04)
        _format(axis)

    roc.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.1, label="Random")
    for model in [*models, *covariates]:
        y = model.predictions.dataset_label.to_numpy(dtype=int)
        probability = model.predictions.probability.to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, probability)
        roc.plot(
            fpr, tpr, color=model.color,
            linewidth=2.2 if model in models else 1.8,
            linestyle="-" if model in models else "--",
            label=f"{model.label} (AUROC {roc_auc_score(y, probability):.3f})",
        )
    roc.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))
    roc.legend(loc="lower right", frameon=False, fontsize=6.2, ncol=2)

    categories = list(frozen.RESIDUAL_CATEGORY_ORDER)
    x = np.arange(len(categories))
    colors = {"linear": "#1b9e77", "tree": "#d95f02"}
    for (family, seed), group in residuals.groupby(["family", "seed"], sort=True):
        values = group.set_index("category").reindex(categories).test_auroc
        if values.isna().any():
            raise ValueError(f"incomplete residual AUROC categories for {family} seed {seed}")
        residual.plot(x, values.to_numpy(float), marker="o", linewidth=1.4, linestyle=":", color=colors[family], alpha=0.8, label=f"{family.title()} split {seed}")
    residual.set(ylabel="Test AUROC", xticks=x, xticklabels=["All covariates\nremoved", "pLDDT\nremoved", "Raw\nembedding"], ylim=(0.5, 0.9))
    residual.legend(loc="lower left", frameon=False, fontsize=5.8, ncol=1)

    points = cnn.predictions.merge(catalog, on="protein_id", how="inner", validate="one_to_one", suffixes=("_prediction", "_catalog"))
    points = points.loc[points.alphafold_mean_plddt.ge(70)].copy()
    plddt.scatter(points.probability, points.alphafold_mean_plddt, c=points.dataset_label_prediction.map({0: "#4c78a8", 1: "#e45756"}), alpha=0.72, s=11, linewidth=0.25, edgecolors="white")
    plddt.axhline(70, color="#777777", linewidth=0.9, linestyle="--")
    plddt.axhline(90, color="#777777", linewidth=0.9, linestyle=":")
    plddt.set(xlabel="Multistate probability", ylabel="Mean pLDDT", xlim=(0, 1))
    plddt.yaxis.set_label_coords(-0.09, 0.5)
    style = dict(boxstyle="round,pad=0.3", facecolor="#BFBFD6", edgecolor="#676775", alpha=0.5)
    plddt.text(0.02, 0.075, "medium", transform=plddt.transAxes, va="top", fontsize=8.0, bbox=style, weight="bold")
    plddt.text(0.02, 0.705, "high", transform=plddt.transAxes, va="top", fontsize=8.0, bbox=style, weight="bold")
    plddt.scatter([], [], color="#4c78a8", label="Static")
    plddt.scatter([], [], color="#e45756", label="Dynamic")
    plddt.legend(loc="lower right", frameon=True, framealpha=0.3, facecolor="#BFBFD6", prop={"size": 5.5, "weight": "bold"}, markerscale=0.35, borderpad=0.25, handlelength=0.8, handletextpad=0.25, labelspacing=0.15)


def _draw_sae(
    association: plt.Axes, structural: plt.Axes, protein: plt.Axes
) -> plt.Axes:
    associations = sae._load_associations(sae.DEFAULT_ESMFOLD_ASSOCIATIONS)
    comparison, _ = sae._comparison_table(sae.DEFAULT_ESMFOLD_STRUCTURAL, sae.DEFAULT_BIOEMU_STRUCTURAL, sae.DEFAULT_ESMFOLD_TRANSITIONS, sae.DEFAULT_BIOEMU_TRANSITIONS)
    for axis, letter in ((association, "D"), (structural, "E"), (protein, "F")):
        _label(axis, letter, y=1.10 if letter in {"E", "F"} else 1.04)
    sae._format_axis(association)
    sae._format_axis(structural)
    sae._render_association_panel(association, associations)
    association.set_ylabel("PRS association (ρ × 10⁻²)")
    association.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f"{value * 100:.0f}")
    )
    density_axis = sae._render_structural_panel(structural, comparison)
    association_position = association.get_position()
    association.set_position(
        [
            association_position.x0 + 0.012,
            association_position.y0,
            association_position.width - 0.012,
            association_position.height,
        ]
    )
    structural_position = structural.get_position()
    structural.set_position(
        [
            structural_position.x0 + 0.012,
            structural_position.y0,
            structural_position.width - 0.012,
            structural_position.height,
        ]
    )
    # Keep the left-axis titles outside their plotting areas. Positive axis
    # coordinates placed both labels directly on the y-axis spines.
    association.yaxis.set_label_coords(-0.045, 0.5)
    structural.yaxis.set_label_coords(-0.11, 0.5)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0}
    association.yaxis.label.set_bbox(label_box)
    structural.yaxis.label.set_bbox(label_box)
    density_axis.set_position(structural.get_position())
    # Keep E's secondary (top/right) axis attached to its plotting area. The
    # default padding is tuned for a standalone panel and looked detached in
    # this compact multi-panel layout.
    density_axis.tick_params(axis="x", pad=2)
    density_axis.tick_params(axis="y", pad=1)
    density_axis.xaxis.labelpad = 2
    density_axis.yaxis.labelpad = 2
    density_axis.xaxis.set_label_coords(0.5, 1.15)
    # Use one square, symmetric direct-contact coordinate system. This makes
    # its zero exactly coincide with the SASA zero-cross and keeps both direct
    # contact scales centered and extended to their outer ticks.
    direct_limit = int(np.ceil(max(*np.abs(density_axis.get_xlim()), *np.abs(density_axis.get_ylim()))))
    direct_limit = max(1, direct_limit)
    direct_ticks = np.arange(-direct_limit, direct_limit + 1)
    density_axis.set(xlim=(-direct_limit, direct_limit), ylim=(-direct_limit, direct_limit))
    density_axis.set_xticks(direct_ticks)
    density_axis.set_yticks(direct_ticks)
    # Separate the two y-axis titles cleanly: the primary title remains close
    # to E, while the secondary title sits beyond its right-hand tick labels.
    density_axis.yaxis.set_label_coords(1.10, 0.5)
    # The two overlaid scales should read as one comparison plot.  Keep the
    # primary SASA zero-cross/identity guide only, rather than duplicating
    # a second set of guides for the direct-contact scale.
    for guide in density_axis.lines:
        guide.set_visible(False)

    image_path = sae.DEFAULT_PANEL_B_IMAGE
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    # This is the original high-resolution ChimeraX export, placed directly
    # into its final-size axes (not an image of a PDF page).
    protein_image = plt.imread(image_path)
    # The saved paired view has a small text/header band intended for its
    # standalone PDF.  Remove it here so the actual ChimeraX rendering fills
    # panel F rather than being shrunk inside unnecessary white space.
    height, width = protein_image.shape[:2]
    protein_image = protein_image[
        round(0.16 * height) : round(0.98 * height),
        round(0.02 * width) : round(0.98 * width),
    ]
    # Treat the two saved ChimeraX state views as separate panels, trim their
    # unused outside/centre margins, then join them with no artificial gap.
    # This moves the proteins closer together without anisotropically scaling
    # either structure.
    trimmed_width = protein_image.shape[1]
    state_a = protein_image[:, : trimmed_width // 2]
    state_b = protein_image[:, trimmed_width // 2 :]

    def trim_horizontal_whitespace(image: np.ndarray, margin: int = 36) -> np.ndarray:
        """Keep the complete ChimeraX structure, trimming only empty x-space."""
        rgb = image[..., :3]
        content = np.min(rgb, axis=-1) < 0.97
        columns = np.flatnonzero(content.any(axis=0))
        if not len(columns):
            return image
        start = max(0, int(columns[0]) - margin)
        stop = min(image.shape[1], int(columns[-1]) + margin + 1)
        return image[:, start:stop]

    state_a = trim_horizontal_whitespace(state_a)
    state_b = trim_horizontal_whitespace(state_b)
    protein_image = np.concatenate((state_a, state_b), axis=1)
    protein.imshow(protein_image, aspect="auto", interpolation="none")
    protein.set_axis_off()
    return density_axis


def main() -> None:
    # A/D share the top row exactly. B/C/E/F share the bottom-row height;
    # E is widened to two columns and F spans three columns.
    figure = plt.figure(figsize=(12.8, 7.1), facecolor="white")
    grid = figure.add_gridspec(
        2,
        9,
        width_ratios=(0.68, 0.68, 1.12, 1.12, 1.0, 1.0, 1.0, 1.0, 1.0),
        height_ratios=(1, 1),
        hspace=0.38,
        wspace=0.54,
    )
    roc = figure.add_subplot(grid[0, 0:4])
    association = figure.add_subplot(grid[0, 4:9])
    residual = figure.add_subplot(grid[1, 0:2])
    plddt = figure.add_subplot(grid[1, 2:4])
    structural = figure.add_subplot(grid[1, 4:6])
    protein = figure.add_subplot(grid[1, 6:9])
    _draw_frozen(roc, residual, plddt)
    density_axis = _draw_sae(association, structural, protein)
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.95)
    # add_axes() overlays do not participate in subplots_adjust(). Reapply
    # the final structural axes bounds so both coordinate systems have the
    # same physical centre and the right ticks are truly on E's edge.
    density_axis.set_position(structural.get_position())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="pdf")
    figure.savefig(OUTPUT.with_suffix(".svg"), format="svg")
    plt.close(figure)
    print(json.dumps({"output": str(OUTPUT), "layout": "A|D over B|C|E|F"}, indent=2))


if __name__ == "__main__":
    main()
