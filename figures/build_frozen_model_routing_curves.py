"""Render held-out ROC and operational routing curves for frozen best models.

The routing curve ranks proteins by a frozen model's dynamic probability.
At each threshold, its x-coordinate is the fraction of all held-out proteins
routed to dynamic-aware processing, and its y-coordinate is recall among the
held-out dynamic proteins.  Unlike a conventional ROC curve, the x-axis
therefore represents the operational workload sent to the expensive pathway.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "ml/results/homology35_rerun"
DEFAULT_CONFOUNDER_RESULTS = ROOT / "ml/results/homology35_confounder_rerun"
DEFAULT_CATALOG = ROOT / "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
DEFAULT_OUTPUT = ROOT / "neurips-workshop/figures"
RESIDUAL_CATEGORY_ORDER = (
    "Full covariate residualized",
    "pLDDT residualized",
    "Full embedding",
)
RESIDUAL_CATEGORY_ALIASES = {
    "full covariate residualized": "Full covariate residualized",
    "all covariate residualized": "Full covariate residualized",
    "plddt residualized": "pLDDT residualized",
    "full embedding": "Full embedding",
    "raw embedding": "Full embedding",
}


@dataclass(frozen=True)
class ModelPredictions:
    """One verified frozen-model prediction vector."""

    label: str
    color: str
    path: Path
    predictions: pd.DataFrame
    validation_auroc: float | None


def selected_validation_auroc(model_directory: Path) -> float:
    """Return the validation AUROC of a directory's selected candidate."""
    selection_path = model_directory / "validation_selection.json"
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_candidate"]
    matches = [
        float(trial["validation_metric"])
        for trial in selection["trials"]
        if trial.get("candidate") == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"could not resolve selected validation AUROC: {selection_path}")
    return matches[0]


def load_predictions(label: str, color: str, model_directory: Path) -> ModelPredictions:
    """Load one test-prediction table and enforce its binary-probability contract."""
    path = model_directory / "test_predictions.parquet"
    predictions = pd.read_parquet(path)[["protein_id", "dataset_label", "probability"]].copy()
    if predictions.empty or predictions.protein_id.duplicated().any():
        raise ValueError(f"invalid held-out predictions: {path}")
    if not set(predictions.dataset_label.unique()).issubset({0, 1}):
        raise ValueError(f"expected binary labels: {path}")
    if (
        not np.isfinite(predictions.probability).all()
        or not predictions.probability.between(0, 1).all()
    ):
        raise ValueError(f"expected finite probabilities in [0, 1]: {path}")
    return ModelPredictions(
        label=label,
        color=color,
        path=path,
        predictions=predictions.sort_values("protein_id").reset_index(drop=True),
        validation_auroc=selected_validation_auroc(model_directory),
    )


def choose_best_cnn(results_root: Path) -> Path:
    """Choose the full-matrix CNN variant by validation AUROC, never test performance."""
    root = results_root / "full_matrix_cnn"
    candidates = [
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and (directory / "test_predictions.parquet").is_file()
        and (directory / "validation_selection.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no frozen CNN predictions found under {root}")
    return max(candidates, key=selected_validation_auroc)


def load_repeated_predictions(
    label: str, color: str, results_root: Path, model_name: str
) -> ModelPredictions:
    """Pool repeated held-out predictions for a split-stability ROC estimate."""
    directories = sorted(
        directory
        for directory in results_root.glob("seed_*/" + model_name)
        if (directory / "test_predictions.parquet").is_file()
        and (directory / "validation_selection.json").is_file()
    )
    if not directories:
        raise FileNotFoundError(f"no repeated predictions found for {model_name} under {results_root}")
    tables = [load_predictions(label, color, directory) for directory in directories]
    predictions = pd.concat([item.predictions for item in tables], ignore_index=True)
    return ModelPredictions(
        label=label,
        color=color,
        path=results_root / model_name,
        predictions=predictions,
        validation_auroc=float(np.mean([item.validation_auroc for item in tables if item.validation_auroc is not None])),
    )


def load_residual_panel(results_root: Path) -> pd.DataFrame:
    """Load the two-split linear/tree residual follow-up AUROCs."""
    path = results_root / "followup/residual_metrics.csv"
    frame = pd.read_csv(path)
    required = {"seed", "family", "feature_view", "test_auroc"}
    if missing := required - set(frame):
        raise ValueError(f"residual metrics missing columns: {sorted(missing)}")
    view_labels = {
        "residual_full_covariates": RESIDUAL_CATEGORY_ORDER[0],
        "residual_plddt": RESIDUAL_CATEGORY_ORDER[1],
        "raw_esmfold": RESIDUAL_CATEGORY_ORDER[2],
    }
    frame = frame.loc[frame.feature_view.isin(view_labels)].copy()
    frame["category"] = frame.feature_view.map(view_labels)
    frame["category"] = frame["category"].map(
        lambda value: RESIDUAL_CATEGORY_ALIASES.get(str(value).strip().lower(), value)
    )
    if frame.empty or set(frame.family) != {"linear", "tree"}:
        raise ValueError("expected linear and tree residual follow-up metrics")
    return frame


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    """Load pLDDT and labels for the CNN test predictions."""
    catalog = pd.read_parquet(catalog_path)
    required = {"protein_id", "dataset_label", "alphafold_mean_plddt"}
    if missing := required - set(catalog):
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    return catalog[["protein_id", "dataset_label", "alphafold_mean_plddt"]].drop_duplicates(
        "protein_id"
    )


def routing_curve(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return workload fraction and dynamic recall at each distinct score threshold."""
    order = np.argsort(-probabilities, kind="stable")
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]
    group_end = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    cumulative_routed = np.cumsum(np.ones(len(sorted_labels), dtype=int))[group_end]
    cumulative_dynamic = np.cumsum(sorted_labels)[group_end]
    total_dynamic = int(labels.sum())
    if total_dynamic == 0:
        raise ValueError("routing curve requires at least one dynamic protein")
    return (
        np.r_[0.0, cumulative_routed / len(labels)],
        np.r_[0.0, cumulative_dynamic / total_dynamic],
    )


def verify_shared_test_set(models: list[ModelPredictions]) -> np.ndarray:
    """Require identical held-out proteins and labels for a fair overlay."""
    reference = models[0].predictions
    identifiers = reference.protein_id.to_numpy()
    labels = reference.dataset_label.to_numpy(dtype=int)
    for model in models[1:]:
        actual = model.predictions
        if not np.array_equal(actual.protein_id.to_numpy(), identifiers):
            raise ValueError(f"test protein IDs differ for {model.label}")
        if not np.array_equal(actual.dataset_label.to_numpy(dtype=int), labels):
            raise ValueError(f"test labels differ for {model.label}")
    return labels


def render(
    models: list[ModelPredictions],
    covariate_models: list[ModelPredictions],
    residual_panel: pd.DataFrame,
    cnn_model: ModelPredictions,
    catalog: pd.DataFrame,
    output_directory: Path,
) -> dict[str, object]:
    """Create the requested ROC, residual-AUROC, and pLDDT scatter figure."""
    labels = verify_shared_test_set(models)
    output_directory.mkdir(parents=True, exist_ok=True)
    # This is deliberately half-width so it can sit beside another figure in
    # the workshop manuscript. Panel A spans the top; B and C share the row
    # below it.
    figure = plt.figure(figsize=(6.2, 5.9))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(0.78, 1.22),
        height_ratios=(1.05, 0.95),
        hspace=0.42,
        wspace=0.34,
    )
    roc_axis = figure.add_subplot(grid[0, :])
    residual_axis = figure.add_subplot(grid[1, 0])
    plddt_axis = figure.add_subplot(grid[1, 1])
    axes = ((roc_axis, "A"), (residual_axis, "B"), (plddt_axis, "C"))
    for axis, panel in axes:
        # Keep labels above the axes rather than in the data region. This is
        # especially important for the compact B/C panels.
        axis.text(
            0.0,
            1.04,
            panel,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
        axis.grid(alpha=0.22, linewidth=0.7)

    roc_axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.1, label="Random")
    metadata_models: list[dict[str, object]] = []
    for model in [*models, *covariate_models]:
        is_frozen_model = any(model is frozen for frozen in models)
        model_labels = model.predictions.dataset_label.to_numpy(dtype=int)
        probabilities = model.predictions.probability.to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(model_labels, probabilities)
        auc = float(roc_auc_score(model_labels, probabilities))
        roc_axis.plot(
            fpr,
            tpr,
            color=model.color,
            linewidth=2.2 if is_frozen_model else 1.8,
            linestyle="-" if is_frozen_model else "--",
            label=f"{model.label} (AUROC {auc:.3f})",
        )
        metadata_models.append(
            {
                "label": model.label,
                "test_prediction_path": str(model.path.relative_to(ROOT)),
                "validation_auroc": model.validation_auroc,
                "test_auroc": auc,
                "prediction_rows": int(len(model_labels)),
            }
        )
    roc_axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    roc_axis.legend(loc="lower right", frameon=False, fontsize=5.0, ncol=2)

    category_order = list(RESIDUAL_CATEGORY_ORDER)
    x_positions = np.arange(len(category_order))
    family_colors = {"linear": "#1b9e77", "tree": "#d95f02"}
    panel_b_metadata: list[dict[str, object]] = []
    for (family, seed), group in residual_panel.groupby(["family", "seed"], sort=True):
        values = group.set_index("category").reindex(category_order).test_auroc
        if values.isna().any():
            raise ValueError(f"incomplete residual AUROC categories for {family} seed {seed}")
        residual_axis.plot(
            x_positions,
            values.to_numpy(dtype=float),
            marker="o",
            linewidth=1.4,
            linestyle=":",
            color=family_colors[family],
            alpha=0.8,
            label=f"{family.title()} split {seed}",
        )
        panel_b_metadata.append(
            {
                "family": family,
                "seed": int(seed),
                "categories": dict(zip(category_order, values.tolist(), strict=True)),
            }
        )
    residual_axis.set(
        ylabel="Test AUROC",
        xticks=x_positions,
        xticklabels=[
            "All covariate\nresidualized",
            "pLDDT\nresidualized",
            "Raw\nembedding",
        ],
        ylim=(0.5, 0.9),
    )
    residual_axis.legend(loc="lower left", frameon=False, fontsize=4.8, ncol=2)

    scatter = cnn_model.predictions.merge(
        catalog,
        on="protein_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", "_catalog"),
    )
    scatter = scatter.loc[scatter.alphafold_mean_plddt.ge(70)].copy()
    if scatter.empty:
        raise ValueError("CNN pLDDT scatter has no medium/high pLDDT test proteins")
    colors = scatter.dataset_label_prediction.map({0: "#4c78a8", 1: "#e45756"})
    highlight_style = dict(
        boxstyle="round,pad=0.3",
        facecolor="#BFBFD6",
        edgecolor="#676775",
        alpha=0.5
    )
    plddt_axis.scatter(
        scatter.probability,
        scatter.alphafold_mean_plddt,
        c=colors,
        alpha=0.72,
        s=11,
        linewidth=0.25,
        edgecolors="white",
    )
    plddt_axis.axhline(70, color="#777777", linewidth=0.9, linestyle="--")
    plddt_axis.axhline(90, color="#777777", linewidth=0.9, linestyle=":")
    plddt_axis.set(
        xlabel="Predicted multistate probability",
        ylabel="Mean pLDDT",
        xlim=(0, 1),
    )
    plddt_axis.text(0.02, 0.075, "medium", transform=plddt_axis.transAxes, va="top", fontsize=8, bbox=highlight_style,weight="bold")
    plddt_axis.text(0.02, 0.705, "high", transform=plddt_axis.transAxes, va="top", fontsize=8, bbox=highlight_style,weight="bold")
    plddt_axis.scatter([], [], color="#4c78a8", label="Static")
    plddt_axis.scatter([], [], color="#e45756", label="Dynamic")

    plddt_axis.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.3,
        facecolor="#BFBFD6",
        prop={"size": 4.0, "weight": "bold"},
        markerscale=0.45,
        borderpad=0.25,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.2,
    )

    # Explicit margins are more predictable than tight_layout for this mixed
    # spanning/side-by-side GridSpec, while keeping the row gap compact.
    for axis in (roc_axis, residual_axis, plddt_axis):
        axis.tick_params(axis="both", labelsize=7)
        axis.xaxis.label.set_size(7.5)
        axis.yaxis.label.set_size(7.5)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.94, hspace=0.22, wspace=0.34)
    stem = output_directory / "small_frozen_model_roc_residual_plddt_panels"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "test_set_size": int(len(labels)),
        "dynamic_count": int(labels.sum()),
        "static_count": int((labels == 0).sum()),
        "models": metadata_models,
        "residual_panel": panel_b_metadata,
        "cnn_scatter": {
            "model": cnn_model.label,
            "validation_auroc": cnn_model.validation_auroc,
            "test_prediction_path": str(cnn_model.path.relative_to(ROOT)),
            "pLDDT_filter": "mean pLDDT >= 70 (medium and high strata)",
            "points": int(len(scatter)),
        },
        "outputs": [str(stem.with_suffix(suffix).relative_to(ROOT)) for suffix in (".pdf", ".svg")],
    }
    (output_directory / "small_frozen_model_roc_residual_plddt_panels.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--confounder-results", type=Path, default=DEFAULT_CONFOUNDER_RESULTS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    confounder_results = args.confounder_results.resolve()
    cnn_directory = choose_best_cnn(results_root)
    models = [
        load_predictions(
            "Logistic regression",
            "#1b9e77",
            results_root / "pooled_frozen_models/esmfold_single_linear",
        ),
        load_predictions(
            "Histogram gradient tree",
            "#d95f02",
            results_root / "pooled_frozen_models/esmfold_single_tree",
        ),
        load_predictions("Full-matrix CNN", "#7570b3", cnn_directory),
    ]
    covariate_models = [
        load_repeated_predictions(
            "Covariate-only linear",
            "#66a61e",
            confounder_results / "pooled_confounder",
            "linear_covariates",
        ),
        load_repeated_predictions(
            "Covariate-only tree",
            "#e6ab02",
            confounder_results / "pooled_confounder",
            "tree_covariates",
        ),
    ]
    residual_panel = load_residual_panel(confounder_results)
    print(
        json.dumps(
            render(
                models,
                covariate_models,
                residual_panel,
                models[-1],
                load_catalog(args.catalog.resolve()),
                args.output_directory.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
