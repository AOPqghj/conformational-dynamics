"""Seed-42 CNN residue masking and grouped SAE-feature ablations.

The default ``cnn-residues`` mode preserves the original legacy CNN analysis.
``latent-feature-groups`` evaluates simultaneous ablation of ESMFold SAE features
in the current frozen pooled linear and tree router models.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/protein-state-router-matplotlib")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpretability.analyze_sae_transition_residue_associations import (
    load_frozen_sae,
    sha256_file,
)
from interpretability.contracts import load_residue_matrix, pool_residue_matrix
from interpretability.model import FrozenPooledModel, FrozenResidueCNNModel

LEGACY_ML = Path("ml/results/archive/legacy_seed42")
CATALOG = LEGACY_ML / "frozen_models/seed_42_catalog.parquet"
CNN_ROOT = LEGACY_ML / "frozen_seed42_residue_models"
SAE_ROOT = LEGACY_ML / "frozen_saes/esmfold_matrix_topk64_seed42"
DISP = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/archive/legacy_seed42/seed42_dynamic_transition_residue_ca_displacements.csv"
)
PRS = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/archive/legacy_seed42/seed42_dynamic_transition_prs_scores.csv"
)
OUT = Path("interpretability/results/06_cnn_residue_attribution_ablation")

# This immutable copy is checksum-bound to the frozen SAE and pooled routers.
GROUP_CATALOG = Path("ml/results/homology35_rerun/pooled_frozen_models/seed_42_catalog.parquet")
GROUP_SAE = Path("ml/results/homology35_rerun/frozen_saes/esmfold_matrix_topk64_seed42")
GROUP_MODELS = Path("ml/results/homology35_rerun/pooled_frozen_models")
GROUP_ASSOCIATIONS = Path("interpretability/results/homology35_rerun/sae_transition_associations")
GROUP_OUT = Path("interpretability/results/homology35_rerun/esmfold_group_latent_ablation")


@dataclass(frozen=True)
class GroupConfig:
    seed: int = 42
    device: str = "auto"
    pool_size: int = 250
    groups_per_size: int = 5
    group_sizes: tuple[int, ...] = (10, 20, 50)
    bootstrap_draws: int = 2_000

    @property
    def config_hash(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def status(event: str, **kw: object) -> None:
    print(json.dumps({"event": event, **kw}, sort_keys=True), flush=True)


def atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def baseline(catalog: pd.DataFrame) -> np.ndarray:
    total = np.zeros(1024, dtype=np.float64)
    count = 0
    for i, row in enumerate(catalog.loc[catalog.split.eq("train")].itertuples(index=False), 1):
        x = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
        )
        total += x.sum(0)
        count += len(x)
        if i == 1 or i % 250 == 0:
            status("baseline", completed=i, total=6018)
    return (total / count).astype(np.float32)


def attribution(
    model: FrozenResidueCNNModel, x: np.ndarray, base: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = torch.tensor(x, device=model.device, requires_grad=True)
    value = raw.unsqueeze(0)
    mask = torch.ones((1, len(x)), dtype=torch.bool, device=model.device)
    logit = model.model(value, mask).squeeze()
    logit.backward()
    gxi = (raw.grad * (raw.detach() - torch.tensor(base, device=model.device))).sum(1)
    delta = torch.tensor(x - base, device=model.device)
    integral = torch.zeros_like(delta)
    for alpha in torch.linspace(1 / steps, 1, steps, device=model.device):
        point = (
            (torch.tensor(base, device=model.device) + alpha * delta)
            .unsqueeze(0)
            .detach()
            .requires_grad_(True)
        )
        model.model.zero_grad(set_to_none=True)
        model.model(point, mask).squeeze().backward()
        integral += point.grad.squeeze(0)
    ig = (delta * integral / steps).sum(1)
    return gxi.detach().cpu().numpy(), ig.detach().cpu().numpy()


def top(values: np.ndarray, fraction: float) -> np.ndarray:
    return np.argsort(-np.abs(values), kind="stable")[
        : max(1, int(np.ceil(len(values) * fraction)))
    ]


def run(device: str, steps: int, random_controls: int, seed: int, output: Path) -> None:
    catalog = pd.read_parquet(CATALOG)
    test = catalog.loc[catalog.split.eq("test")].copy().reset_index(drop=True)
    if len(test) != 1289:
        raise ValueError("expected the frozen Seed-42 1,289-protein test partition")
    cnn = FrozenResidueCNNModel(CNN_ROOT, "residue_cnn_expanded", device=device)
    base = baseline(catalog)
    np.save(output / "train_residue_mean.npy", base)
    sae, center, _ = load_frozen_sae(SAE_ROOT, torch.device(device))
    rows = []
    matrices = {}
    for i, row in enumerate(test.itertuples(index=False), 1):
        x = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
        )
        matrices[row.protein_id] = x
        gxi, ig = attribution(cnn, x, base, steps)
        with torch.no_grad():
            z = sae.encode(torch.tensor(x - center, device=device)).detach().cpu().numpy()[:, 2722]
        rows.extend(
            {
                "protein_id": row.protein_id,
                "sequence_sha256": row.sequence_sha256,
                "residue_index": j,
                "canonical_residue_number": j + 1,
                "gradient_x_input": float(gxi[j]),
                "integrated_gradients": float(ig[j]),
                "attribution_abs": float(abs(ig[j])),
                "sae_2722_activation": float(z[j]),
            }
            for j in range(len(x))
        )
        if i == 1 or i % 25 == 0 or i == len(test):
            status("attributions", completed=i, total=len(test))
    attrs = pd.DataFrame(rows)
    atomic(attrs, output / "cnn_residue_attributions.csv")
    scores = pd.read_csv(DISP, low_memory=False)[
        [
            "protein_id",
            "sequence_sha256",
            "canonical_residue_number",
            "ca_displacement_after_global_kabsch_angstrom",
        ]
    ].merge(
        pd.read_csv(PRS, low_memory=False)[
            ["protein_id", "sequence_sha256", "canonical_residue_number", "prs_max_overlap"]
        ],
        on=["protein_id", "sequence_sha256", "canonical_residue_number"],
        validate="one_to_one",
    )
    joined = attrs.merge(
        scores, on=["protein_id", "sequence_sha256", "canonical_residue_number"], how="inner"
    )
    associations = []
    for target in ["ca_displacement_after_global_kabsch_angstrom", "prs_max_overlap"]:
        per = []
        for _, g in joined.groupby("protein_id"):
            if len(g) > 2:
                per.append(spearmanr(g.attribution_abs, g[target]).statistic)
        associations.append(
            {"target": target, "proteins": len(per), "balanced_spearman": float(np.nanmean(per))}
        )
    atomic(pd.DataFrame(associations), output / "cnn_transition_associations.csv")
    rng = np.random.default_rng(seed)
    ablations = []
    labels = test.dataset_label.to_numpy()
    methods = ["cnn_ig", "sae_2722"]
    for frac in (0.05, 0.10, 0.20):
        predictions = {m: [] for m in methods}
        predictions["native"] = []
        for i, row in enumerate(test.itertuples(index=False), 1):
            x = matrices[row.protein_id]
            native = cnn.score_matrix(x).probability
            predictions["native"].append(native)
            a = attrs.loc[attrs.protein_id.eq(row.protein_id)]
            sets = {
                "cnn_ig": top(a.integrated_gradients.to_numpy(), frac),
                "sae_2722": top(a.sae_2722_activation.to_numpy(), frac),
            }
            for method, indices in sets.items():
                y = x.copy()
                y[indices] = base
                p = cnn.score_matrix(y).probability
                predictions[method].append(p)
                controls = []
                for _ in range(random_controls):
                    z = x.copy()
                    z[rng.choice(len(x), len(indices), replace=False)] = base
                    controls.append(cnn.score_matrix(z).probability)
                ablations.append(
                    {
                        "protein_id": row.protein_id,
                        "dataset_label": row.dataset_label,
                        "method": method,
                        "fraction": frac,
                        "native_probability": native,
                        "masked_probability": p,
                        "probability_drop": native - p,
                        "random_probability_drop_mean": native - float(np.mean(controls)),
                        "targeted_minus_random_drop": (native - p)
                        - (native - float(np.mean(controls))),
                    }
                )
            if i % 25 == 0:
                status("ablations", completed=i, total=len(test), fraction=frac)
        for method in methods:
            ablations.append(
                {
                    "protein_id": "__metric__",
                    "dataset_label": -1,
                    "method": method,
                    "fraction": frac,
                    "native_probability": roc_auc_score(labels, predictions["native"]),
                    "masked_probability": roc_auc_score(labels, predictions[method]),
                    "probability_drop": roc_auc_score(labels, predictions["native"])
                    - roc_auc_score(labels, predictions[method]),
                    "random_probability_drop_mean": average_precision_score(
                        labels, predictions["native"]
                    )
                    - average_precision_score(labels, predictions[method]),
                    "targeted_minus_random_drop": np.nan,
                }
            )
    atomic(pd.DataFrame(ablations), output / "residue_ablation_results.csv")
    # Explicit placeholders make unsupported structural and RMSD/PRS ablations visible rather than silently absent.
    atomic(
        pd.DataFrame(
            [
                {
                    "status": "pending_structure_context",
                    "reason": "requires the CNN-specific temporary-structure pass",
                }
            ]
        ),
        output / "cnn_structural_enrichment.csv",
    )
    atomic(
        pd.DataFrame(
            [
                {
                    "status": "pending_rmsd_prs_ablation",
                    "reason": "dynamic-only 595-protein paired analysis",
                }
            ]
        ),
        output / "cross_method_overlap.csv",
    )
    (output / "index.html").write_text(
        "<h1>Seed-42 CNN residue attribution</h1><p>See CSV artifacts and progress output.</p>"
    )


def _group_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; use --device auto or --device cpu")
    return torch.device(name)


def _require_group_catalog(catalog: pd.DataFrame) -> None:
    required = {
        "protein_id",
        "sequence",
        "sequence_sha256",
        "sequence_length",
        "dataset_label",
        "split",
        "embedding_path",
    }
    if required - set(catalog) or catalog.protein_id.duplicated().any():
        raise ValueError("group ablation requires a complete unique frozen catalog")
    if set(catalog.split) != {"train", "val", "test"}:
        raise ValueError("group ablation requires train, val, and test partitions")


def _ranking_tables(root: Path, pool_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    displacement = pd.read_csv(root / "ranked_displacement_features.csv")
    prs = pd.read_csv(root / "ranked_prs_features.csv")
    required = {"feature_id", "activation_frequency"}
    if required - set(displacement) or required - set(prs):
        raise ValueError("transition association rankings are missing required columns")
    if displacement.feature_id.duplicated().any() or prs.feature_id.duplicated().any():
        raise ValueError("transition association rankings must have unique feature IDs")
    if pool_size < 1 or pool_size > len(displacement):
        raise ValueError("pool size must be between 1 and the SAE latent dimension")
    d = displacement.reset_index(names="displacement_rank")
    p = prs[["feature_id"]].reset_index(names="prs_rank")
    combined = d.merge(p, on="feature_id", validate="one_to_one")
    combined["combined_rank"] = (combined.displacement_rank + combined.prs_rank) / 2
    combined = combined.sort_values(["combined_rank", "feature_id"]).head(pool_size).copy()
    displacement_pool = d.sort_values(["displacement_rank", "feature_id"]).head(pool_size).copy()
    frequency = displacement[["feature_id", "activation_frequency"]].copy()
    return combined, displacement_pool, frequency


def build_feature_groups(
    combined: pd.DataFrame,
    displacement: pd.DataFrame,
    frequency: pd.DataFrame,
    config: GroupConfig,
) -> pd.DataFrame:
    """Create deterministic target and frequency-decile-matched control groups."""
    rng = np.random.default_rng(config.seed)
    all_features = frequency.feature_id.to_numpy(dtype=int)
    bins = pd.qcut(frequency.activation_frequency, q=10, duplicates="drop", labels=False)
    bin_for = dict(zip(all_features, bins.astype(int), strict=True))
    pools = {
        "combined": combined.feature_id.to_numpy(dtype=int),
        "displacement": displacement.feature_id.to_numpy(dtype=int),
    }
    rows: list[dict[str, object]] = []
    for pool_name, pool in pools.items():
        if len(pool) < max(config.group_sizes):
            raise ValueError(f"{pool_name} pool is too small for requested group size")
        eligible_controls = np.setdiff1d(all_features, pool, assume_unique=False)
        for size in config.group_sizes:
            for draw in range(config.groups_per_size):
                target = rng.choice(pool, size=size, replace=False).astype(int)
                controls: list[int] = []
                for feature in target:
                    choices = np.setdiff1d(
                        eligible_controls[
                            np.asarray(
                                [
                                    bin_for[value] == bin_for[int(feature)]
                                    for value in eligible_controls
                                ]
                            )
                        ],
                        np.asarray(controls, dtype=int),
                        assume_unique=False,
                    )
                    if not len(choices):
                        choices = np.setdiff1d(eligible_controls, np.asarray(controls, dtype=int))
                    controls.append(int(rng.choice(choices)))
                for role, members in (
                    ("transition", target),
                    ("random_control", np.asarray(controls)),
                ):
                    group_id = f"{pool_name}_{role}_n{size}_draw{draw + 1}"
                    for position, feature in enumerate(members, 1):
                        rows.append(
                            {
                                "group_id": group_id,
                                "pool": pool_name,
                                "role": role,
                                "group_size": size,
                                "draw": draw + 1,
                                "feature_position": position,
                                "feature_id": int(feature),
                                "activation_frequency": float(
                                    frequency.loc[
                                        frequency.feature_id.eq(feature), "activation_frequency"
                                    ].iloc[0]
                                ),
                                "activation_frequency_decile": int(bin_for[int(feature)]),
                            }
                        )
    result = pd.DataFrame(rows)
    if result.groupby("group_id").feature_id.nunique().ne(result.groupby("group_id").size()).any():
        raise AssertionError("feature group contains duplicates")
    return result


def group_ablated_reconstructions(
    reconstruction: torch.Tensor,
    latents: torch.Tensor,
    decoder_weight: torch.Tensor,
    groups: list[np.ndarray],
) -> torch.Tensor:
    """Remove all selected decoder contributions at once for each feature group."""
    values = []
    for group in groups:
        indices = torch.as_tensor(group, dtype=torch.long, device=latents.device)
        contribution = latents[:, indices] @ decoder_weight[:, indices].T
        values.append(reconstruction - contribution)
    return torch.stack(values)


def _pooled_torch(values: torch.Tensor) -> np.ndarray:
    pooled = torch.cat(
        (values.mean(dim=1), values.std(dim=1, unbiased=False), values.max(dim=1).values), dim=1
    )
    return pooled.detach().cpu().numpy().astype(np.float32)


def _scores(model: FrozenPooledModel, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.model.predict_proba(features)[:, 1].astype(np.float32)
    predictions = model.model.predict(features).astype(np.int8)
    return probabilities, predictions


def _stratified_bootstrap_indices(labels: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    return np.stack(
        [
            np.concatenate(
                (
                    rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True),
                )
            )
            for _ in range(draws)
        ]
    )


def _bootstrap_accuracy_drop(
    labels: np.ndarray, baseline: np.ndarray, ablated: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    baseline_correct = baseline == labels
    ablated_correct = ablated == labels
    drops = baseline_correct[indices].mean(axis=1) - ablated_correct[indices].mean(axis=1)
    return float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))


def _group_metrics(
    labels: np.ndarray,
    groups: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    config: GroupConfig,
) -> pd.DataFrame:
    group_index = (
        groups[["group_id", "pool", "role", "group_size", "draw"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    bootstrap = _stratified_bootstrap_indices(labels, config.bootstrap_draws, config.seed)
    rows: list[dict[str, object]] = []
    for family in ("linear", "tree"):
        base_prediction = predictions[f"{family}_reconstruction_prediction"]
        base_accuracy = accuracy_score(labels, base_prediction)
        for index, group in group_index.iterrows():
            probability = predictions[f"{family}_group_probability"][index]
            prediction = predictions[f"{family}_group_prediction"][index]
            low, high = _bootstrap_accuracy_drop(
                labels,
                base_prediction,
                prediction,
                bootstrap,
            )
            rows.append(
                {
                    **group.to_dict(),
                    "model": family,
                    "accuracy": accuracy_score(labels, prediction),
                    "balanced_accuracy": balanced_accuracy_score(labels, prediction),
                    "auroc": roc_auc_score(labels, probability),
                    "auprc": average_precision_score(labels, probability),
                    "reconstruction_accuracy": base_accuracy,
                    "accuracy_drop_from_reconstruction": base_accuracy
                    - accuracy_score(labels, prediction),
                    "accuracy_drop_ci_low": low,
                    "accuracy_drop_ci_high": high,
                }
            )
    result = pd.DataFrame(rows)
    for family in ("linear", "tree"):
        for pool in ("combined", "displacement"):
            for size in config.group_sizes:
                controls = result.loc[
                    (result.model == family)
                    & (result.pool == pool)
                    & (result.group_size == size)
                    & (result.role == "random_control"),
                    "accuracy_drop_from_reconstruction",
                ].to_numpy()
                mask = (
                    (result.model == family)
                    & (result.pool == pool)
                    & (result.group_size == size)
                    & (result.role == "transition")
                )
                result.loc[mask, "control_percentile"] = result.loc[
                    mask, "accuracy_drop_from_reconstruction"
                ].map(lambda value, control=controls: float((control <= value).mean()))
                result.loc[mask, "control_p"] = result.loc[
                    mask, "accuracy_drop_from_reconstruction"
                ].map(
                    lambda value, control=controls: float(
                        (1 + np.count_nonzero(control >= value)) / (len(control) + 1)
                    )
                )
    return result


def _write_group_figure(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, family in zip(axes, ("linear", "tree"), strict=True):
        values = metrics.loc[metrics.model.eq(family)]
        for (pool, role), subset in values.groupby(["pool", "role"]):
            x = subset.group_size.to_numpy(dtype=float)
            jitter = -0.16 if role == "transition" else 0.16
            axis.scatter(
                x + jitter,
                subset.accuracy_drop_from_reconstruction,
                alpha=0.8,
                label=f"{pool}: {role}",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=family, xlabel="Features ablated", xticks=[10, 20, 50])
    axes[0].set_ylabel("Accuracy drop from full SAE reconstruction")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(figures / "group_ablation_accuracy.pdf")
    plt.close(fig)


def run_latent_feature_groups(output: Path, config: GroupConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    if progress.is_file():
        prior = json.loads(progress.read_text())
        if prior.get("status") == "complete" and prior.get("config_hash") == config.config_hash:
            status("already_complete", output=str(output))
            return
    device = _group_device(config.device)
    catalog = pd.read_parquet(GROUP_CATALOG)
    _require_group_catalog(catalog)
    sae, center, manifest = load_frozen_sae(GROUP_SAE, device)
    if manifest.get("catalog_sha256") != sha256_file(GROUP_CATALOG):
        raise ValueError(
            "frozen ESMFold SAE catalog checksum does not match group-ablation catalog"
        )
    combined, displacement, frequency = _ranking_tables(GROUP_ASSOCIATIONS, config.pool_size)
    groups = build_feature_groups(combined, displacement, frequency, config)
    groups.to_csv(output / "feature_groups.csv", index=False)
    group_index = (
        groups[["group_id", "pool", "role", "group_size", "draw"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    members = [
        groups.loc[groups.group_id.eq(group_id), "feature_id"].to_numpy(dtype=int)
        for group_id in group_index.group_id
    ]
    test = catalog.loc[catalog.split.eq("test")].reset_index(drop=True)
    labels = test.dataset_label.to_numpy(dtype=np.int8)
    models = {
        name: FrozenPooledModel(GROUP_MODELS, f"esmfold_single_{name}")
        for name in ("linear", "tree")
    }
    n, g = len(test), len(members)
    work = output / ".work"
    work.mkdir(exist_ok=True)
    feature_progress = work / "feature_progress.json"

    def open_features(path: Path, shape: tuple[int, ...]) -> np.memmap:
        if path.is_file():
            values = np.lib.format.open_memmap(path, mode="r+")
            if values.shape != shape:
                raise ValueError(f"cached feature shape differs for {path.name}")
            return values
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)

    group_features = open_features(work / "group_pooled_features.npy", (g, n, len(center) * 3))
    native_features = open_features(work / "native_pooled_features.npy", (n, len(center) * 3))
    reconstruction_features = open_features(
        work / "reconstruction_pooled_features.npy", (n, len(center) * 3)
    )
    predictions = {
        **{f"{name}_native_probability": np.empty(n, dtype=np.float32) for name in models},
        **{f"{name}_native_prediction": np.empty(n, dtype=np.int8) for name in models},
        **{f"{name}_reconstruction_probability": np.empty(n, dtype=np.float32) for name in models},
        **{f"{name}_reconstruction_prediction": np.empty(n, dtype=np.int8) for name in models},
        **{f"{name}_group_probability": np.empty((g, n), dtype=np.float32) for name in models},
        **{f"{name}_group_prediction": np.empty((g, n), dtype=np.int8) for name in models},
    }
    center_tensor = torch.from_numpy(center).to(device)
    decoder = sae.decoder.weight.detach()
    prior = json.loads(feature_progress.read_text()) if feature_progress.is_file() else {}
    if prior and prior.get("config_hash") != config.config_hash:
        raise ValueError("feature cache was created with a different group-ablation configuration")
    completed = int(prior.get("completed", 0))
    for index, row in enumerate(test.itertuples(index=False), 1):
        if index <= completed:
            continue
        matrix = load_residue_matrix(
            Path(row.embedding_path),
            protein_id=row.protein_id,
            sequence=row.sequence,
            sequence_sha256=row.sequence_sha256,
            sequence_length=int(row.sequence_length),
            expected_width=len(center),
        )
        centered = torch.from_numpy(matrix.astype(np.float32, copy=False) - center).to(device)
        with torch.inference_mode():
            reconstruction, latents = sae(centered)
            reconstruction = reconstruction + center_tensor
            ablated_features = _pooled_torch(
                group_ablated_reconstructions(reconstruction, latents, decoder, members)
            )
        group_features[:, index - 1] = ablated_features
        native_features[index - 1] = pool_residue_matrix(matrix)
        reconstruction_features[index - 1] = _pooled_torch(reconstruction.unsqueeze(0))[0]
        if index == 1 or index % 25 == 0 or index == n:
            group_features.flush()
            native_features.flush()
            reconstruction_features.flush()
            feature_progress.write_text(
                json.dumps({"config_hash": config.config_hash, "completed": index, "total": n})
                + "\n"
            )
            status("group_ablations", completed=index, total=n, groups=g, device=str(device))
    group_features.flush()
    native_features.flush()
    reconstruction_features.flush()
    feature_progress.write_text(
        json.dumps(
            {"config_hash": config.config_hash, "completed": n, "total": n, "status": "complete"}
        )
        + "\n"
    )
    for name, model in models.items():
        native_probability, native_prediction = _scores(model, native_features)
        reconstruction_probability, reconstruction_prediction = _scores(
            model, reconstruction_features
        )
        predictions[f"{name}_native_probability"] = native_probability
        predictions[f"{name}_native_prediction"] = native_prediction
        predictions[f"{name}_reconstruction_probability"] = reconstruction_probability
        predictions[f"{name}_reconstruction_prediction"] = reconstruction_prediction
        for group_start in range(0, g, 5):
            group_stop = min(group_start + 5, g)
            features = group_features[group_start:group_stop].reshape(-1, len(center) * 3)
            probability, prediction = _scores(model, features)
            predictions[f"{name}_group_probability"][group_start:group_stop] = probability.reshape(
                group_stop - group_start, n
            )
            predictions[f"{name}_group_prediction"][group_start:group_stop] = prediction.reshape(
                group_stop - group_start, n
            )
    np.savez_compressed(
        output / "group_ablation_predictions.npz",
        labels=labels,
        group_ids=group_index.group_id.to_numpy(),
        **predictions,
    )
    metrics = _group_metrics(labels, groups, predictions, config)
    metrics.to_csv(output / "group_ablation_metrics.csv", index=False)
    baseline = []
    for name in models:
        for condition in ("native", "reconstruction"):
            probability = predictions[f"{name}_{condition}_probability"]
            prediction = predictions[f"{name}_{condition}_prediction"]
            baseline.append(
                {
                    "model": name,
                    "condition": condition,
                    "accuracy": accuracy_score(labels, prediction),
                    "balanced_accuracy": balanced_accuracy_score(labels, prediction),
                    "auroc": roc_auc_score(labels, probability),
                    "auprc": average_precision_score(labels, probability),
                }
            )
    pd.DataFrame(baseline).to_csv(output / "baseline_metrics.csv", index=False)
    _write_group_figure(metrics, output)
    manifest_payload = {
        "config": asdict(config),
        "config_hash": config.config_hash,
        "catalog": str(GROUP_CATALOG),
        "catalog_sha256": sha256_file(GROUP_CATALOG),
        "sae": str(GROUP_SAE),
        "models": str(GROUP_MODELS),
        "association_root": str(GROUP_ASSOCIATIONS),
        "test_proteins": len(test),
        "groups": g,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest_payload, indent=2) + "\n")
    (output / "index.html").write_text(
        '<h1>ESMFold grouped SAE latent ablation</h1><p>Primary metric: accuracy loss from full SAE reconstruction.</p><ul><li><a href="feature_groups.csv">Feature groups</a></li><li><a href="baseline_metrics.csv">Baselines</a></li><li><a href="group_ablation_metrics.csv">Group metrics</a></li><li><a href="figures/group_ablation_accuracy.pdf">Accuracy figure</a></li></ul>'
    )
    progress.write_text(
        json.dumps({"status": "complete", "config_hash": config.config_hash}, indent=2) + "\n"
    )
    status("complete", output=str(output), groups=g, test_proteins=n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("cnn-residues", "latent-feature-groups"), default="cnn-residues"
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--random-controls", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    p.add_argument("--pool-size", type=int, default=250)
    p.add_argument("--groups-per-size", type=int, default=5)
    p.add_argument("--group-sizes", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--bootstrap-draws", type=int, default=2_000)
    a = p.parse_args()
    if a.mode == "cnn-residues":
        output = a.output or OUT
        output.mkdir(parents=True, exist_ok=True)
        run(a.device, a.ig_steps, a.random_controls, a.seed, output)
        return
    if min(a.pool_size, a.groups_per_size, a.bootstrap_draws, *a.group_sizes) < 1:
        p.error("group sizes, pool size, group count, and bootstrap draws must be positive")
    output = a.output or GROUP_OUT
    run_latent_feature_groups(
        output,
        GroupConfig(
            seed=a.seed,
            device=a.device,
            pool_size=a.pool_size,
            groups_per_size=a.groups_per_size,
            group_sizes=tuple(a.group_sizes),
            bootstrap_draws=a.bootstrap_draws,
        ),
    )


if __name__ == "__main__":
    main()
