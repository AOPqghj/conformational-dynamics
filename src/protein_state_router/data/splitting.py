"""Leakage-resistant catalog split generation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from protein_state_router.constants import DEFAULT_SPLIT_FRACTIONS

ROUTER_SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def make_splits(
    catalog: pd.DataFrame,
    mode: str = "cluster",
    seed: int = 42,
    holdout_source: str | None = None,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
) -> pd.DataFrame:
    """Return a deterministic split table; grouped modes keep each group intact."""
    if mode not in {"random", "cluster", "family", "source"}:
        raise ValueError("mode must be random, cluster, family, or source")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    frame = catalog.copy()
    if mode == "source":
        source = holdout_source or sorted(frame.source_dataset.unique())[-1]
        frame["split"] = np.where(frame.source_dataset.eq(source), "test", "train")
        train_ids = frame.index[frame.split.eq("train")].to_numpy()
        rng.shuffle(train_ids)
        n_valid = max(1, round(len(train_ids) * fractions[1] / (fractions[0] + fractions[1])))
        frame.loc[train_ids[:n_valid], "split"] = "val"
    else:
        if mode == "random":
            groups = pd.Series(frame.protein_id.values, index=frame.index)
        elif mode == "family":
            groups = frame.family_id.fillna(frame.sequence_cluster_id)
        else:
            groups = frame.sequence_cluster_id
        unique_groups = np.array(sorted(groups.unique()))
        rng.shuffle(unique_groups)
        cut_train = round(len(unique_groups) * fractions[0])
        cut_valid = cut_train + round(len(unique_groups) * fractions[1])
        labels = {group: "train" for group in unique_groups[:cut_train]}
        labels.update({group: "val" for group in unique_groups[cut_train:cut_valid]})
        labels.update({group: "test" for group in unique_groups[cut_valid:]})
        frame["split"] = groups.map(labels)
    result = frame[["protein_id", "sequence_cluster_id", "split"]].copy()
    assert_no_leakage(frame, mode)
    return result


def assert_no_leakage(catalog_with_split: pd.DataFrame, mode: str = "cluster") -> None:
    """Reject exact sequence, cluster, or family leakage as appropriate."""
    for column in ("sequence", "sequence_cluster_id"):
        if (
            column in catalog_with_split
            and (catalog_with_split.groupby(column).split.nunique() > 1).any()
        ):
            raise ValueError(f"{column} crosses split boundaries")
    if mode == "family" and "family_id" in catalog_with_split:
        grouped = (
            catalog_with_split.dropna(subset=["family_id"]).groupby("family_id").split.nunique()
        )
        if (grouped > 1).any():
            raise ValueError("family_id crosses split boundaries")


def save_splits(splits: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)


def prevalence_report(catalog: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    merged = catalog.merge(splits[["protein_id", "split"]], on="protein_id", suffixes=("", "_new"))
    split = merged.pop("split_new") if "split_new" in merged else merged["split"]
    return (
        merged.assign(split=split)
        .groupby("split")
        .single_structure_insufficient.agg(["count", "mean"])
    )


def make_grouped_splits(
    dataset: pd.DataFrame,
    seed: int = 42,
    *,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign whole homology groups while balancing rows and both labels."""
    data = dataset.copy()
    required = {"protein_id", "dataset_label"}
    if missing := required - set(data):
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    selected_column = group_column or _default_group_column(data)
    data["group_id"] = _group_ids(data, selected_column)
    group_counts = data.groupby(["group_id", "dataset_label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in group_counts:
            group_counts[label] = 0
    group_counts = group_counts[[0, 1]].astype(int)
    if len(group_counts) < len(ROUTER_SPLITS):
        raise ValueError("too few homology groups for train/val/test")
    mapping = _balanced_group_assignment(group_counts, seed)
    result = data[["protein_id", "group_id", "dataset_label"]].copy()
    result["split"] = result.group_id.map(mapping)
    _assert_safe_grouped_splits(data, result)
    names = [name for name, _ in ROUTER_SPLITS]
    row_counts = result.split.value_counts().reindex(names, fill_value=0)
    label_counts = result.groupby(["split", "dataset_label"]).size()
    return result, {
        "seed": seed,
        "group_column": selected_column,
        "assignment_method": "deterministic_greedy_row_and_class_balance_v1",
        "target_fractions": dict(ROUTER_SPLITS),
        "rows_by_split": row_counts.to_dict(),
        "row_fraction_by_split": (row_counts / len(result)).to_dict(),
        "class_counts_by_split": {
            f"{split}:{label}": int(rows) for (split, label), rows in label_counts.items()
        },
        "groups": int(result.group_id.nunique()),
        "mixed_label_groups": int((group_counts.gt(0).sum(axis=1) > 1).sum()),
        "largest_group": int(group_counts.sum(axis=1).max()),
    }


def _default_group_column(data: pd.DataFrame) -> str:
    for column in ("homology_group_id", "sequence_cluster_id"):
        if column in data:
            return column
    raise ValueError(
        "a homology grouping column is required; refusing to fall back to exact sequence hashes"
    )


def _group_ids(data: pd.DataFrame, group_column: str) -> pd.Series:
    if group_column not in data:
        raise ValueError(f"homology grouping column is missing: {group_column}")
    groups = data[group_column]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError(f"homology grouping column is incomplete: {group_column}")
    return groups.astype(str)


def _balanced_group_assignment(group_counts: pd.DataFrame, seed: int) -> dict[str, str]:
    names = [name for name, _ in ROUTER_SPLITS]
    fractions = pd.Series(dict(ROUTER_SPLITS), dtype=float)
    totals = group_counts.sum(axis=0).to_numpy(dtype=float)
    targets = fractions.to_numpy()[:, None] * totals[None, :]
    target_rows = fractions.to_numpy() * float(group_counts.to_numpy().sum())
    assigned = pd.DataFrame(0.0, index=names, columns=[0, 1])
    assigned_rows = pd.Series(0.0, index=names)
    ordered = sorted(
        group_counts.index.astype(str),
        key=lambda group: (
            -int(group_counts.loc[group].sum()),
            -abs(int(group_counts.loc[group, 1]) - int(group_counts.loc[group, 0])),
            hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
        ),
    )
    mapping: dict[str, str] = {}
    for group in ordered:
        values = group_counts.loc[group].to_numpy(dtype=float)
        size = float(values.sum())
        candidates: list[tuple[float, str]] = []
        for index, name in enumerate(names):
            proposed = assigned.to_numpy(copy=True)
            proposed[index] += values
            proposed_rows = assigned_rows.to_numpy(copy=True)
            proposed_rows[index] += size
            class_error = np.square((proposed - targets) / np.maximum(targets, 1.0)).sum()
            row_error = np.square(
                (proposed_rows - target_rows) / np.maximum(target_rows, 1.0)
            ).sum()
            overflow = np.maximum(proposed_rows - target_rows, 0.0)
            overflow_penalty = np.square(overflow / np.maximum(target_rows, 1.0)).sum()
            tie = int(hashlib.sha256(f"{seed}:{group}:{name}".encode()).hexdigest()[:8], 16)
            candidates.append(
                (class_error + row_error + 4 * overflow_penalty + tie / 2**32 * 1e-9, name)
            )
        selected = min(candidates)[1]
        mapping[group] = selected
        assigned.loc[selected] += values
        assigned_rows.loc[selected] += size
    return mapping


def _assert_safe_grouped_splits(data: pd.DataFrame, splits: pd.DataFrame) -> None:
    merged = data.drop(columns="split", errors="ignore").merge(
        splits[["protein_id", "split"]], on="protein_id", validate="one_to_one"
    )
    for column in ("protein_id", "group_id"):
        if (merged.groupby(column).split.nunique() > 1).any():
            raise ValueError(f"{column} crosses split boundaries")
    if set(merged.split) != {"train", "val", "test"}:
        raise ValueError("missing split")
    if (merged.groupby("split").dataset_label.nunique() != 2).any():
        raise ValueError("each split must contain both classes")
