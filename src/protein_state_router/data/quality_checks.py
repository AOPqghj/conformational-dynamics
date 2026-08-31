"""Dataset invariants and compact, serializable quality reports."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

REQUIRED = (
    "protein_id",
    "sequence",
    "source_dataset",
    "dataset_label",
    "single_structure_insufficient",
    "label_confidence_tier",
)


def build_bias_report(
    dataset: pd.DataFrame, screened: pd.DataFrame | None = None
) -> dict[str, object]:
    """Return validation diagnostics and descriptive source-screening summaries."""
    report: dict[str, object] = {"dataset": validate_router_dataset(dataset)}
    for column in ("n_experimental_structures", "max_ca_rmsd", "aligned_coverage"):
        if column in dataset:
            report[f"{column}_by_class"] = (
                dataset.groupby("dataset_label")[column].describe().to_dict()
            )
    if screened is not None:
        for column in (
            "candidate_status",
            "secondary_review",
            "has_contextual_signal",
            "has_condition_signal",
        ):
            if column in screened:
                report[f"screened_{column}_counts"] = (
                    screened[column].fillna("missing").value_counts().to_dict()
                )
    return report


def report_markdown(report: dict[str, object], title: str = "Router dataset bias report") -> str:
    """Render a compact Markdown report from a bias report."""
    return f"# {title}\n\n```json\n{json.dumps(report, indent=2, default=str)}\n```\n"


def validate_router_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Raise for hard leakage/schema errors and return descriptive diagnostics."""
    missing = [column for column in REQUIRED if column not in frame or frame[column].isna().any()]
    if missing:
        raise ValueError(f"Missing required dataset values: {missing}")
    if set(frame.dataset_label.unique()) != {0, 1}:
        raise ValueError("Dataset must contain labels 0 and 1")
    if frame.protein_id.duplicated().any():
        raise ValueError("Duplicate protein_id")
    sequence_hash = frame.sequence.astype(str).map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    classes = frame.dataset_label.astype(int)
    if set(sequence_hash[classes.eq(0)]) & set(sequence_hash[classes.eq(1)]):
        raise ValueError("Cross-class exact sequence overlap")
    if "uniprot_id" in frame or "uniprot_ids_json" in frame:
        identifiers = _uniprot_identifiers(frame, classes)
        if identifiers[0] & identifiers[1]:
            raise ValueError("Cross-class UniProt overlap")
    _reject_high_similarity_cross_class_pairs(frame.sequence.astype(str), classes)

    def counts(column: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for key, number in frame.groupby(["dataset_label", column]).size().items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"Unexpected grouped key: {key!r}")
            label, value = key
            result[f"{label}:{value}"] = int(number)
        return result

    return {
        "rows": len(frame),
        "class_counts": classes.value_counts().sort_index().to_dict(),
        "unique_sequences_by_class": {
            str(label): int(sequence_hash[classes.eq(label)].nunique()) for label in (0, 1)
        },
        "duplicate_sequences_by_class": {
            str(label): int(sequence_hash[classes.eq(label)].duplicated().sum()) for label in (0, 1)
        },
        "sequence_length_by_class": {
            str(label): frame.loc[classes.eq(label), "sequence"]
            .astype(str)
            .str.len()
            .describe()
            .to_dict()
            for label in (0, 1)
        },
        "source_dataset_counts": counts("source_dataset"),
        "label_confidence_tier_counts": counts("label_confidence_tier"),
    }


def _uniprot_identifiers(frame: pd.DataFrame, classes: pd.Series) -> dict[int, set[str]]:
    values: dict[int, set[str]] = {0: set(), 1: set()}
    for position, row in enumerate(frame.to_dict("records")):
        label = int(classes.iloc[position])
        uniprot_id = row.get("uniprot_id")
        if uniprot_id is not None and pd.notna(uniprot_id):
            values[label].add(str(uniprot_id))
        identifiers_json = row.get("uniprot_ids_json")
        if identifiers_json is None or pd.isna(identifiers_json):
            continue
        try:
            identifiers = json.loads(str(identifiers_json))
        except json.JSONDecodeError:
            continue
        values[label].update(str(value) for value in identifiers if value)
    return values


def _reject_high_similarity_cross_class_pairs(sequences: pd.Series, classes: pd.Series) -> None:
    """Reject near-duplicate cross-class sequences with a global 3-mer cosine guard."""
    positive = sequences.loc[classes.eq(1)].tolist()
    negative = sequences.loc[classes.eq(0)].tolist()
    if not positive or not negative:
        return
    matrix = CountVectorizer(
        analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False
    ).fit_transform([*positive, *negative])
    similarity = normalize(matrix[: len(positive)]) @ normalize(matrix[len(positive) :]).T
    if similarity.nnz and float(similarity.data.max()) >= 0.95:
        raise ValueError("Cross-class high sequence similarity (3-mer cosine >= 0.95)")
