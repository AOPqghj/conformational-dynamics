"""Cache AlphaFold DB residue pLDDT vectors and build query-mapped summaries."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
DEFAULT_CATALOG = Path(
    "data/lifecycle/final/initial_8598_dataset/homology35_seed42/catalog.parquet"
)
DEFAULT_CACHE = Path("data/cache/alphafold_plddt_vectors.parquet")
DEFAULT_FEATURES = Path(
    "data/lifecycle/final/initial_8598_dataset/analysis/alphafold_plddt_distribution_features.parquet"
)
THREAD_LOCAL = threading.local()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "protein-state-router/0.1 residue-pLDDT-control"})
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2)
        session.mount("https://", adapter)
        THREAD_LOCAL.session = session
    return session


def _json_request(url: str, attempts: int = 4) -> Any:
    session = _session()
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=45)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            if response.status_code == 429 or response.status_code >= 500:
                delay = float(response.headers.get("Retry-After", 2**attempt))
                time.sleep(min(delay, 20.0))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError(f"request failed after {attempts} attempts: {url}") from error
            time.sleep(min(2**attempt, 10.0))
    raise RuntimeError(f"request failed: {url}")


def fetch_vector(accession: str) -> dict[str, object]:
    """Fetch one current AFDB sequence and its aligned per-residue confidence vector."""
    try:
        models = _json_request(API_URL.format(accession=accession))
        if not isinstance(models, list) or not models:
            raise ValueError("prediction response is empty")
        model = models[0]
        sequence = str(model["sequence"])
        plddt_url = str(model["plddtDocUrl"])
        payload = _json_request(plddt_url)
        scores = np.asarray(payload.get("confidenceScore", []), dtype=np.float32)
        if not sequence or scores.ndim != 1 or len(scores) != len(sequence):
            raise ValueError("AFDB sequence and confidence vector lengths do not match")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 100)):
            raise ValueError("AFDB confidence vector is invalid")
        return {
            "uniprot_accession": accession,
            "status": "ok",
            "entry_id": str(model.get("entryId", "unknown")),
            "model_created_date": str(model.get("modelCreatedDate", "unknown")),
            "sequence": sequence,
            "plddt_url": plddt_url,
            "scores": scores.tolist(),
        }
    except Exception as error:  # noqa: BLE001 - checkpoint the accession-level failure
        return {
            "uniprot_accession": accession,
            "status": f"error:{type(error).__name__}",
            "error": str(error),
        }


def cache_vectors(
    catalog: pd.DataFrame,
    cache_path: Path,
    workers: int,
    checkpoint_every: int,
) -> pd.DataFrame:
    """Resume the accession cache and atomically checkpoint network results."""
    cached = pd.read_parquet(cache_path) if cache_path.is_file() else pd.DataFrame()
    records = (
        cached.drop_duplicates("uniprot_accession", keep="last")
        .set_index("uniprot_accession")
        .to_dict("index")
        if not cached.empty
        else {}
    )
    accessions = sorted(catalog.uniprot_accession.unique())
    pending = [
        accession
        for accession in accessions
        if accession not in records or records[accession].get("status") != "ok"
    ]
    print(
        json.dumps(
            {
                "event": "plddt_cache_started",
                "accessions": len(accessions),
                "cached_ok": len(accessions) - len(pending),
                "pending": len(pending),
                "workers": workers,
            }
        ),
        flush=True,
    )

    def checkpoint() -> None:
        output = pd.DataFrame(
            [{"uniprot_accession": key, **value} for key, value in sorted(records.items())]
        )
        _write_parquet(output, cache_path)
        ok = int(output.status.eq("ok").sum())
        print(
            json.dumps(
                {
                    "event": "plddt_cache_checkpoint",
                    "new_completed": completed,
                    "new_total": len(pending),
                    "cached_ok": ok,
                    "cached_total": len(output),
                }
            ),
            flush=True,
        )

    completed = 0
    pending_iter = iter(pending)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, object]], str] = {}
    try:
        for accession in pending_iter:
            futures[executor.submit(fetch_vector, accession)] = accession
            if len(futures) == workers:
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                accession = str(result.pop("uniprot_accession"))
                records[accession] = result
                completed += 1
                next_accession = next(pending_iter, None)
                if next_accession is not None:
                    futures[executor.submit(fetch_vector, next_accession)] = next_accession
                if completed % checkpoint_every == 0 or completed == len(pending):
                    checkpoint()
    except KeyboardInterrupt:
        if completed:
            checkpoint()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()
    result = pd.read_parquet(cache_path)
    failures = result.loc[~result.status.eq("ok")]
    if not failures.empty:
        examples = failures[["uniprot_accession", "status"]].head(10).to_dict("records")
        raise RuntimeError(
            f"{len(failures)} AlphaFold DB accessions remain unavailable: {examples}"
        )
    return result


def _distribution_features(scores: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(scores, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "plddt_mean": float(scores.mean()),
        "plddt_std": float(scores.std(ddof=0)),
        "plddt_q10": float(quantiles[0]),
        "plddt_q25": float(quantiles[1]),
        "plddt_median": float(quantiles[2]),
        "plddt_q75": float(quantiles[3]),
        "plddt_q90": float(quantiles[4]),
        "plddt_fraction_below_50": float(np.mean(scores < 50)),
        "plddt_fraction_below_70": float(np.mean(scores < 70)),
        "plddt_fraction_below_90": float(np.mean(scores < 90)),
    }


def build_feature_table(catalog: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    """Summarize each complete AFDB vector used for the catalog mean."""
    lookup = cache.set_index("uniprot_accession")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(catalog.itertuples(index=False), start=1):
        cached = lookup.loc[row.uniprot_accession]
        target = str(cached.sequence)
        full_scores = np.asarray(cached.scores, dtype=np.float32)
        if len(target) != len(full_scores):
            raise ValueError(f"cached sequence/vector mismatch for {row.uniprot_accession}")
        full_mean_difference = abs(float(full_scores.mean()) - float(row.alphafold_mean_plddt))
        if full_mean_difference > 0.01:
            raise ValueError(
                f"AlphaFold DB version drift for {row.uniprot_accession}: "
                f"mean difference={full_mean_difference:.4f}"
            )
        rows.append(
            {
                "protein_id": row.protein_id,
                "uniprot_accession": row.uniprot_accession,
                "alphafold_model_id": row.alphafold_model_id,
                "vector_residue_count": len(full_scores),
                "full_mean_difference_from_catalog": full_mean_difference,
                **_distribution_features(full_scores),
            }
        )
        if index % 500 == 0 or index == len(catalog):
            print(
                json.dumps(
                    {
                        "event": "plddt_mapping_progress",
                        "completed_proteins": index,
                        "total_proteins": len(catalog),
                    }
                ),
                flush=True,
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint_every must be positive")
    catalog = pd.read_parquet(args.catalog)
    required = {
        "protein_id",
        "sequence",
        "uniprot_accession",
        "alphafold_mean_plddt",
        "alphafold_model_id",
    }
    if missing := required - set(catalog):
        raise ValueError(f"catalog missing columns: {sorted(missing)}")
    catalog = catalog.loc[catalog.alphafold_mean_plddt.notna()].copy()
    if len(catalog) != 7_032 or catalog.protein_id.duplicated().any():
        raise ValueError("expected the 7,032-protein pLDDT-observed cohort")
    cache = cache_vectors(catalog, args.cache, args.workers, args.checkpoint_every)
    features = build_feature_table(catalog, cache)
    _write_parquet(features, args.output)
    print(
        json.dumps(
            {
                "event": "plddt_features_completed",
                "proteins": len(features),
                "accessions": int(features.uniprot_accession.nunique()),
                "maximum_catalog_mean_difference": float(
                    features.full_mean_difference_from_catalog.max()
                ),
                "output": str(args.output),
            }
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
