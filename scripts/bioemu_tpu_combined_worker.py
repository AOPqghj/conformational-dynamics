"""Generate missing A3Ms and BioEmu AF2 Evoformer embeddings on a Colab TPU.

This is a single-process, resumable worker for the 4,000-row new-A3M manifest.
Google Drive is the durable write-ahead store. Hugging Face synchronization is
deliberately handled by ``bioemu_hf_sync.py`` so Hub outages cannot stop TPU work.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BIOEMU_VERSION = "1.4.1"
MSA_ARCHIVE_SIZE = 500
CHECKPOINT_SIZE = 25
EMBEDDING_WIDTH = 384
MSA_RETRIES = 6
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
MODEL_ID = "bioemu.colabfold_inline.alphafold2_model_3"
CONFIG = {
    "representation": "alphafold2_evoformer_single",
    "model_type": "alphafold2",
    "model_number": 3,
    "num_recycles": 0,
    "num_ensemble": 1,
    "templates": False,
    "output_dtype": "float32",
    "bioemu_version": BIOEMU_VERSION,
}
QUERY_ONLY_CONFIG = {
    **CONFIG,
    "msa_source": "query_only_a3m",
    "msa_depth": 1,
    "msa_network_access": False,
}
MSA_MODES = ("colabfold_remote", "query_only")


class MsaCallDeadline(BaseException):
    """Escape BioEmu's internal unbounded ``except Exception`` retry loop."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def local_scratch() -> Path:
    configured = os.environ.get("BIOEMU_LOCAL_SCRATCH")
    if configured:
        return Path(configured)
    content = Path("/content")
    return content if content.is_dir() else Path(tempfile.gettempdir())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def query_sequence(text: str) -> str:
    query: list[str] = []
    in_query = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_query:
                break
            in_query = True
            continue
        if in_query:
            query.append(line)
    return "".join(query).replace("-", "").upper()


def query_only_a3m(sequence: str) -> str:
    """Return BioEmu's required A3M container with no homologous sequences."""
    normalized = sequence.upper()
    if not normalized or "\n" in normalized or ">" in normalized:
        raise ValueError("query-only A3M requires one non-empty sequence")
    return f">query\n{normalized}\n"


def a3m_depth(text: str) -> int:
    return sum(line.lstrip().startswith(">") for line in text.splitlines())


def valid_a3m(path: Path, sequence: str) -> bool:
    try:
        return path.is_file() and query_sequence(path.read_text()) == sequence
    except (OSError, UnicodeDecodeError):
        return False


def validate_msa_archive(path: Path, rows: Any) -> int:
    expected = {digest: str(row.sequence) for digest, row in rows.iterrows()}
    with tarfile.open(path, "r:gz") as archive:
        members = {
            Path(member.name).stem: member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".a3m")
        }
        if set(members) != set(expected):
            raise ValueError(f"MSA archive membership mismatch: {path}")
        for digest, member in members.items():
            handle = archive.extractfile(member)
            if handle is None or query_sequence(handle.read().decode()) != expected[digest]:
                raise ValueError(f"MSA archive query mismatch: {digest}")
    return len(members)


class Progress:
    def __init__(self, path: Path, base: dict[str, Any]) -> None:
        self.path = path
        self.state = dict(base)
        self.lock = threading.Lock()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values, updated_at_utc=now_utc())
            atomic_text(self.path, json.dumps(self.state, indent=2, sort_keys=True) + "\n")


class DriveLease:
    """Cross-runtime lease that reclaims only locks from a dead VM or process."""

    def __init__(self, root: Path, manifest_sha256: str) -> None:
        self.root = root
        self.owner = root / "owner.json"
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        self.boot_id = (
            boot_id_path.read_text().strip()
            if boot_id_path.is_file()
            else f"{socket.gethostname()}:{uuid.getnode()}"
        )
        self.payload = {
            "run_id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "boot_id": self.boot_id,
            "manifest_sha256": manifest_sha256,
            "created_at_utc": now_utc(),
        }

    def acquire(self) -> None:
        if self.root.exists():
            stale = True
            try:
                previous = json.loads(self.owner.read_text())
                same_boot = previous.get("boot_id") == self.boot_id
                pid = int(previous.get("pid", -1))
                if same_boot and pid > 0:
                    os.kill(pid, 0)
                    stale = False
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            if not stale:
                raise RuntimeError(f"another live worker owns {self.root}")
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        atomic_text(self.owner, json.dumps(self.payload, indent=2, sort_keys=True) + "\n")

    def release(self) -> None:
        try:
            current = json.loads(self.owner.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if current.get("run_id") == self.payload["run_id"]:
            shutil.rmtree(self.root)


def install_dependencies() -> None:
    try:
        if importlib.metadata.version("bioemu") == BIOEMU_VERSION:
            __import__("bioemu")
            __import__("huggingface_hub")
            __import__("pyarrow")
            return
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pass
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--prefer-binary",
        f"bioemu=={BIOEMU_VERSION}",
        "huggingface_hub",
        "pyarrow",
    ]
    last_error: Exception | None = None
    for attempt in range(1, 7):
        print(f"DEPENDENCY_INSTALL attempt={attempt}/6", flush=True)
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 6:
                time.sleep(min(180, 20 * attempt))
    raise RuntimeError("BioEmu dependency installation failed after six attempts") from last_error


def load_catalog(path: Path, msa_mode: str = "colabfold_remote"):
    import pandas as pd

    catalog = pd.read_csv(path)
    required = {"protein_id", "sequence", "sequence_sha256", "sequence_length", "cache_status"}
    if required - set(catalog):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(catalog))}")
    catalog = catalog.loc[:, sorted(required)].copy()
    catalog["sequence"] = catalog.sequence.astype(str).str.upper()
    if (
        catalog.empty
        or catalog.protein_id.duplicated().any()
        or catalog.sequence_sha256.duplicated().any()
        or not catalog.cache_status.eq(
            "query_only" if msa_mode == "query_only" else "new_a3m_required"
        ).all()
    ):
        raise ValueError("expected a nonempty, unique manifest whose rows require new A3Ms")
    calculated = catalog.sequence.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    if not calculated.equals(catalog.sequence_sha256):
        raise ValueError("manifest sequence hashes do not match")
    if not catalog.sequence.map(lambda value: bool(value) and set(value) <= PROTEIN_ALPHABET).all():
        raise ValueError("manifest contains unsupported protein residues")
    if not catalog.sequence.str.len().equals(catalog.sequence_length.astype(int)):
        raise ValueError("manifest sequence lengths do not match")
    catalog = catalog.sort_values("sequence_sha256").reset_index(drop=True)
    catalog["msa_archive_index"] = catalog.index // MSA_ARCHIVE_SIZE
    catalog["pad_length"] = ((catalog.sequence_length.astype(int) + 15) // 16) * 16
    return catalog


class QueryOnlyMsaStore:
    """Network-free MSA provider that always emits exactly the query sequence."""

    def __init__(self, local_root: Path) -> None:
        self.local_root = local_root
        self.local_root.mkdir(parents=True, exist_ok=True)

    def ensure(self, rows: list[Any]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for row in rows:
            text = query_only_a3m(str(row.sequence))
            if a3m_depth(text) != 1:
                raise AssertionError("query-only A3M must have depth one")
            path = self.local_root / f"{row.sequence_sha256}.a3m"
            if not valid_a3m(path, str(row.sequence)) or path.read_text() != text:
                atomic_text(path, text)
            paths[str(row.sequence_sha256)] = path
        return paths

    def finalize_archives(self) -> None:
        return


def bind_embedding_lane(output_root: Path, manifest_sha256: str) -> None:
    """Prevent a second manifest from overwriting a checkpoint lane."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    config_path = output_root / "run_config.json"
    if manifest_path.is_file() and sha256_file(manifest_path) != manifest_sha256:
        raise ValueError(
            "embedding output root already belongs to another manifest; choose a new "
            "BIOEMU_EMBED_ROOT"
        )
    if config_path.is_file():
        try:
            previous = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"embedding output run configuration is unreadable: {config_path}"
            ) from error
        if previous.get("manifest_sha256") not in {None, manifest_sha256}:
            raise ValueError(
                "embedding output root run configuration belongs to another manifest; "
                "choose a new BIOEMU_EMBED_ROOT"
            )


def bind_msa_lane(msa_root: Path, manifest_sha256: str, output_root: Path) -> None:
    """Bind MSA archives to one manifest so archive indices cannot collide across lanes."""
    msa_root.mkdir(parents=True, exist_ok=True)
    marker_path = msa_root / "lane_manifest.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"MSA lane marker is unreadable: {marker_path}") from error
        if marker.get("manifest_sha256") != manifest_sha256:
            raise ValueError("MSA cache belongs to another manifest; choose a new BIOEMU_MSA_ROOT")
        return
    has_existing_artifacts = any(msa_root.iterdir())
    existing_embedding_manifest = output_root / "manifest.csv"
    if has_existing_artifacts and (
        not existing_embedding_manifest.is_file()
        or sha256_file(existing_embedding_manifest) != manifest_sha256
    ):
        raise ValueError(
            "MSA cache has unbound artifacts and cannot be proven to match this manifest; "
            "choose a new BIOEMU_MSA_ROOT"
        )
    atomic_text(
        marker_path,
        json.dumps({"format_version": 1, "manifest_sha256": manifest_sha256}, indent=2) + "\n",
    )


def hydrate_a3m_archives(catalog, msa_root: Path, local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    expected = set(catalog.sequence_sha256)
    for archive_path in sorted((msa_root / "checkpoints").glob("missing_*.tar.gz")):
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                digest = Path(member.name).stem
                if (
                    not member.isfile()
                    or not member.name.endswith(".a3m")
                    or digest not in expected
                ):
                    continue
                destination = local_root / f"{digest}.a3m"
                if destination.is_file():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable A3M member: {member.name}")
                destination.write_bytes(handle.read())


class MsaStore:
    def __init__(self, catalog, drive_root: Path, local_root: Path, progress: Progress) -> None:
        self.catalog = catalog.set_index("sequence_sha256")
        self.drive_root = drive_root
        self.local_root = local_root
        self.progress = progress
        self.write_lock = threading.Lock()

    def local_path(self, digest: str) -> Path:
        return self.local_root / f"{digest}.a3m"

    def drive_path(self, row: Any) -> Path:
        return (
            self.drive_root
            / "partial"
            / f"missing_{int(row.msa_archive_index):03d}"
            / "a3m"
            / f"{row.sequence_sha256}.a3m"
        )

    def resolve(self, row: Any) -> Path | None:
        local = self.local_path(row.sequence_sha256)
        if valid_a3m(local, row.sequence):
            return local
        drive = self.drive_path(row)
        if valid_a3m(drive, row.sequence):
            shutil.copy2(drive, local)
            return local
        return None

    def save(self, row: Any, text: str) -> Path:
        if query_sequence(text) != row.sequence:
            raise ValueError(f"A3M query mismatch for {row.protein_id}")
        local = self.local_path(row.sequence_sha256)
        local.write_text(text)
        with self.write_lock:
            atomic_copy(local, self.drive_path(row))
        return local

    def ensure(self, rows: list[Any]) -> dict[str, Path]:
        paths = {row.sequence_sha256: self.resolve(row) for row in rows}
        pending = [row for row in rows if paths[row.sequence_sha256] is None]
        batch_size = int(os.environ.get("BIOEMU_MSA_QUERY_BATCH_SIZE", str(len(pending) or 1)))
        if batch_size < 1:
            raise ValueError("BIOEMU_MSA_QUERY_BATCH_SIZE must be positive")
        submit_delay = int(os.environ.get("BIOEMU_MSA_SUBMIT_DELAY_SECONDS", "0"))
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = resilient_mmseqs(
                [row.sequence for row in batch],
                local_scratch() / "bioemu_mmseqs" / uuid.uuid4().hex,
                self.progress,
            )
            for row, text in zip(batch, results, strict=True):
                paths[row.sequence_sha256] = self.save(row, text)
            self.progress.update(
                phase="msa",
                msa_saved_rows=sum(path is not None for path in paths.values()),
                msa_requested_rows=len(rows),
                msa_query_batch_size=batch_size,
            )
            print(
                f"MSA_BATCH_SAVED rows={len(batch)} "
                f"completed={sum(path is not None for path in paths.values())}/{len(rows)}",
                flush=True,
            )
            if submit_delay > 0 and start + batch_size < len(pending):
                time.sleep(submit_delay)
        return {digest: path for digest, path in paths.items() if path is not None}

    def finalize_archives(self) -> None:
        for index in sorted(self.catalog.msa_archive_index.unique()):
            archive_path = self.drive_root / "checkpoints" / f"missing_{index:03d}.tar.gz"
            if archive_path.is_file():
                continue
            rows = self.catalog.loc[self.catalog.msa_archive_index.eq(index)]
            paths = [
                self.drive_root / "partial" / f"missing_{index:03d}" / "a3m" / f"{digest}.a3m"
                for digest in rows.index
            ]
            if not paths or not all(
                valid_a3m(path, str(rows.at[path.stem, "sequence"])) for path in paths
            ):
                continue
            local_archive = local_scratch() / f"missing_{index:03d}.tar.gz"
            with tarfile.open(local_archive, "w:gz") as archive:
                for path in paths:
                    archive.add(path, arcname=f"a3m/{path.name}")
            atomic_copy(local_archive, archive_path)
            validate_msa_archive(archive_path, rows)
            local_archive.unlink(missing_ok=True)
            print(f"MSA_CHECKPOINT index={index} rows={len(paths)}", flush=True)


def resilient_mmseqs(
    sequences: list[str], prefix: Path, progress: Progress, depth: int = 0
) -> list[str]:
    from bioemu.colabfold_inline.msa_client import run_mmseqs2

    prefix.parent.mkdir(parents=True, exist_ok=True)
    retries = int(os.environ.get("BIOEMU_MSA_RETRIES", str(MSA_RETRIES)))
    call_timeout = int(os.environ.get("BIOEMU_MSA_CALL_TIMEOUT_SECONDS", "0"))
    outage_wait = int(os.environ.get("BIOEMU_MSA_OUTAGE_WAIT_SECONDS", "600"))
    if retries < 1 or call_timeout < 0 or outage_wait < 1:
        raise ValueError("MSA retry, timeout, and outage-wait settings must be positive")
    for attempt in range(1, retries + 1):
        shutil.rmtree(prefix, ignore_errors=True)
        shutil.rmtree(f"{prefix}_env", ignore_errors=True)
        try:
            progress.update(
                phase="msa",
                msa_request_size=len(sequences),
                msa_attempt=attempt,
                msa_split_depth=depth,
            )
            result = _bounded_mmseqs_call(run_mmseqs2, sequences, prefix, call_timeout)
            if len(result) != len(sequences):
                raise ValueError(f"expected {len(sequences)} A3Ms, got {len(result)}")
            return result
        except (Exception, MsaCallDeadline) as error:
            delay = min(600, 30 * 2 ** (attempt - 1))
            print(
                f"MSA_RETRY size={len(sequences)} attempt={attempt}/{retries} "
                f"error={type(error).__name__} delay={delay}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(delay)
    if len(sequences) > 1:
        midpoint = len(sequences) // 2
        left = resilient_mmseqs(
            sequences[:midpoint], prefix.with_name(prefix.name + "_l"), progress, depth + 1
        )
        right = resilient_mmseqs(
            sequences[midpoint:], prefix.with_name(prefix.name + "_r"), progress, depth + 1
        )
        return left + right
    print(f"MSA_WAIT single sequence unavailable; retrying in {outage_wait} seconds", flush=True)
    progress.update(
        phase="msa_wait",
        msa_request_size=1,
        msa_wait_seconds=outage_wait,
        msa_split_depth=depth,
    )
    time.sleep(outage_wait)
    return resilient_mmseqs(sequences, prefix, progress, depth + 1)


def _bounded_mmseqs_call(run_mmseqs2: Any, sequences: list[str], prefix: Path, timeout: int):
    """Interrupt BioEmu's unbounded timeout loop when running on the main thread."""
    if timeout == 0 or threading.current_thread() is not threading.main_thread():
        return run_mmseqs2(sequences, prefix=str(prefix))

    def timeout_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise MsaCallDeadline(f"ColabFold submission exceeded {timeout} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return run_mmseqs2(sequences, prefix=str(prefix))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def restore_params(drive_archive: Path, local_root: Path) -> None:
    if not drive_archive.is_file():
        return
    with tarfile.open(drive_archive, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        required = {"params/params_model_3.npz", "params/download_finished.txt"}
        if not required <= names:
            raise ValueError("Drive AF2 parameter checkpoint is incomplete")
        archive.extractall(local_root, filter="data")


def save_params(local_root: Path, drive_archive: Path) -> None:
    if drive_archive.is_file():
        return
    required = [
        local_root / "params" / "params_model_3.npz",
        local_root / "params" / "download_finished.txt",
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("AF2 model 3 parameters were not downloaded completely")
    temporary = local_scratch() / "alphafold_model3_params.tar.gz"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in required:
            archive.add(path, arcname=f"params/{path.name}")
    atomic_copy(temporary, drive_archive)
    temporary.unlink(missing_ok=True)


def load_tpu_model(params_root: Path):
    import importlib.metadata

    import jax
    from bioemu.colabfold_inline.model_runner import _load_model_and_params

    if importlib.metadata.version("bioemu") != BIOEMU_VERSION:
        raise RuntimeError("unexpected BioEmu version")
    devices = jax.devices()
    if not any(device.platform == "tpu" for device in devices):
        raise RuntimeError(f"TPU runtime required; JAX devices={devices}")
    model_runner, _ = _load_model_and_params(params_root)
    return model_runner, [str(device) for device in devices]


def embed_single(model_runner: Any, sequence: str, a3m_text: str, pad_length: int):
    import numpy as np
    from bioemu.colabfold_inline.features import build_monomer_feature
    from bioemu.colabfold_inline.model_runner import _pad_input

    features = build_monomer_feature(sequence, a3m_text)
    inputs = model_runner.process_features(features, random_seed=0)
    if len(sequence) < pad_length:
        inputs = _pad_input(inputs, model_runner, pad_length)
    result, _ = model_runner.predict(inputs, random_seed=0, return_representations=True)
    single = np.asarray(result["representations_evo"]["single"][: len(sequence)], dtype=np.float32)
    del result, inputs, features
    gc.collect()
    return single


def expected_metadata(
    row: Any,
    manifest_sha256: str,
    a3m_path: Path,
    extraction_config: dict[str, Any] = CONFIG,
) -> dict[str, Any]:
    metadata = {
        "protein_id": row.protein_id,
        "sequence_sha256": row.sequence_sha256,
        "sequence_length": int(row.sequence_length),
        "embedding_width": EMBEDDING_WIDTH,
        "model_id": MODEL_ID,
        "model_revision": BIOEMU_VERSION,
        "extraction_config": extraction_config,
        "manifest_sha256": manifest_sha256,
        "a3m_sha256": sha256_file(a3m_path),
    }
    if extraction_config == QUERY_ONLY_CONFIG:
        text = a3m_path.read_text()
        if a3m_depth(text) != 1 or query_sequence(text) != row.sequence:
            raise ValueError("query-only A3M contains homologous or mismatched sequences")
        metadata.update(msa_mode="query_only", msa_depth=1)
    return metadata


def valid_npz(
    path: Path,
    row: Any,
    manifest_sha256: str,
    a3m_path: Path,
    extraction_config: dict[str, Any] = CONFIG,
) -> bool:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"single", "metadata"}:
                return False
            single = archive["single"]
            metadata = json.loads(str(archive["metadata"].item()))
        return (
            single.dtype == np.float32
            and single.shape == (int(row.sequence_length), EMBEDDING_WIDTH)
            and np.isfinite(single).all()
            and metadata == expected_metadata(
                row, manifest_sha256, a3m_path, extraction_config
            )
        )
    except Exception:
        return False


def valid_checkpoint(archive_path: Path, sidecar_path: Path, expected_hashes: list[str]) -> bool:
    try:
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("sequence_sha256") != expected_hashes:
            return False
        if sidecar.get("archive_sha256") != sha256_file(archive_path):
            return False
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name for member in archive.getmembers() if member.isfile()}
        return members == {
            "manifest.json",
            *(f"embeddings/{digest}.npz" for digest in expected_hashes),
        }
    except Exception:
        return False


def finalize_checkpoint(
    index: int,
    rows: list[Any],
    working_root: Path,
    checkpoint_root: Path,
    manifest_sha256: str,
    extraction_config: dict[str, Any] = CONFIG,
) -> None:
    hashes = [row.sequence_sha256 for row in rows]
    local_archive = local_scratch() / f"embedding_{index:05d}.tar.gz"
    manifest = {
        "format_version": 1,
        "checkpoint_index": index,
        "created_at_utc": now_utc(),
        "manifest_sha256": manifest_sha256,
        "sequence_sha256": hashes,
        "model_id": MODEL_ID,
        "model_revision": BIOEMU_VERSION,
        "embedding_width": EMBEDDING_WIDTH,
        "extraction_config": extraction_config,
    }
    manifest_path = local_scratch() / f"embedding_{index:05d}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with tarfile.open(local_archive, "w:gz") as archive:
        archive.add(manifest_path, arcname="manifest.json")
        for digest in hashes:
            archive.add(working_root / f"{digest}.npz", arcname=f"embeddings/{digest}.npz")
    sidecar = {
        **manifest,
        "archive_sha256": sha256_file(local_archive),
        "size": local_archive.stat().st_size,
    }
    archive_path = checkpoint_root / local_archive.name
    sidecar_path = checkpoint_root / f"embedding_{index:05d}.json"
    atomic_copy(local_archive, archive_path)
    atomic_text(sidecar_path, json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    if not valid_checkpoint(archive_path, sidecar_path, hashes):
        raise ValueError(f"embedding checkpoint {index} failed validation")
    shutil.rmtree(working_root)
    local_archive.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)


def main() -> None:
    drive_root = Path(
        os.environ.get(
            "BIOEMU_DRIVE_ROOT", "/content/drive/MyDrive/dynamic_protein_router/bioemu_io"
        )
    )
    manifest_path = Path(
        os.environ.get(
            "BIOEMU_MANIFEST_PATH", str(drive_root / "input/bioemu_af2_6086_new_manifest.csv")
        )
    )
    msa_root = Path(os.environ.get("BIOEMU_MSA_ROOT", str(drive_root / "msa_cache")))
    output_root = Path(
        os.environ.get("BIOEMU_EMBED_ROOT", str(drive_root / "embedding_cache_tpu_v1"))
    )
    max_checkpoints = int(os.environ.get("BIOEMU_MAX_CHECKPOINTS", "0"))
    disable_msa_prefetch = os.environ.get("BIOEMU_DISABLE_MSA_PREFETCH", "0") == "1"
    msa_mode = os.environ.get("BIOEMU_MSA_MODE", "colabfold_remote")
    if msa_mode not in MSA_MODES:
        raise ValueError(f"BIOEMU_MSA_MODE must be one of {MSA_MODES}")
    extraction_config = QUERY_ONLY_CONFIG if msa_mode == "query_only" else CONFIG
    if not manifest_path.is_file():
        raise FileNotFoundError(f"mounted Drive manifest is unavailable: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    bind_embedding_lane(output_root, manifest_sha256)
    if msa_mode != "query_only":
        bind_msa_lane(msa_root, manifest_sha256, output_root)
    catalog = load_catalog(manifest_path, msa_mode)
    expected_rows = len(catalog)
    work = catalog.sort_values(["pad_length", "sequence_sha256"]).reset_index(drop=True)
    lease = DriveLease(output_root / "worker.lock", manifest_sha256)
    msa_lease = (
        DriveLease(msa_root / "worker.lock", manifest_sha256)
        if msa_mode != "query_only"
        else None
    )
    lease.acquire()
    try:
        if msa_lease is not None:
            msa_lease.acquire()
    except BaseException:
        lease.release()
        raise
    progress = Progress(
        output_root / "progress.json",
        {
            "format_version": 1,
            "phase": "starting",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "expected_rows": expected_rows,
            "checkpoint_size": CHECKPOINT_SIZE,
            "started_at_utc": now_utc(),
        },
    )
    try:
        install_dependencies()
        import pandas as pd

        local_a3m = local_scratch() / "bioemu_tpu_a3m"
        if msa_mode == "query_only":
            msa_store = QueryOnlyMsaStore(local_a3m)
        else:
            hydrate_a3m_archives(catalog, msa_root, local_a3m)
            msa_store = MsaStore(catalog, msa_root, local_a3m, progress)
        params_root = local_scratch() / "bioemu_af2_params"
        params_archive = output_root / "alphafold_model3_params.tar.gz"
        restore_params(params_archive, params_root)
        progress.update(phase="loading_model")
        model_runner, devices = load_tpu_model(params_root)
        save_params(params_root, params_archive)
        config = {
            "manifest_sha256": manifest_sha256,
            "expected_rows": expected_rows,
            "checkpoint_size": CHECKPOINT_SIZE,
            "model_id": MODEL_ID,
            "model_revision": BIOEMU_VERSION,
            "extraction_config": extraction_config,
            "msa_mode": msa_mode,
            "jax_devices": devices,
            "order": "pad_length_then_sequence_sha256",
        }
        atomic_text(
            output_root / "run_config.json", json.dumps(config, indent=2, sort_keys=True) + "\n"
        )
        atomic_copy(manifest_path, output_root / "manifest.csv")
        checkpoints = [
            list(work.iloc[start : start + CHECKPOINT_SIZE].itertuples(index=False))
            for start in range(0, len(work), CHECKPOINT_SIZE)
        ]
        if max_checkpoints:
            if max_checkpoints < 1:
                raise ValueError("BIOEMU_MAX_CHECKPOINTS must be positive")
            checkpoints = checkpoints[:max_checkpoints]
        expected_this_run = sum(len(rows) for rows in checkpoints)
        completed = 0
        failed_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="msa-prefetch") as executor:
            future: Future[dict[str, Path]] | None = None
            for index, rows in enumerate(checkpoints):
                hashes = [row.sequence_sha256 for row in rows]
                archive_path = output_root / "checkpoints" / f"embedding_{index:05d}.tar.gz"
                sidecar_path = output_root / "checkpoints" / f"embedding_{index:05d}.json"
                if valid_checkpoint(archive_path, sidecar_path, hashes):
                    if future is not None:
                        future.result()
                        future = None
                    completed += len(rows)
                    progress.update(
                        phase="running",
                        completed_rows=completed,
                        checkpoint_index=index,
                        reused=True,
                    )
                    continue
                a3m_paths = future.result() if future is not None else msa_store.ensure(rows)
                next_rows = checkpoints[index + 1] if index + 1 < len(checkpoints) else None
                future = (
                    executor.submit(msa_store.ensure, next_rows)
                    if next_rows is not None and not disable_msa_prefetch
                    else None
                )
                working_root = output_root / "in_progress" / f"embedding_{index:05d}"
                working_root.mkdir(parents=True, exist_ok=True)
                for position, row in enumerate(rows, start=1):
                    a3m_path = a3m_paths[row.sequence_sha256]
                    destination = working_root / f"{row.sequence_sha256}.npz"
                    if not valid_npz(
                        destination, row, manifest_sha256, a3m_path, extraction_config
                    ):
                        try:
                            single = embed_single(
                                model_runner,
                                row.sequence,
                                a3m_path.read_text(),
                                int(row.pad_length),
                            )
                            if single.shape != (int(row.sequence_length), EMBEDDING_WIDTH):
                                raise ValueError(f"invalid single shape {single.shape}")
                            metadata = expected_metadata(
                                row, manifest_sha256, a3m_path, extraction_config
                            )
                            local_npz = local_scratch() / f"{row.sequence_sha256}.npz"
                            import numpy as np

                            with local_npz.open("wb") as handle:
                                np.savez_compressed(
                                    handle,
                                    single=single,
                                    metadata=np.array(json.dumps(metadata, sort_keys=True)),
                                )
                            atomic_copy(local_npz, destination)
                            local_npz.unlink(missing_ok=True)
                            if not valid_npz(
                                destination, row, manifest_sha256, a3m_path, extraction_config
                            ):
                                raise ValueError("written embedding failed validation")
                        except Exception as error:
                            failed_records.append(
                                {
                                    "protein_id": row.protein_id,
                                    "sequence_sha256": row.sequence_sha256,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            pd.DataFrame(failed_records).to_csv(
                                output_root / "failed.csv", index=False
                            )
                            progress.update(
                                phase="embedding_error",
                                checkpoint_index=index,
                                checkpoint_position=position,
                                failed_rows=len(failed_records),
                            )
                            print(
                                f"EMBEDDING_FAILED protein={row.protein_id} error={type(error).__name__}: {error}",
                                flush=True,
                            )
                            continue
                    completed += 1
                    progress.update(
                        phase="embedding",
                        completed_rows=completed,
                        checkpoint_index=index,
                        checkpoint_position=position,
                        checkpoint_rows=len(rows),
                        protein_id=row.protein_id,
                        pad_length=int(row.pad_length),
                        failed_rows=len(failed_records),
                    )
                    print(
                        f"EMBEDDING_PROGRESS completed={completed}/{expected_rows} "
                        f"checkpoint={index + 1}/{len(checkpoints)} protein={row.protein_id}",
                        flush=True,
                    )
                if all(
                    valid_npz(
                        working_root / f"{row.sequence_sha256}.npz",
                        row,
                        manifest_sha256,
                        a3m_paths[row.sequence_sha256],
                        extraction_config,
                    )
                    for row in rows
                ):
                    finalize_checkpoint(
                        index,
                        rows,
                        working_root,
                        output_root / "checkpoints",
                        manifest_sha256,
                        extraction_config,
                    )
                    print(f"EMBEDDING_CHECKPOINT index={index} rows={len(rows)}", flush=True)
                msa_store.finalize_archives()
        if completed == expected_rows and not failed_records:
            phase = "complete"
        elif max_checkpoints and completed == expected_this_run and not failed_records:
            phase = "smoke_complete"
        else:
            phase = "incomplete"
        progress.update(phase=phase, completed_rows=completed, failed_rows=len(failed_records))
        print(
            "BIOEMU_TPU_COMPLETE="
            + json.dumps({"phase": phase, "completed": completed, "failed": len(failed_records)}),
            flush=True,
        )
    except BaseException as error:
        progress.update(
            phase="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if msa_lease is not None:
            msa_lease.release()
        lease.release()


if __name__ == "__main__":
    main()
