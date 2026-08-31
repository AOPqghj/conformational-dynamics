"""Zenodo manifest, verification, download, and safe extraction helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import certifi

DYNAMICMPNN_ZENODO_RECORD = "19687631"
DYNAMICMPNN_ZENODO_DOI = "10.5281/zenodo.19687631"
DYNAMICMPNN_FILES = {
    "test_pt_multi_chain.tar": {"size": "4.0 MB", "md5": "050e1c17a842bc8aac0249f5f3a11042"},
    "test_pt_single_chain.tar": {"size": "3.4 MB", "md5": "1cf2161731f2b2b2e05779a016e8f0e2"},
    "train_pt_multi_chain.tar.gz": {"size": "3.7 GB", "md5": "f53fbce346f32dbfbe5c882be8cd5c35"},
    "train_pt_single_chain.tar.gz": {"size": "1.8 GB", "md5": "23b53722808866ea490c4e9f4cc83821"},
    "val_pt_multi_chain.tar": {"size": "4.3 MB", "md5": "1336956ba48f4bc7e17763a5fdfb9d70"},
    "val_pt_single_chain.tar": {"size": "3.7 MB", "md5": "55b9476d784ccaf955bbdba53a42ce93"},
}
SUBSETS = {
    "smoke": ("test_pt_single_chain.tar", "val_pt_single_chain.tar"),
    "val_test": (
        "test_pt_single_chain.tar",
        "val_pt_single_chain.tar",
        "test_pt_multi_chain.tar",
        "val_pt_multi_chain.tar",
    ),
    "train_single": ("train_pt_single_chain.tar.gz",),
    "train_multi": ("train_pt_multi_chain.tar.gz",),
    "all": tuple(DYNAMICMPNN_FILES),
}


def zenodo_file_url(filename: str) -> str:
    if filename not in DYNAMICMPNN_FILES:
        raise ValueError(f"unknown DynamicMPNN archive: {filename}")
    return f"https://zenodo.org/records/{DYNAMICMPNN_ZENODO_RECORD}/files/{filename}?download=1"


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()  # nosec B324 - Zenodo publishes MD5 as its integrity metadata.
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(
    archive_dir: str | Path, filenames: tuple[str, ...] | None = None
) -> dict[str, dict[str, object]]:
    """Find known archives recursively and report checksum status without moving them."""
    root = Path(archive_dir)
    names = filenames or tuple(DYNAMICMPNN_FILES)
    report: dict[str, dict[str, object]] = {}
    for name in names:
        matches = sorted(root.rglob(name))
        if len(matches) > 1:
            raise ValueError(f"multiple copies of {name} found below {root}")
        path = matches[0] if matches else None
        actual = md5_file(path) if path else None
        report[name] = {
            "path": str(path) if path else None,
            "expected_md5": DYNAMICMPNN_FILES[name]["md5"],
            "actual_md5": actual,
            "valid": actual == DYNAMICMPNN_FILES[name]["md5"],
        }
    return report


def download_subset(subset: str, output_dir: str | Path) -> dict[str, dict[str, object]]:
    """Download selected files, skipping only files that pass the published MD5 check."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {sorted(SUBSETS)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in SUBSETS[subset]:
        path = output / name
        if path.is_file() and md5_file(path) == DYNAMICMPNN_FILES[name]["md5"]:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        with (
            urlopen(
                zenodo_file_url(name),
                timeout=120,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as response,
            temporary.open("wb") as handle,
        ):  # nosec B310
            shutil.copyfileobj(response, handle)
        if md5_file(temporary) != DYNAMICMPNN_FILES[name]["md5"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch after downloading {name}")
        temporary.replace(path)
    report = verify_archives(output, SUBSETS[subset])
    (output / "download_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def extract_archives(
    archive_dir: str | Path, output_dir: str | Path, force: bool = False
) -> dict[str, dict[str, object]]:
    """Safely extract valid known archives into one stable folder per archive stem."""
    report = verify_archives(archive_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, dict[str, object]] = {}
    for name, status in report.items():
        if status["path"] is None:
            continue
        if not status["valid"]:
            raise ValueError(f"refusing to extract checksum-invalid archive {name}")
        stem = name.removesuffix(".tar.gz").removesuffix(".tar")
        target = destination / stem
        if target.exists() and not force:
            extracted[name] = {
                "path": str(target),
                "status": "existing",
                "pt_files": len(list(target.rglob("*.pt"))),
            }
            continue
        with tempfile.TemporaryDirectory(
            prefix="dynamicmpnn-extract-", dir=destination
        ) as temporary:
            staging = Path(temporary) / stem
            staging.mkdir()
            with tarfile.open(str(status["path"]), "r:*") as archive:
                _safe_extract_tar(archive, staging)
            nested = staging / stem
            if nested.is_dir():
                staging = nested
            incoming = destination / f".{stem}.incoming"
            if incoming.exists():
                shutil.rmtree(incoming)
            staging.replace(incoming)
            if target.exists():
                shutil.rmtree(target)
            incoming.replace(target)
        extracted[name] = {
            "path": str(target),
            "status": "extracted",
            "pt_files": len(list(target.rglob("*.pt"))),
        }
    (destination / "extraction_manifest.json").write_text(
        json.dumps(extracted, indent=2, sort_keys=True)
    )
    return extracted


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        target = destination / member.name
        if (
            member.issym()
            or member.islnk()
            or not target.resolve().is_relative_to(destination.resolve())
        ):
            raise ValueError(f"unsafe archive member: {member.name}")
    archive.extractall(destination, filter="data")
