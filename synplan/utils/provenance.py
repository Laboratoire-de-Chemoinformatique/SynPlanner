"""Source fingerprints and atomic JSON artifacts for reproducible planning."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path

import synplan


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "SynPlanner",
        "chython-synplan",
        "chytorch-synplan",
        "torch",
        "rdkit",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def source_identity(package_path: Path) -> dict:
    """Fingerprint loaded source, including local changes, rather than just a tag."""
    root = package_path.resolve()
    files = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    result = {"path": str(root), "source_sha256": digest.hexdigest()}
    try:
        # An unpacked source tree may sit inside an unrelated repository.
        # Report a git revision only when this source is tracked by that repo.
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                str(files[0].relative_to(root)),
            ],
            stderr=subprocess.DEVNULL,
        )
        result["revision"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        result["dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision_file = root.parent / "REVISION"
        result["revision"] = (
            revision_file.read_text().strip() if revision_file.exists() else None
        )
    return result


def runtime_identity(resources: dict[str, str]) -> dict:
    return {
        "installed_versions": package_versions(),
        "synplanner_source": source_identity(Path(synplan.__file__).parent),
        "resources": {
            name: {"path": str(path), "sha256": sha256(Path(path))}
            for name, path in resources.items()
        },
    }


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            if path.suffix == ".gz":
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                    compressed.write(json.dumps(data, allow_nan=False).encode())
            else:
                raw.write(json.dumps(data, indent=2, allow_nan=False).encode())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)
