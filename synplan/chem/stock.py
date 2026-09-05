"""Versioned stock conversion using the same Chython normalization as search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from synplan.chem import utils as chem_utils
from synplan.utils.files import iter_csv_smiles
from synplan.utils.provenance import atomic_json, package_versions, read_json, sha256

NORMALIZATION = "safe_canonicalization"


def normalization_identity() -> dict[str, str]:
    return {
        "name": NORMALIZATION,
        "chython_version": package_versions()["chython-synplan"],
        "implementation_sha256": sha256(
            Path(chem_utils.safe_canonicalization.__code__.co_filename)
        ),
    }


def load_stock_cache(
    stock_path: Path, cache_dir: Path | None = None
) -> tuple[set[str], dict]:
    """A cache is usable only for this source content and normalization version.

    Conversion failures are retained with input record numbers and SMILES. Cache
    records are atomically replaced and their normalized contents are verified.
    """
    identity = {
        "source_sha256": sha256(stock_path),
        "normalization": normalization_identity(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = (
        cache_dir or stock_path.parent / "synplanner-cache"
    ) / f"{key}.json.gz"
    if cache_path.exists():
        try:
            record = read_json(cache_path)
            contents_hash = hashlib.sha256(
                "\n".join(record["smiles"]).encode()
            ).hexdigest()
            if (
                record["identity"] == identity
                and record["contents_sha256"] == contents_hash
            ):
                return set(record["smiles"]), {
                    k: v for k, v in record.items() if k != "smiles"
                }
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            pass
    inputs = list(iter_csv_smiles(stock_path))
    failures = []
    molecules = set(chem_utils.standardize_smiles_batch(inputs, failures=failures))
    strings = sorted(molecules)
    record = {
        "identity": identity,
        "cache_path": str(cache_path),
        "input_rows": len(inputs),
        "unique_molecules": len(strings),
        "failures": failures,
        "smiles": strings,
        "contents_sha256": hashlib.sha256("\n".join(strings).encode()).hexdigest(),
    }
    atomic_json(cache_path, record)
    return molecules, {k: v for k, v in record.items() if k != "smiles"}
