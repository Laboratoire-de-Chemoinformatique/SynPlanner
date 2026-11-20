"""Successful molecular calculations, selective invalidation, and atomic checkpoints."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from synplan.utils.files import read_json, write_json_atomic

from .config import AIMNetConfig, MolecularCalculationError
from .molecular import PROTON_REFERENCE
from .registry import MolecularRegistry
from .results import ASSESSMENT_VERSION
from .species import is_proton

CACHE_VERSION = 1


class MolecularCalculator(Protocol):
    """Adapter identity identifies model/runtime; overrides require symmetry_number."""

    def identity(self) -> dict: ...

    def calculate(self, smiles: str) -> dict: ...


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def symmetry_error(smiles: str, result: dict, config: AIMNetConfig) -> str | None:
    """Legacy default results may omit metadata; overrides require proof."""
    number = result.get("symmetry_number")
    if number is None and smiles not in config.symmetry_numbers:
        return None
    expected = config.symmetry_numbers.get(smiles, 1)
    if type(number) is not int or number != expected:
        return (
            f"Molecular symmetry for {smiles}: recorded {number!r}, expected {expected}"
        )
    return None


class MolecularCache:
    """One writer, preserving the existing molecular cache format and identity."""

    def __init__(self, config: AIMNetConfig, path: Path | None):
        self.config = config
        self.cache_path = path
        self.disk_backed = path is not None and path.suffix in {
            ".sqlite",
            ".sqlite3",
            ".db",
        }
        self.data: dict[str, Any] = (
            read_json(path)
            if not self.disk_backed and path is not None and path.exists()
            else {"version": CACHE_VERSION, "calculations": {}}
        )
        if self.data.get("version") != CACHE_VERSION:
            raise ValueError("Unsupported AIMNet cache version")
        self.identity = None
        self._entries = None
        self.begin_assessment()

    def begin_assessment(self):
        """Failures are retried on another assessment; successful entries persist."""
        self.seen_species: set[str] = set()
        self.failures: dict[str, MolecularCalculationError] = {}

    def get(self, smiles: str, calculator: MolecularCalculator, stats: dict) -> dict:
        if self._entries is None:
            self.identity = {
                "assessment_version": ASSESSMENT_VERSION,
                "config": self.config.molecular_settings(),
                "calculator": calculator.identity(),
            }
            key = _fingerprint(self.identity)
            if self.disk_backed:
                self._entries = MolecularRegistry(self.cache_path, key, self.identity)
            else:
                self._entries = self.data["calculations"].setdefault(
                    key, {"identity": self.identity, "molecules": {}}
                )["molecules"]
        self.seen_species.add(smiles)
        cached = self._entries.get(smiles)
        if cached is not None:
            stale_proton = (
                is_proton(smiles)
                and cached.get("electronic_energy_reference") != PROTON_REFERENCE
            )
            if stale_proton or symmetry_error(smiles, cached, self.config):
                del self._entries[smiles]
                stats["cache_invalidations"] += 1
            else:
                stats["cache_hits"] += 1
                return cached
        if smiles in self.failures:
            raise self.failures[smiles]
        stats["molecular_calculations"] += 1
        started = perf_counter()
        try:
            result = calculator.calculate(smiles)
            if problem := symmetry_error(smiles, result, self.config):
                raise ValueError(problem)
            if not all(
                isfinite(result[k]) for k in ("energy_eV", "enthalpy_eV", "gibbs_eV")
            ):
                raise ValueError("Molecular calculator returned nonfinite E/H/G")
        except MolecularCalculationError as exc:
            self.failures[smiles] = exc
            stats["molecular_failures"] += 1
            raise
        finally:
            stats["calculation_seconds"] += perf_counter() - started
        write_started = perf_counter()
        self._entries[smiles] = deepcopy(result)
        if self.cache_path is not None and not self.disk_backed:
            write_json_atomic(self.cache_path, self.data)
        stats["molecular_write_seconds"] = (
            stats.get("molecular_write_seconds", 0.0) + perf_counter() - write_started
        )
        return result
