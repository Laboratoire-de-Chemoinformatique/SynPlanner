"""Torch-free Query-CGR Morgan fingerprint adapter for reaction-rule graphs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from chython.algorithms.fingerprints import MorganFingerprint
from chython.containers import QueryCGRContainer, ReactionContainer

from synplan.chem.reaction.rules.representation.query_cgr import (
    query_cgr_atom_label,
    query_cgr_bond_label,
)

_QUERY_ATOM_LABEL_FIELDS = (
    "atomic_number",
    "atomic_symbol",
    "isotope",
    "charge",
    "is_radical",
    "neighbors",
    "heteroatoms",
    "hybridization",
    "implicit_hydrogens",
    "ring_sizes",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if isinstance(value, set | frozenset):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return repr(value)


def _stable_hash(label: Any) -> int:
    payload = json.dumps(
        _jsonable(label),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _safe_attr(obj: object, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _query_atom_label(atom: object) -> tuple:
    return tuple((field, _safe_attr(atom, field)) for field in _QUERY_ATOM_LABEL_FIELDS)


def query_reaction_atom_labels(reaction_query: ReactionContainer) -> dict[int, tuple]:
    """Return side-aware query atom labels not preserved by QueryCGR composition."""
    labels: dict[int, list[tuple[str, tuple]]] = {}
    for side_name, molecules in (
        ("reactants", reaction_query.reactants),
        ("reagents", reaction_query.reagents),
        ("products", reaction_query.products),
    ):
        for molecule in molecules:
            for atom_number, atom in molecule._atoms.items():
                labels.setdefault(atom_number, []).append(
                    (side_name, _query_atom_label(atom))
                )
    return {
        atom_number: tuple(atom_labels) for atom_number, atom_labels in labels.items()
    }


@dataclass(frozen=True)
class _FingerprintBond:
    label: int

    def __int__(self) -> int:
        return self.label


class QueryCGRMorganFingerprintAdapter(MorganFingerprint):
    """Expose Chython QueryCGRContainer through the MorganFingerprint protocol.

    Chython query-CGRs do not expose a public Morgan fingerprint API, so this
    adapter reads the stable private graph slots (``_atoms`` and ``_bonds``) and
    uses the same private label helpers as SynPlanner's canonical QueryCGR key.
    """

    def __init__(
        self,
        query_cgr: QueryCGRContainer,
        *,
        atom_labels: Mapping[int, Any] | None = None,
    ):
        self._query_cgr = query_cgr
        self._atom_labels = dict(atom_labels or {})
        self._bonds = {
            atom: {
                neighbor: _FingerprintBond(
                    _stable_hash(query_cgr_bond_label(query_cgr, atom, neighbor))
                )
                for neighbor in neighbors
            }
            for atom, neighbors in query_cgr._bonds.items()
        }

    @property
    def _atom_identifiers(self) -> dict[int, int]:
        return {
            atom: _stable_hash(
                (
                    query_cgr_atom_label(self._query_cgr, atom),
                    self._atom_labels.get(atom),
                )
            )
            for atom in self._query_cgr._atoms
        }


def query_cgr_morgan_fingerprint(
    query_cgr: QueryCGRContainer,
    *,
    atom_labels: Mapping[int, Any] | None = None,
    min_radius: int = 1,
    max_radius: int = 4,
    length: int = 1024,
    number_active_bits: int = 2,
):
    """Calculate a Morgan fingerprint for a Chython QueryCGRContainer."""
    return QueryCGRMorganFingerprintAdapter(
        query_cgr,
        atom_labels=atom_labels,
    ).morgan_fingerprint(
        min_radius=min_radius,
        max_radius=max_radius,
        length=length,
        number_active_bits=number_active_bits,
    )


__all__ = [
    "QueryCGRMorganFingerprintAdapter",
    "query_cgr_morgan_fingerprint",
    "query_reaction_atom_labels",
]
