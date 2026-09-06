"""Bounded correspondence using Chython atom, bond and stereo predicates.

Budgets count attempted assignments inside the matcher, including rejected
matches. Limiting only the number of returned products cannot do that. Chython
perception and canonicalization remain separate backend costs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from chython.algorithms.isomorphism import QueryIsomorphism
from chython.containers import QueryContainer

from synplan.chem.stereo import has_stereo


class MappingBudgetExceeded(RuntimeError):
    """The result is incomplete; this is never evidence of incompatibility."""


@dataclass
class MappingBudget:
    limit: int = 100_000
    used: int = 0

    def consume(self, amount=1):
        self.used += amount
        if self.used > self.limit:
            raise MappingBudgetExceeded(
                f"mapping work exceeded {self.limit} assignments/checks"
            )


_budget = ContextVar("synplan_mapping_budget", default=None)


@contextmanager
def mapping_budget(limit=100_000, *, budget=None):
    if limit < 1:
        raise ValueError("mapping budget must be positive")
    budget = budget or MappingBudget(limit)
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)


# The paired Chython patch owns native matcher limits. Keep the independent
# bounded fallback usable with the currently released 1.105 dependency.
try:
    from chython.algorithms.mapping_budget import (
        MappingBudget,
        MappingBudgetExceeded,
        mapping_budget,
    )
    from chython.algorithms.mapping_budget import (
        current_budget as _budget,
    )
except ImportError:
    _native_budget = False
else:
    _native_budget = True


@contextmanager
def backend_preparation():
    """Keep backend normalization cost separate from rule correspondence work."""
    token = _budget.set(None)
    try:
        yield
    finally:
        _budget.reset(token)


def bounded_mappings(
    query,
    molecule,
    *,
    automorphism_filter=False,
    searching_scope=None,
    budget=None,
    _cython=True,
):
    """Yield graph correspondences with an enforceable candidate work bound.

    Uses Chython's compiled traversal and equality predicates. Stereo is checked
    by Chython's QueryIsomorphism after this generator, or by the caller using
    mapped orientation. No atom-set deduplication precedes stereo validation.
    Recursive SMARTS use Chython's AND/OR/negation evaluator with bounded nested
    queries, sharing the same work budget as the enclosing correspondence.
    """
    budget = budget or _budget.get() or MappingBudget()
    if _native_budget:
        from chython.algorithms.isomorphism import Isomorphism

        with mapping_budget(budget=budget):
            compiled = isinstance(query, QueryContainer) and getattr(
                QueryIsomorphism, "_budgeted_cython", False
            )
            matcher = (
                QueryIsomorphism._get_mapping if compiled else Isomorphism._get_mapping
            )
            yield from matcher(
                query,
                molecule,
                automorphism_filter=automorphism_filter,
                searching_scope=searching_scope,
                **({"_cython": _cython} if compiled else {}),
            )
        return
    if isinstance(query, QueryContainer) and not isinstance(query, BoundedQuery):
        query = bounded_query(query)
    with mapping_budget(budget=budget):
        recursive = (
            query._precompute_recursive(molecule)
            if isinstance(query, BoundedQuery)
            else None
        ) or {}
    components, _ = query._compiled_query
    order = [entry for component in components for entry in component]
    if not order or len(query) > len(molecule):
        return
    if len(order) > 512:
        raise MappingBudgetExceeded(
            "query exceeds the bounded mapping depth of 512 atoms"
        )
    scope = set(molecule) if searching_scope is None else set(searching_scope)
    mapping, reverse, seen = {}, {}, set()

    def walk(depth):
        if depth == len(order):
            budget.consume(len(mapping))
            if automorphism_filter:
                atom_set = frozenset(reverse)
                if atom_set in seen:
                    return
                seen.add(atom_set)
            yield mapping.copy()
            return
        number, back, atom, _ = order[depth]
        candidates = (
            molecule._bonds[mapping[back]] if back in mapping else molecule._atoms
        )
        for target in candidates:
            budget.consume()
            if (
                target not in scope
                or target in reverse
                or atom != molecule.atom(target)
            ):
                continue
            if number in recursive and target not in recursive[number]:
                continue
            expected = {
                mapping[n]: b for n, b in query._bonds[number].items() if n in mapping
            }
            actual = {n: b for n, b in molecule._bonds[target].items() if n in reverse}
            budget.consume(len(expected) + len(actual))
            if expected.keys() != actual.keys() or any(
                b != actual[n] for n, b in expected.items()
            ):
                continue
            mapping[number] = target
            reverse[target] = number
            yield from walk(depth + 1)
            del mapping[number]
            del reverse[target]

    yield from walk(0)


class BoundedQuery(QueryContainer):
    """Retain Chython stereo matching with a bounded connectivity generator."""

    __slots__ = ()

    def get_mapping(
        self, other, *, automorphism_filter=True, searching_scope=None, _cython=True
    ):
        # Chython deduplicates atom sets before checking stereo. That may discard
        # the only valid orientation. Keep mappings until after stereo checks.
        if has_stereo(other) or has_stereo(self):
            automorphism_filter = False
        return super().get_mapping(
            other,
            automorphism_filter=automorphism_filter,
            searching_scope=searching_scope,
            _cython=_cython and getattr(QueryIsomorphism, "_budgeted_cython", False),
        )

    def _get_mapping(
        self, other, *, automorphism_filter=True, searching_scope=None, **kwargs
    ):
        return bounded_mappings(
            self,
            other,
            automorphism_filter=automorphism_filter,
            searching_scope=searching_scope,
            _cython=kwargs.get("_cython", True),
        )


def bounded_query(query, *, _depth=0):
    if _depth > 32:
        raise MappingBudgetExceeded("recursive SMARTS exceeds the nesting limit of 32")
    copy = query.copy()
    copy.__class__ = BoundedQuery
    for _, atom in copy.atoms():
        recursive = getattr(atom, "_recursive_smarts", None)
        if recursive:
            atom._recursive_smarts = tuple(
                (positive, bounded_query(subquery, _depth=_depth + 1), root, group)
                for positive, subquery, root, group in recursive
            )
    return copy
