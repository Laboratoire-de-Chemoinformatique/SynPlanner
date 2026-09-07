"""Bounded correspondence using Chython atom, bond and stereo predicates.

Budgets count attempted assignments inside the matcher, including rejected
matches. Limiting only the number of returned products cannot do that. Chython
perception and canonicalization remain separate backend costs.
"""

from contextlib import contextmanager

from chython.algorithms.isomorphism import Isomorphism, QueryIsomorphism
from chython.algorithms.mapping_budget import (
    MappingBudget,
    MappingBudgetExceeded,
    mapping_budget,
)
from chython.algorithms.mapping_budget import current_budget as _budget
from chython.containers import QueryContainer

from synplan.chem.stereo import has_stereo


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
    with mapping_budget(budget=budget):
        compiled = isinstance(query, QueryContainer)
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


class BoundedQuery(QueryContainer):
    """Retain Chython stereo matching with a bounded connectivity generator."""

    __slots__ = ()

    def get_mapping(
        self, other, *, automorphism_filter=True, searching_scope=None, _cython=True
    ):
        # Keep all orientations even for stereo-free queries against chiral
        # targets: the caller may assess inherited stereo after matching.
        if has_stereo(other) or has_stereo(self):
            automorphism_filter = False
        return super().get_mapping(
            other,
            automorphism_filter=automorphism_filter,
            searching_scope=searching_scope,
            _cython=_cython,
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
