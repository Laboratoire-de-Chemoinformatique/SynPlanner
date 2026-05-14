"""Pin the contract of the atom-mapping validators in :mod:`synplan.chem.utils`.

Rule extraction, CGR composition and reactor application all need shared atom
numbers between reactants and products to identify the reaction centre. Chython
silently auto-numbers bare atoms with a per-side counter, so an unmapped
reaction parses successfully but produces a degenerate CGR. These tests pin
the three-state classifier and the assert helper so a future refactor cannot
regress to "passes through unchanged on garbage input".

Two entry points are tested:

* :func:`reaction_mapping_status` — container-based. Reliable on SMILES
  (``_parsed_mapping`` is preserved) but limited on SMARTS-parsed rules
  (chython drops the attribute and bare atoms can collide on auto-counter
  values, hiding "fully unmapped" behind a coincidental intersection).
* :func:`reaction_string_mapping_status` — string-based. Uses the chython
  tokenizer to read each atom's original ``parsed_mapping`` token before
  parsing strips it. The reliable path for SMARTS rules (e.g. RDKit /
  RDChiral output).

The combined :func:`is_reaction_atom_mapped` / :func:`assert_reaction_atom_mapped`
dispatch on input type — strings go through the tokenizer route, containers
through the parsed route.
"""

from __future__ import annotations

import warnings

import pytest
from chython import smarts as read_smarts
from chython import smiles as read_smiles
from chython.exceptions import MappingError

from synplan.chem.utils import (
    assert_reaction_atom_mapped,
    is_reaction_atom_mapped,
    reaction_mapping_status,
    reaction_string_mapping_status,
)

# ---- Container API on SMILES-parsed reactions ------------------------------


def test_status_smiles_fully_mapped() -> None:
    rxn = read_smiles("[CH3:1][CH2:2][Br:3].[OH2:4]>>[CH3:1][CH2:2][OH:4].[BrH:3]")
    assert reaction_mapping_status(rxn) == "fully_mapped"
    assert is_reaction_atom_mapped(rxn) is True
    assert_reaction_atom_mapped(rxn)
    assert_reaction_atom_mapped(rxn, allow_partial=False)


def test_status_smiles_fully_unmapped() -> None:
    rxn = read_smiles("CCBr.O>>CCO.Br")
    assert reaction_mapping_status(rxn) == "unmapped"
    assert is_reaction_atom_mapped(rxn) is False
    with pytest.raises(MappingError):
        assert_reaction_atom_mapped(rxn)
    with pytest.raises(MappingError):
        assert_reaction_atom_mapped(rxn, allow_partial=False)


def test_status_smiles_partially_mapped_warns_by_default() -> None:
    """``[CH3:1]`` is shared, but ``Br`` / ``O`` are bare — chython
    auto-numbers them, and the auto-numbers do not overlap across sides,
    so leaving/incoming group identification by chython's compose is
    unreliable."""
    rxn = read_smiles("[CH3:1]Br.O>>[CH3:1]O.Br")
    assert reaction_mapping_status(rxn) == "partially_mapped"
    assert is_reaction_atom_mapped(rxn) is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_reaction_atom_mapped(rxn)
    assert any(
        issubclass(w.category, UserWarning)
        and "partially atom-mapped" in str(w.message)
        for w in caught
    ), f"expected partial-mapping UserWarning, got {[str(w.message) for w in caught]}"


def test_status_smiles_partially_mapped_strict_raises() -> None:
    rxn = read_smiles("[CH3:1]Br.O>>[CH3:1]O.Br")
    with pytest.raises(MappingError, match="partially atom-mapped"):
        assert_reaction_atom_mapped(rxn, allow_partial=False)


def test_status_smiles_non_overlapping_explicit_maps_is_unmapped() -> None:
    """All atoms have explicit maps but reactant and product map sets are
    disjoint — the maps don't relate any reactant atom to any product
    atom, so the reaction is effectively unmapped."""
    rxn = read_smiles("[CH3:1][CH3:2]>>[CH3:10][CH3:11]")
    assert reaction_mapping_status(rxn) == "unmapped"
    with pytest.raises(MappingError):
        assert_reaction_atom_mapped(rxn)


# ---- Container API on SMARTS-parsed rules (limited; pinned for awareness) --
#
# chython.smarts drops ``_parsed_mapping`` when building the QueryElement and
# auto-numbers bare atoms per-side. Container-based partial detection on
# SMARTS rules is therefore not possible — only fully-unmapped detection
# *might* work, but even that is fragile if the per-side counters collide.
# Callers needing reliable detection on SMARTS rules should use the
# string-based helpers below.


def test_status_smarts_rule_fully_mapped_via_container() -> None:
    rule = read_smarts("[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]")
    assert reaction_mapping_status(rule) == "fully_mapped"
    assert is_reaction_atom_mapped(rule) is True


def test_status_smarts_rule_partial_via_container_is_indistinguishable() -> None:
    """Known limitation pinned: chython drops the SMARTS ``parsed_mapping``
    trace, so the container API cannot tell partial from fully mapped.
    The string-based API below is the correct entry point."""
    rule = read_smarts("[C;D3:1]-[O;D1]>>[C;D3:1]=[O;D1]")
    assert reaction_mapping_status(rule) == "fully_mapped"  # FALSE NEGATIVE


# ---- String API — reliable on both SMILES and SMARTS ----------------------


def test_string_status_smiles_fully_mapped() -> None:
    text = "[CH3:1][CH2:2][Br:3].[OH2:4]>>[CH3:1][CH2:2][OH:4].[BrH:3]"
    assert reaction_string_mapping_status(text) == "fully_mapped"
    assert is_reaction_atom_mapped(text) is True
    assert_reaction_atom_mapped(text, allow_partial=False)


def test_string_status_smiles_unmapped() -> None:
    text = "CCBr.O>>CCO.Br"
    assert reaction_string_mapping_status(text) == "unmapped"
    assert is_reaction_atom_mapped(text) is False
    with pytest.raises(MappingError):
        assert_reaction_atom_mapped(text)


def test_string_status_smiles_partial() -> None:
    text = "[CH3:1]Br.O>>[CH3:1]O.Br"
    assert reaction_string_mapping_status(text) == "partially_mapped"
    assert is_reaction_atom_mapped(text) is False
    with pytest.raises(MappingError, match="partially atom-mapped"):
        assert_reaction_atom_mapped(text, allow_partial=False)


def test_string_status_smarts_rule_fully_mapped() -> None:
    text = "[C;D3:1]-[O;D1:2]>>[C;D3:1]-[O;D2:2]-[C;D1:3]"
    assert reaction_string_mapping_status(text) == "fully_mapped"
    assert is_reaction_atom_mapped(text) is True


def test_string_status_smarts_rule_partial() -> None:
    """The container API can't see this (parsed_mapping is dropped). The
    string API reads the tokenizer output directly and catches it."""
    text = "[C;D3:1]-[O;D1]>>[C;D3:1]=[O;D1]"
    assert reaction_string_mapping_status(text) == "partially_mapped"
    assert is_reaction_atom_mapped(text) is False
    with pytest.raises(MappingError, match="partially atom-mapped"):
        assert_reaction_atom_mapped(text, allow_partial=False)


def test_string_status_smarts_rule_fully_unmapped() -> None:
    """The container API also gets this wrong — both sides parse to
    ``{1, 2}`` so the intersection check coincidentally reports
    fully_mapped. The string API checks the original tokens."""
    text = "[C]-[O]>>[C]=[O]"
    assert reaction_string_mapping_status(text) == "unmapped"
    with pytest.raises(MappingError):
        assert_reaction_atom_mapped(text)


def test_string_status_with_reagents() -> None:
    """``reactants>reagents>products`` form is accepted; reagents are
    spectators and don't need their own mapping."""
    text = "[CH3:1][OH:2]>[Na+]>[CH3:1][OH:2]"
    assert reaction_string_mapping_status(text) == "fully_mapped"


def test_string_status_malformed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="malformed reaction string"):
        reaction_string_mapping_status("not a reaction")
    with pytest.raises(ValueError, match="malformed reaction string"):
        reaction_string_mapping_status("a>b>c>d")
