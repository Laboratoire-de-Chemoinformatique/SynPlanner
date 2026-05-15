"""Helpers for loading priority-rule SMARTS into chython :class:`Reactor` objects.

The priority-rule pipeline expects ``dict[str, list[Reactor]]`` (set name →
rules). Hand-written SMARTS authored against RDKit conventions sometimes fail
chython's parser silently or with terse errors. :func:`parse_priority_rules`
wraps ``Reactor.from_smarts`` per SMARTS, attaches the failing string to the
exception, and (if RDKit is installed) reports whether RDKit can read the
SMARTS — that distinguishes "this SMARTS is broken" from "this SMARTS uses a
dialect chython doesn't accept".
"""

from __future__ import annotations

from collections.abc import Iterable

from chython.reactor import Reactor

from synplan.chem.reaction import CanonicalRetroReactor

POLICY_SOURCE_NAME: str = "policy"
"""Reserved ``rule_source`` label for the learned-policy bucket; cannot be reused as a priority set name."""


class PrioritySmartsError(ValueError):
    """Raised when a priority-rule SMARTS cannot be parsed by chython."""


def _check_rdkit_parseability(smarts: str) -> str:
    """Return a one-line diagnostic about whether RDKit can parse this SMARTS.

    Used only to enrich :exc:`PrioritySmartsError` messages — never raises.
    """
    try:
        from rdkit.Chem import (
            rdChemReactions,
        )
    except ImportError:
        return "RDKit not installed; cannot cross-check SMARTS dialect"
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
    except Exception:
        return "RDKit also rejects this SMARTS — likely malformed"
    if rxn is None or rxn.GetNumReactantTemplates() == 0:
        return "RDKit also cannot parse this SMARTS — likely malformed"
    return (
        "RDKit parses this SMARTS successfully — likely a chython/RDKit "
        "dialect mismatch (e.g. aromatic kekulization, valence, charge)"
    )


def parse_priority_rules(
    smarts_by_set: dict[str, Iterable[str]],
    *,
    automorphism_filter: bool = False,
    delete_atoms: bool = False,
) -> dict[str, list[Reactor]]:
    """Parse a ``{set_name: [SMARTS, ...]}`` mapping into chython :class:`Reactor`s.

    On a parsing failure, raises :exc:`PrioritySmartsError` naming the offending
    set, the offending SMARTS, the underlying chython error, and (when RDKit is
    available) whether RDKit accepts the same SMARTS — so the caller can tell
    "broken SMARTS" apart from "RDKit-flavoured SMARTS that chython rejects".

    :param smarts_by_set: Mapping of priority-set name to an iterable of
        SMARTS strings.
    :param automorphism_filter: Forwarded to :meth:`Reactor.from_smarts`.
    :param delete_atoms: Forwarded to :meth:`Reactor.from_smarts`.
    :return: ``{set_name: [Reactor, ...]}`` ready to pass as
        ``Tree(priority_rules=...)``.
    """
    result: dict[str, list[Reactor]] = {}
    for set_name, smarts_list in smarts_by_set.items():
        if not isinstance(set_name, str) or not set_name:
            raise ValueError(
                f"priority-rule set names must be non-empty strings, got {set_name!r}"
            )
        if set_name == POLICY_SOURCE_NAME:
            raise ValueError(
                f"priority-rule set name {set_name!r} collides with the reserved "
                f"policy source name. Rename your priority set."
            )
        rules: list[Reactor] = []
        for index, smarts in enumerate(smarts_list):
            try:
                rules.append(
                    CanonicalRetroReactor.from_smarts(
                        smarts,
                        automorphism_filter=automorphism_filter,
                        delete_atoms=delete_atoms,
                    )
                )
            except Exception as err:  # chython exceptions are diverse
                hint = _check_rdkit_parseability(smarts)
                raise PrioritySmartsError(
                    f"Failed to parse priority_rules[{set_name!r}][{index}]:\n"
                    f"  SMARTS: {smarts}\n"
                    f"  chython error: {type(err).__name__}: {err}\n"
                    f"  diagnostic: {hint}"
                ) from err
        if not rules:
            raise ValueError(
                f"priority_rules[{set_name!r}] is empty. Either populate the "
                f"set or remove the key."
            )
        result[set_name] = rules
    return result
