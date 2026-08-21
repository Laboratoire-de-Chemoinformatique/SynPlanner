"""Is a mapped reaction already covered by one of the shipped synthon disconnections?

Used to strip such reactions out of a training corpus: a one-step policy learns nothing by
rediscovering chemistry the curated rules already provide.

The atom mapping says which bond the reaction FORMS; each rule says which bond it BREAKS (LHS
bonds minus RHS bonds). Covered = the rule's LHS matches the product with its broken bond sitting
exactly on the formed bond, AND the reactant-side leaving groups agree with the rule's ``_label``
tokens.

The labels are read off the RAW ``rules.json`` SMARTS. Capped SMARTS carry none, and chython's
``QueryElement.__eq__`` never consults ``_label``, so no substructure match can be label-aware —
:func:`load_coverage_rules` refuses a rule set that has lost its tokens, because a label-blind
matcher fails by over-covering, silently.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from chython import smarts
from chython.containers import MoleculeContainer, ReactionContainer

from synplan.synthon.config import SynthonConfig, load_data
from synplan.synthon.reactor import (
    RULE_NUCLEOPHILE_CAPS,
    SynthonRuleError,
    query_labels,
)

# Atoms that count as a leaving group departing from an ELECTROPHILE. S is deliberately out:
# Julia-Kocienski loses a sulfonyl from the NUCLEOPHILIC carbon, so S in the departing set must
# not brand an atom electrophilic. P likewise (Wittig ylide).
ELECTROPHUGE = frozenset(("F", "Cl", "Br", "I", "O", "N"))

# ponytail: a symmetric query on a symmetric product can enumerate many mappings; we only need
# one that lands on the formed bond. Cap the walk, upgrade path is a seeded VF2 if it ever bites.
_MAPPING_CAP = 500

_ELEMENT = re.compile(r"[A-Z][a-z]?")

# The reagent a cap spells, widened to the family that does the same chemistry: `[Mg]Br` stands
# for the Li/Zn/Sn/Cu organometallics too, `B(O)O` for boron in any of its coupling disguises.
_NUCLEOPHILE_FAMILY = {
    "Mg": frozenset(("Mg", "Li", "Zn", "Sn", "Cu")),
    "B": frozenset(("B",)),
}

# `nuc` alone is a NEGATIVE constraint — "no halide left this atom" — which every unfunctionalised
# carbon satisfies by doing nothing, so it waves Friedel-Crafts and enolate acylations through as
# organometallic chemistry. The rules that NAME their nucleophile are exactly the ones
# `RULE_NUCLEOPHILE_CAPS` has to spell a reagent for; for those, demand the element is there.
_RULE_NUCLEOPHILE_ELEMENTS: dict[str, frozenset[str]] = {
    rule_id: family
    for rule_id, cap in RULE_NUCLEOPHILE_CAPS.items()
    for symbol, family in _NUCLEOPHILE_FAMILY.items()
    if symbol in set(_ELEMENT.findall(cap))
}


class Coverage(NamedTuple):
    """Verdict for one reaction."""

    covered: bool
    rule_ids: tuple[str, ...]
    evidence: str


class CoverageRule:
    """One ``rules.json`` disconnection reduced to what the classifier needs."""

    __slots__ = ("broken", "elements", "id", "labels", "name", "query")

    def __init__(self, record: dict) -> None:
        lhs_text, rhs_text = record["smarts"].split(">>", 1)
        lhs, rhs = smarts(lhs_text.strip()), smarts(rhs_text.strip())
        lhs_bonds = {frozenset((a, b)) for a, b, _ in lhs.bonds()}
        rhs_bonds = {frozenset((a, b)) for a, b, _ in rhs.bonds()}
        broken = lhs_bonds - rhs_bonds
        if len(broken) != 1:
            raise SynthonRuleError(
                f"{record['id']} breaks {len(broken)} bonds, expected 1"
            )
        self.id: str = record["id"]
        self.name: str = record["name"]
        self.query = lhs
        self.broken: tuple[int, ...] = tuple(sorted(broken.pop()))
        # the labels live ONLY on the RHS; the LHS query we match against carries none
        self.labels: dict[int, str] = query_labels(rhs)
        if not self.labels:
            raise SynthonRuleError(
                f"{record['id']} carries no synthon label: the matcher cannot tell an "
                "electrophile from a nucleophile and would over-cover silently"
            )
        # aromatic c -> "C"
        symbols = {n: atom.atomic_symbol for n, atom in lhs.atoms()}
        self.elements: frozenset[str] = frozenset(symbols[n] for n in self.broken)


def load_coverage_rules(config: SynthonConfig | None = None) -> list[CoverageRule]:
    """The shipped acyclic, non-macrocyclic disconnections.

    The R16 heterocyclisations are out: they break TWO bonds, and coverage is built on
    "one formed bond, one broken bond".
    """
    data = load_data((config or SynthonConfig()).rules_path)
    return [
        CoverageRule(r)
        for r in data["disconnections"]
        if not r["macro"] and not r.get("ring")
    ]


def _token_ok(token: str, departing: set[str], pi_reduced: bool, rule_id: str) -> bool:
    """Does the observed reactant-side leaving group agree with this ``_label`` token?"""
    if token in ("elec", "elec2"):
        # a leaving group left this atom, or a C=X it was attached to got reduced (R10.1/R10.2)
        return bool(departing & ELECTROPHUGE) or pi_reduced
    if token in ("nuc", "nuc2", "nuc*"):
        if required := _RULE_NUCLEOPHILE_ELEMENTS.get(rule_id):
            return bool(departing & required)
        # otherwise: nucleophiles lose H, a metal, boron, PPh3, a sulfonyl — never a halide
        return not departing & ELECTROPHUGE
    if token == "elecB":
        return "B" in departing
    if token == "elec*":
        return not departing  # C-H functionalisation: nothing heavy leaves
    if token == "neut2":
        # metathesis sheds an alkylidene carbon
        return "C" in departing or not departing
    raise SynthonRuleError(f"unknown synthon label token {token!r}")


def _departing(
    molecule: MoleculeContainer, number: int, product_numbers: set[int]
) -> tuple[set[str], bool]:
    """Heavy neighbours of *number* that never reach the product, and whether a pi bond dropped."""
    departing: set[str] = set()
    pi_reduced = False
    neighbours = molecule._bonds[number]  # no public accessor in chython
    for neighbour, bond in neighbours.items():
        if neighbour not in product_numbers:
            departing.add(molecule.atom(neighbour).atomic_symbol)
        elif bond.order > 1 and molecule.atom(neighbour).atomic_symbol != "C":
            pi_reduced = True
    return departing, pi_reduced


def _split_sides(
    reaction: ReactionContainer,
) -> tuple[MoleculeContainer, dict[int, MoleculeContainer]]:
    """Main product, plus every left-hand atom number (reactants AND agents) to its molecule."""
    product = max(reaction.products, key=len)
    left: dict[int, MoleculeContainer] = {}
    for molecule in (*reaction.reactants, *reaction.reagents):
        for number, _ in molecule.atoms():
            left.setdefault(number, molecule)
    return product, left


def _formed_bonds(
    product: MoleculeContainer, left: dict[int, MoleculeContainer]
) -> set[frozenset[int]]:
    """Product bonds between two mapped atoms that were not bonded on the left."""
    formed = set()
    for a, b, _ in product.bonds():
        if a not in left or b not in left:
            continue  # unmapped in the product: cannot tell, do not guess
        if left[a] is left[b] and b in left[a]._bonds[a]:
            continue
        formed.add(frozenset((a, b)))
    return formed


def classify_coverage(
    reaction: ReactionContainer,
    rules: list[CoverageRule] | None = None,
    *,
    check_labels: bool = True,
) -> Coverage:
    """Is this mapped reaction already covered by a shipped synthon disconnection?

    Coverage is DISCONNECTION-level, not mechanism-level: a reductive amination builds the bond
    R3.1 disconnects, so it counts as covered even though the rule is not that mechanism.

    :param reaction: A mapped :class:`ReactionContainer`; agents and spectators are welcome.
    :param rules: Output of :func:`load_coverage_rules`; load once, reuse across a corpus.
    :param check_labels: Off makes the matcher label-blind. Diagnostic only — it measures what
        the substructure+bond match alone would absorb.
    """
    rules = load_coverage_rules() if rules is None else rules
    if not reaction.products or not reaction.reactants:
        return Coverage(False, (), "no product side")

    product, left = _split_sides(reaction)
    formed = _formed_bonds(product, left)
    if len(formed) != 1:
        return Coverage(
            False,
            (),
            f"{len(formed)} bonds formed; every synthon rule breaks exactly 1",
        )
    bond = formed.pop()
    product_numbers = {n for n, _ in product.atoms()}
    elements = frozenset(product.atom(n).atomic_symbol for n in bond)

    # USPTO ships kekulised, hand-written records ship lowercase; neither matches [c:1] as parsed.
    product.kekule()
    product.thiele()

    matched: list[str] = []
    notes: list[str] = []
    for rule in rules:
        if rule.elements != elements:
            continue
        i, j = rule.broken
        # a symmetric query lands on the same bond in both orientations, and only one of them
        # can satisfy the labels (R12.1 is elec + elecB) — so try every placement, not the first.
        # automorphism_filter=True would drop the second orientation of a symmetric query
        # ([c:1]-[c:2]), and with it the only placement whose labels can be satisfied.
        placements = []
        for count, mapping in enumerate(
            rule.query.get_mapping(product, automorphism_filter=False)
        ):
            if frozenset((mapping[i], mapping[j])) == bond:
                placements.append(mapping)
            if count >= _MAPPING_CAP:
                break
        if not placements:
            continue
        if not check_labels:
            matched.append(rule.id)
            continue
        failures: list[str] = []
        for placed in placements:
            problems = []
            for number, token in rule.labels.items():
                atom = placed[number]
                if atom not in left:
                    problems.append(
                        f"{rule.id} label {token} rejected: atom unmapped on the left"
                    )
                    continue
                departing, pi_reduced = _departing(left[atom], atom, product_numbers)
                if not _token_ok(token, departing, pi_reduced, rule.id):
                    problems.append(
                        f"{rule.id} label {token} rejected: departing="
                        f"{sorted(departing) or 'nothing'}, pi_reduced={pi_reduced}"
                    )
            if not problems:
                failures = []
                break
            failures = problems
        if failures:
            notes.extend(failures)
        else:
            matched.append(rule.id)

    if matched:
        return Coverage(
            True,
            tuple(matched),
            f"formed bond {sorted(bond)} ({'-'.join(sorted(elements))}) is the "
            f"disconnection of {', '.join(matched)}; labels confirmed",
        )
    return Coverage(
        False,
        (),
        "; ".join(notes)
        or f"no synthon rule disconnects the formed {'-'.join(sorted(elements))} bond",
    )


__all__ = [
    "ELECTROPHUGE",
    "Coverage",
    "CoverageRule",
    "classify_coverage",
    "load_coverage_rules",
]
