"""The check a hand-authored ring ``retro_smarts`` must pass before it ships.

A ring rule cuts two bonds, so its ``smarts`` synthon form names the wrong reagent class and
``capped_smarts`` cannot repair it — a triazole disconnects to a triazene and a styrene, not to an
azide and an alkyne. The reagent form is written by hand instead, and the failure mode it invites
is silent: chython's patcher accepts an RHS atom carrying no map number, then hands back the
INTACT target plus a free fragment. That is plausible, purchasable and completely wrong, with no
error anywhere, so it has to be refused here.

Three classes of silent failure, two mechanical and one not. An unmapped RHS atom and a SMIRKS
that never breaks the ring are both refused outright. The third is only WARNED about: where the
target is an N-H azole whose tautomers are degenerate, chython canonicalises to one of them and a
map number can land on the neighbouring ring carbon, so the rule emits an alpha-bromo aldehyde
where phenacyl bromide was intended. Ring breaks, maps unique, products parse, ``ok`` is True and
the chemistry is wrong — on a warned rule, read the products, do not trust the verdict.

The same swap also arrives by automorphism, with no tautomer in sight: a rule can emit
benzaldoxime + acetonitrile where acetaldoxime + benzonitrile was meant. Both stable, both
stocked, ring broken, maps unique. The only thing that catches a swap whatever caused it is
checking the ANSWER, so a record carrying ``expected_reagents`` has its products compared
against them — per product set, because a rule that fires in two ring directions adds a whole
coherent set, whereas a swap puts the wrong partner inside the intended one.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from chython import smarts, smiles
from chython.containers import MoleculeContainer

from synplan.chem.reaction.reactor import apply_reaction_rule
from synplan.chem.reaction.rules import parse_priority_rules, rule_query_pattern

RING_RULES_PATH = Path(__file__).with_name("ring_rules.json")
STOCK_PATH = (
    Path(__file__).resolve().parents[4]
    / "synplan_data/building_blocks/emolecules-salt-ln/building_blocks.tsv"
)

_BRACKET_ATOM = re.compile(r"\[[^\]]*\]")
_ATOM_MAP = re.compile(r":(\d+)\]")

TAUTOMER_WARNING = "tautomer-degenerate N-H azole — verify product identity"


@dataclass(frozen=True)
class RetroCheck:
    """One ring record's verdict; ``ok`` is False for a broken rule and for an unauthored one."""

    rule_id: str
    ok: bool
    reason: str
    authored: bool = True
    products: list[list[str]] = field(default_factory=list)
    in_stock: frozenset[str] = frozenset()
    warnings: tuple[str, ...] = ()

    @property
    def all_products(self) -> set[str]:
        """Every distinct product the rule gave, across all of its product sets."""
        return {product for group in self.products for product in group}


@cache
def load_stock(path: str = str(STOCK_PATH)) -> frozenset[str]:
    """The building-block SMILES column, already canonical; empty when the file is absent."""
    file = Path(path)
    if not file.exists():
        return frozenset()
    with file.open(encoding="utf-8") as handle:
        next(handle, None)  # header
        return frozenset(line.split("\t", 1)[0] for line in handle if line.strip())


def _rhs_maps(rhs_text: str) -> list[int]:
    return [
        int(number)
        for atom in _BRACKET_ATOM.findall(rhs_text)
        for number in _ATOM_MAP.findall(atom)
    ]


def _canonical(smi: str) -> str:
    """One spelling for both sides of the identity comparison."""
    molecule = smiles(smi)
    molecule.canonicalize()
    molecule.clean_stereo()
    molecule.canonicalize()
    return str(molecule)


def _tautomer_ambiguous(target: MoleculeContainer) -> bool:
    """Whether the target has an aromatic ring with both an N-H and a proton-accepting N.

    The two tautomers are the same compound, so canonicalisation picks one and the rule's map
    numbers follow that choice rather than the author's intent.
    """
    for ring in target.sssr:
        nitrogens = [
            target.atom(n) for n in ring if target.atom(n).atomic_symbol == "N"
        ]
        if not all(atom.hybridization == 4 for atom in nitrogens):
            continue
        donors = [atom for atom in nitrogens if atom.implicit_hydrogens]
        acceptors = [
            atom
            for atom in nitrogens
            if not atom.implicit_hydrogens and atom.neighbors == 2
        ]
        if donors and acceptors:
            return True
    return False


def check_retro_rule(record: dict, stock: frozenset[str] = frozenset()) -> RetroCheck:
    """Verify one ``ring_rules.json`` record's reagent-form retro SMIRKS.

    :param record: A ring record; ``retro_smarts`` absent or empty means "not yet authored".
    :param stock: Canonical building-block SMILES from :func:`load_stock`. Reported, never fatal.
    :return: The verdict, with the products the rule gave on its own ``example_target``.
    """
    rule_id = record["id"]
    retro_smarts = record.get("retro_smarts")
    if not retro_smarts:
        return RetroCheck(rule_id, False, "not authored", authored=False)
    if ">>" not in retro_smarts:
        return RetroCheck(rule_id, False, "not a reaction SMARTS: no `>>`")

    rhs_text = retro_smarts.split(">>", 1)[1].strip()
    maps = _rhs_maps(rhs_text)
    if len(set(maps)) != len(maps):
        repeated = sorted({n for n in maps if maps.count(n) > 1})
        return RetroCheck(rule_id, False, f"duplicate RHS map numbers {repeated}")

    try:
        rule = parse_priority_rules(
            {"retro": [retro_smarts]}, automorphism_filter=True
        )["retro"][0]
    except Exception as err:
        return RetroCheck(
            rule_id, False, f"does not parse: {type(err).__name__}: {err}"
        )

    if len(maps) != sum(1 for _ in smarts(rhs_text).atoms()):
        return RetroCheck(rule_id, False, "an RHS atom carries no map number")

    pattern = rule_query_pattern(rule)
    if pattern is None:
        return RetroCheck(
            rule_id, False, "no query pattern; PriorityPolicy would gate it off forever"
        )

    target = smiles(record["example_target"])
    target.canonicalize()
    products = [list(group) for group in apply_reaction_rule(target, rule)]
    if not products:
        return RetroCheck(
            rule_id, False, f"does not fire on {record['example_target']}"
        )

    # ponytail: "the ring survives" is read off the rule's own LHS, so a target carrying two
    # copies of the ring reads as a failure; give such a rule a single-ring `example_target`.
    for group in products:
        for product in group:
            try:
                survived = pattern < product
            except TypeError:
                survived = False
            if survived:
                return RetroCheck(
                    rule_id, False, f"the ring survives in the product {product}"
                )

    spelled = [sorted(str(product) for product in group) for group in products]
    found = frozenset(p for group in spelled for p in group if p in stock)

    emitted = [{_canonical(p) for p in group} for group in spelled]
    expected = record.get("expected_reagents")
    warnings = []
    if _tautomer_ambiguous(target):
        warnings.append(TAUTOMER_WARNING)
    if not expected:
        warnings.append("no expected_reagents on the record — identity unchecked")
    elif (wanted := {_canonical(r) for r in expected}) not in emitted:
        # a union-shaped expectation is a recording error, not a swap: say which it is
        warnings.append(
            f"expected_reagents is the union of {len(emitted)} sets, not one — re-record per set"
            if wanted == set().union(*emitted)
            else f"IDENTITY MISMATCH: no emitted set equals {sorted(expected)}"
        )
    if len(emitted) > 1:
        warnings.append(f"{len(emitted)} product sets — only one was vetted")
    return RetroCheck(
        rule_id, True, "", products=spelled, in_stock=found, warnings=tuple(warnings)
    )


def check_ring_rules(
    records: list[dict], stock: frozenset[str] = frozenset()
) -> list[RetroCheck]:
    """:func:`check_retro_rule` over a whole ``ring_rules.json``."""
    return [check_retro_rule(record, stock) for record in records]


def _row(check: RetroCheck) -> str:
    products = check.all_products
    stock = f"{len(check.in_stock)}/{len(products)} in stock" if products else ""
    verdict = (
        "WARN"
        if check.ok and check.warnings
        else ("PASS" if check.ok else ("FAIL" if check.authored else "----"))
    )
    note = check.reason or "; ".join(check.warnings)
    sets = f"{len(check.products)}set" if check.products else ""
    return (
        f"{check.rule_id:<9} {verdict} {len(products):>2}p {sets:<5}{stock:<14} {note}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ring retro SMIRKS.")
    parser.add_argument("--rules", default=str(RING_RULES_PATH))
    parser.add_argument("--stock", default=str(STOCK_PATH))
    parser.add_argument(
        "--id", action="append", help="check only these ids; repeatable"
    )
    parser.add_argument(
        "--all", action="store_true", help="also list the not-yet-authored rules"
    )
    args = parser.parse_args(argv)

    with open(args.rules, encoding="utf-8") as handle:
        records = json.load(handle)
    if args.id:
        wanted = set(args.id)
        records = [record for record in records if record["id"] in wanted]
    checks = check_ring_rules(records, load_stock(args.stock))

    for check in checks:
        if check.authored or args.all:
            print(_row(check))
    authored = [check for check in checks if check.authored]
    failed = [check for check in authored if not check.ok]
    warned = [check for check in authored if check.ok and check.warnings]
    print(
        f"{len(authored)} authored: {len(authored) - len(failed)} pass ({len(warned)} to eyeball), "
        f"{len(failed)} fail; {len(checks) - len(authored)} not authored"
    )
    return 1 if failed else 0


__all__ = ["RetroCheck", "check_retro_rule", "check_ring_rules", "load_stock"]


if __name__ == "__main__":
    raise SystemExit(main())
