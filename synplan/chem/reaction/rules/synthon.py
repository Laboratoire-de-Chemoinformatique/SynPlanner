"""The shipped synthon disconnections as an MCTS priority-rule set."""

from __future__ import annotations

import itertools
import re

from chython import smarts

from synplan.chem.reaction import CanonicalRetroReactor
from synplan.chem.reaction.rules import parse_priority_rules, rule_query_pattern
from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.fragment import _select
from synplan.chem.synthon.transformer import RULE_NUCLEOPHILE_CAPS

SYNTHON_SOURCE_NAME = "synthon"
"""``rule_source`` these rules carry into ``tree.stats.per_priority_source``."""

# `[#6;+0_elec2:1]` — body, token, tail. A trailing `2` marks a double-bond disconnection.
_LABELLED = re.compile(
    r"\[([^\]]*?)_(elec2|nuc2|neut2|elecB|elec\*|nuc\*|elec|nuc)((?:;[^\]:]*)?:(\d+))\]"
)

# Every cap atom needs its own map number: chython's patcher silently drops the whole match on an
# unmapped one, or re-identifies it with an existing atom, which is worse.
_CAP_SMARTS = {
    "Cl": "[Cl:{0}]",
    "Br": "[Br:{0}]",
    "O": "[O:{0}]",
    "C": "[C:{0}]",
    "B(O)O": "[B:{0}]([O:{1}])[O:{2}]",
    "[Mg]Br": "[Mg:{0}][Br:{1}]",
    "OC(=O)c1ccccc1": (
        "[O:{0}][C:{1}](=[O:{2}])[c:{3}]1[c:{4}][c:{5}][c:{6}][c:{7}][c:{8}]1"
    ),
}

# `rules.json` writes a carbon nucleophile as bare `[Mg]`, which is shorthand rather than a
# reagent; capping with it fails chython's valence check, and falling back to H proposes an
# alkane that cannot do the reaction but IS purchasable, so the route terminates on a lie.
_UNSHIPPABLE_CAPS = frozenset(("[Mg]", "[B-](F)(F)F"))


def _aromatic(hybridization) -> bool | None:
    """Aromaticity of a query atom, or None when its spec leaves it open."""
    if hybridization == (4,):
        return True
    if hybridization and 4 not in hybridization:
        return False
    return None


def capped_smarts(
    rule_smarts: str, leaving_groups: dict[str, str], rule_id: str | None = None
) -> str:
    """Spell each label's leaving group inline on the RHS, so a product is a reagent.

    Without this the labels are inert on the :class:`CanonicalRetroReactor` path: the patcher
    builds plain atoms and fills the broken valence with H, so an amide disconnects to an
    aldehyde rather than to an acyl chloride.

    :param rule_smarts: One ``rules.json`` disconnection SMARTS.
    :param leaving_groups: The ``rules.json`` ``leaving_groups`` table.
    :param rule_id: The record's id, consulted against :data:`RULE_NUCLEOPHILE_CAPS` where the
        label-keyed table cannot name the reagent.
    :return: The same SMARTS with every labelled RHS atom carrying its capped leaving group.
    :raises ValueError: when a labelled RHS atom keeps its ``_token`` through capping. The SMARTS
        writer drops the token, so such an atom reaches the reactor as a plain atom with no
        leaving group and no label — a wrong-but-purchasable fragment, with no error.
    """
    lhs_text, rhs_text = rule_smarts.split(">>", 1)
    lhs = dict(smarts(lhs_text.strip()).atoms())
    caps: dict[int, str] = {}
    tokens: dict[int, str] = {}
    for number, atom in smarts(rhs_text.strip()).atoms():
        token = getattr(atom, "_label", None)
        if token is None:
            continue
        # ponytail: aromaticity the specs leave open defaults to aliphatic (ArCl, not ArBr); worth
        # 3 firings in 45 measured, upgrade path is one reactor per aromaticity.
        aromatic = _aromatic(atom.hybridization)
        if aromatic is None and number in lhs:
            aromatic = _aromatic(lhs[number].hybridization)
        symbol = atom.atomic_symbol
        cap = leaving_groups.get(
            f"{symbol.lower() if aromatic else symbol}:{token}", "H"
        )
        if cap in _UNSHIPPABLE_CAPS:
            cap = RULE_NUCLEOPHILE_CAPS.get(rule_id or "", "H")
        caps[number] = cap
        tokens[number] = token

    next_map = itertools.count(90)
    capped: set[int] = set()

    def replace(match: re.Match) -> str:
        number = int(match.group(4))
        capped.add(number)
        atom = f"[{match.group(1)}{match.group(3)}]"
        cap = caps[number]
        if cap == "H":
            return atom
        template = _CAP_SMARTS[cap]
        maps = [next(next_map) for _ in range(template.count("{"))]
        bond = "=" if match.group(2).endswith("2") else "-"
        return atom + bond + template.format(*maps)

    rhs = _LABELLED.sub(replace, rhs_text)
    # the parser and `_LABELLED` must agree on what a token looks like. When they disagree — a
    # ninth token, a spec the regex cannot spell — the atom is capped with nothing and reaches the
    # reactor as a bare atom, and the whole failure is one missing substring in a SMARTS string.
    if missed := sorted(set(caps) - capped):
        raise ValueError(
            f"synthon rule {rule_id or '?'}: labelled RHS atoms {missed} kept their tokens "
            f"({', '.join(sorted({tokens[n] for n in missed}))}) through capping; the SMARTS "
            f"writer drops the token, so the leaving group would vanish silently"
        )
    return f"{lhs_text}>>{rhs}"


def _rule_smarts(record: dict, leaving_groups: dict[str, str], cap: bool) -> str:
    """The SMARTS a record reaches the reactor as.

    A heterocyclisation cuts TWO bonds, so its fragments have no open valence for
    :func:`capped_smarts` to spell a leaving group on and no per-atom cap can express the reagent
    anyway — a triazole comes from an azide and an ALKYNE, not from a triazene and a styrene. Such
    a record ships a hand-authored reagent-form ``retro_smarts`` and bypasses capping entirely.
    """
    if record["ring"]:
        return record["retro_smarts"]
    if cap:
        return capped_smarts(record["smarts"], leaving_groups, record["id"])
    return record["smarts"]


def _records(config: SynthonConfig | None, macro: bool) -> list[dict]:
    """The selected ``rules.json`` disconnection records, macrocyclic half optional."""
    config = config or SynthonConfig()
    data = load_data(config.rules_path)
    # a heterocyclisation without its hand-authored `retro_smarts` is excluded: capping cannot
    # spell its reagents, so it would reach the reactor uncapped and propose a purchasable
    # compound of the wrong class — a styrene where the alkyne belongs
    records = _select(
        [
            r
            for r in data["disconnections"]
            if not r["macro"] and (not r["ring"] or r["retro_smarts"])
        ],
        config.rule_mode,
        config.rules_selection,
    )
    if macro:
        records = records + _select(
            [r for r in data["disconnections"] if r["macro"]],
            config.rule_mode,
            re.sub(r"M?R", "MR", config.rules_selection),
        )
    return records


def synthon_priority_rules(
    config: SynthonConfig | None = None,
    *,
    cap: bool = True,
    macro: bool = False,
) -> dict[str, list[CanonicalRetroReactor]]:
    """Load ``rules.json`` disconnections as ``Tree(priority_rules=...)`` input.

    :param config: Supplies the data path and honours ``rule_mode`` / ``rules_selection``.
    :param cap: Spell leaving groups on the RHS. Off reproduces the H-capped baseline, where
        every label is inert and a fifth of the resulting fragments are purchasable compounds
        of the wrong class. On, R7.1/R10.1/R10.2 are still wrong — their leaving group is a
        property of the rule, not of the labelled atom, and the shipped table is keyed by atom.
        Ignored by the ring rules, which ship their reagent form ready-written.
    :param macro: Also load the macrocyclic ``MR*`` half. They only match ring bonds outside
        r3-r11, so they are dead weight on a non-macrocyclic target.
    :return: ``{SYNTHON_SOURCE_NAME: [rule, ...]}``.
    """
    records = _records(config, macro)
    leaving_groups = load_data((config or SynthonConfig()).rules_path)["leaving_groups"]
    parsed = parse_priority_rules(
        {
            SYNTHON_SOURCE_NAME: [
                _rule_smarts(record, leaving_groups, cap) for record in records
            ]
        },
        automorphism_filter=True,
    )
    # A rule whose query pattern is None is gated off by PriorityPolicy forever, without an error.
    ungated = [
        record["id"]
        for record, rule in zip(records, parsed[SYNTHON_SOURCE_NAME])
        if rule_query_pattern(rule) is None
    ]
    if ungated:
        raise ValueError(
            f"synthon rules {ungated} carry no query pattern; PriorityPolicy would "
            f"silently never fire them"
        )
    # the tree records only (rule_source, rule_id), so the human-readable name has to travel on
    # the rule object for a route report to name the reaction it applied
    for record, rule in zip(records, parsed[SYNTHON_SOURCE_NAME]):
        rule.rule_id = record["id"]
        rule.rule_name = record["name"]
    return parsed


__all__ = [
    "SYNTHON_SOURCE_NAME",
    "capped_smarts",
    "synthon_priority_rules",
]
