"""Offline converter: Synt-On's XML/JSON knowledge base -> the chython-dialect JSON that ships.

Run by hand; the output is committed and reviewed in the diff. Nothing translates at import time.
This is the only place Synt-On's integer reaction-centre codes exist — everything downstream is
the eight token strings.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from chython import smarts
from chython.periodictable import AnyElement

from synplan.chem.synthon.data._dialect import DialectError, to_chython

# the one-time migration. Code 11 ("electrophilic nitrogen") collapses into 'elec': marksCombinations
# has no N:10 key, so on nitrogen "electrophile" already has exactly one meaning.
PAPER_CODE_TO_LABEL = {
    10: "elec",
    11: "elec",
    20: "nuc",
    21: "elecB",
    30: "elec2",
    40: "nuc2",
    50: "neut2",
    60: "elec*",
    70: "nuc*",
}

# upstream __marksCombinations (SyntOn.py), as (element, aromatic, code) pairs. Read verbatim, then
# migrated. The relation is symmetric upstream and stays symmetric here.
MARKS_COMBINATIONS = {
    "C:10": ["N:20", "O:20", "C:20", "c:20", "n:20", "S:20"],
    "c:10": ["N:20", "O:20", "C:20", "c:20", "n:20", "S:20"],
    "c:20": ["N:11", "C:10", "c:10"],
    "C:20": ["C:10", "c:10"],
    "c:21": ["N:20", "O:20", "n:20"],
    "C:21": ["N:20", "n:20"],
    "N:20": ["C:10", "c:10", "C:21", "c:21", "S:10"],
    "N:11": ["c:20"],
    "n:20": ["C:10", "c:10", "C:21", "c:21"],
    "O:20": ["C:10", "c:10", "c:21"],
    "S:20": ["C:10", "c:10"],
    "S:10": ["N:20"],
    "C:30": ["C:40", "N:40"],
    "C:40": ["C:30"],
    "C:50": ["C:50"],
    "C:70": ["C:60", "c:60"],
    "c:60": ["C:70"],
    "C:60": ["C:70"],
    "N:40": ["C:30"],
}
# F7: c:70 is produced by BB synthonisation (Boronics_BF3andMIDA, Bifunctional, SulfonesSulfinates)
# but has no upstream row, so any stocked aryl-BF3/MIDA/aryl-sulfinate synthon raised KeyError.
F7_ADDITIONS = {"c:70": ["C:60", "c:60"]}
# F18: the Suzuki pairing. Upstream's table is only a whole-molecule pre-filter for its seed walk -
# the bond is formed by each rule's ReconstructionReaction - so the table never had to list the
# boronate partner. Here the table IS the join, so without these rows R12.1/12.2/12.6 disconnect a
# biaryl and then reassemble to nothing. Read straight off those SMIRKS ([#23] = code 10,
# [#108] = code 21); note that none of them licenses C:10 + c:21, so neither do we. The plan's own
# Appendix A states this pairing ("in Suzuki the boronate is the nucleophile") while Appendix C
# omits it - this row set resolves that contradiction in favour of Appendix A.
F18_ADDITIONS = {"c:10": ["c:21", "C:21"], "C:10": ["C:21"]}

# upstream forbiddenMarks (SyntOn.py). Four entries are dead — [c:70] x3 and [c:11] x1 are never
# emitted by any of the 39 rules — and {N:11, N:11} collapses to a one-element frozenset that bans
# every mono-functional umpolung-nitrogen synthon, killing R3.3 hierarchically (F5).
FORBIDDEN_MARKS = [
    {"N:11", "c:10"},
    {"N:11", "c:20"},
    {"N:11", "O:20"},
    {"N:11", "C:30"},
    {"N:11", "C:10"},
    {"N:11"},
    {"N:11", "c:70"},
    {"N:11", "C:40"},
    {"N:11", "C:50"},
    {"N:11", "S:10"},
    {"c:11", "C:20"},
    {"c:70", "c:20"},
    {"c:21", "C:60"},
    {"c:21", "N:11"},
    {"c:20", "c:21"},
    {"c:70", "C:21"},
    {"c:20", "C:21"},
]

# the leaving group a planner caps an attachment point with to recover an orderable reagent.
# Upstream has none: enumeration is synthon -> synthon -> molecule and the reagent is recovered by
# lookup in the synthon->BB file.
# ponytail: one representative LG per (element, aromatic, token); add a class-aware variant if a
# planner needs the exact stocked form.
LEAVING_GROUPS = {
    "C:elec": "Cl",
    "c:elec": "Br",
    "S:elec": "Cl",
    "N:elec": "OC(=O)c1ccccc1",
    "C:nuc": "[Mg]",
    "c:nuc": "[Mg]",
    "N:nuc": "H",
    "n:nuc": "H",
    "O:nuc": "H",
    "S:nuc": "H",
    "C:elecB": "B(O)O",
    "c:elecB": "B(O)O",
    "C:elec2": "O",
    "C:nuc2": "H",
    "N:nuc2": "H",
    "C:neut2": "C",
    "C:elec*": "H",
    "c:elec*": "H",
    "C:nuc*": "[B-](F)(F)F",
    "c:nuc*": "[B-](F)(F)F",
}

# dispatch selectors, read verbatim from __synthonsAssignement (SyntOn_BBs.py)
PG_BIFUNCTIONAL = [
    "Bifunctional_Acid_Ester",
    "Bifunctional_Acid_Nitro",
    "Bifunctional_Aldehyde_Ester",
    "Bifunctional_Amine_Ester",
    "Bifunctional_Ester_Isocyanates",
    "Bifunctional_Ester_SO2X",
    "Bifunctional_Aldehyde_Nitro",
    "Bifunctional_NbocAmino_Acid",
    "Bifunctional_NcbzAmino_Acid",
    "Bifunctional_Isothiocyanates_Acid",
    "Bifunctional_NfmocAmino_Acid",
    "Bifunctional_Aldehyde_Nboc",
    "Bifunctional_NTFAcAmino_Acid",
    "Bifunctional_Boronics_Ncbz",
    "Bifunctional_Boronics_Nfmoc",
    "Bifunctional_NbnDi_Amines",
    "Bifunctional_NbocDi_Amines",
    "Bifunctional_NcbzDi_Amines",
    "Bifunctional_NfmocDi_Amines",
    "Bifunctional_NTFAcDi_Amines",
    "Bifunctional_Di_Amines_NotherCarbamates",
    "Trifunctional_Acid_Aldehyde_Nitro",
    "Trifunctional_Acid_ArylHalide_Ester",
    "Trifunctional_Acid_ArylHalide_Nitro",
    "Trifunctional_Amines_ArylHalide_Nitro",
    "Trifunctional_NbocAmino_Acid_AlkyneCH",
    "Trifunctional_NbocAmino_Acid_ArylHalide",
    "Trifunctional_NfmocAmino_Acid_AlkyneCH",
    "Trifunctional_NfmocAmino_Acid_ArylHalide",
]
TWO_PG_TRIFUNCTIONAL = [
    "Trifunctional_Acid_Ester_Nitro",
    "Trifunctional_NbocAmino_Acid_Ester",
    "Trifunctional_NbocAmino_Acid_Nitro",
    "Trifunctional_Amines_Nboc_Ester",
    "Trifunctional_Nboc_NCbz_Amino_Acid",
    "Trifunctional_Nboc_Nfmoc_Amino_Acid",
    "Trifunctional_NfmocAmino_Acid_Ester",
    "Trifunctional_NfmocAmino_Acid_Nitro",
    "Trifunctional_Di_Esters_Amino",
]
FIRST_AS_PREP = [
    "Bifunctional_Acid_Aldehyde",
    "Bifunctional_Aldehyde_ArylHalide",
    "Bifunctional_Aldehyde_SO2X",
    "Bifunctional_Boronics_Acid",
    "Bifunctional_Boronics_Aldehyde",
    "Bifunctional_Hydroxy_Aldehyde",
    "Trifunctional_Acid_Aldehyde_ArylHalide",
    "Trifunctional_Acid_Aldehyde_Acetylenes",
    "Trifunctional_Acid_Aldehyde_Nitro",
    "Trifunctional_Amines_ArylHalide_Nitro",
    "Trifunctional_NbocAmino_Acid_AlkyneCH",
    "Trifunctional_NfmocAmino_Acid_AlkyneCH",
    "Trifunctional_Di_Esters_Amino",
]
POLYMER_REAGENTS = [
    "Reagents_PoliOxiranes",
    "Esters_PoliEsters",
    "Reagents_PoliIsocyanates",
    "SulfonylHalides_Poli_Sulfonylhalides",
]
ADDITIONAL_BIFUNCTIONAL = [
    "Aminoacids_N-AliphaticAmino_Acid",
    "Aminoacids_N-AromaticAmino_Acid",
    "Reagents_DiAmines",
]
# nitro -> amine is not really a protecting group, so the protected form is kept regardless of
# keepPG. Upstream compares this SMIRKS as an exact string twice; any whitespace change in the XML
# would have silently disabled it.
NITRO_REDUCTION = (
    "[N;+0,+1;$([N+](=O)([#6])[O-]),$(N(=O)([#6])=O):1](=[O:2])"
    "=,-[O;+0,-1:3]>>[NH2,+0:1]"
)

CLASS_KEYS = ("ShouldContainAtLeastOne", "ShouldAlsoContain", "shouldNotContain")

_BRANCH_STUB = re.compile(r"\([-=#:~]?(?:\[\*\]|\*)\)")
# the lookbehind matters: `_elec*` and `_nuc*` end in a star that is NOT a stub
_STUB = re.compile(r"[-=#:~]?(?:\[\*\]|(?<![A-Za-z])\*)")
_VANADIUM = re.compile(r"\[V\]")
# `*[V]c->c:10`: sigil, optional vanadium disambiguator, optional bond, element spelling, code
_LABEL_ALT = re.compile(
    r"^\*(?P<v>\[V\])?(?P<bond>=)?(?P<lhs>.+?)->(?P<iso>[0-9]*)(?P<el>[A-Za-z]+):(?P<code>[0-9]+)$"
)


class ConversionError(RuntimeError):
    """The upstream data said something this converter refuses to guess about."""


def _top_level_brackets(text: str) -> list[tuple[int, int]]:
    spans, depth, start = [], 0, None
    for i, c in enumerate(text):
        if c == "[":
            if not depth:
                start = i
            depth += 1
        elif c == "]":
            depth -= 1
            if not depth:
                spans.append((start, i))
    return spans


def _insert_token(template: str, atom_map: int, token: str) -> str:
    """Write `_token` into the bracket carrying `:atom_map`, just before the map."""
    suffix = f":{atom_map}"
    for start, end in _top_level_brackets(template):
        if template[start + 1 : end].endswith(suffix):
            cut = end - len(suffix)
            return f"{template[:cut]}_{token}{template[cut:]}"
    raise ConversionError(f"no bracket with map {atom_map} in {template!r}")


def _strip_stubs(template: str) -> str:
    """Drop the `[*]`/`*` attachment stubs and the `[V]` product-slot disambiguator (D15, W2/W4)."""
    return _STUB.sub("", _VANADIUM.sub("", _BRANCH_STUB.sub("", template)))


def _parse_label_options(labels: str) -> list[dict]:
    """A whole Labels string -> the distinct (element, aromatic, code, via_v, bond) it can assign.

    Isotope spellings collapse: they exist only because the reference matched the label as TEXT
    against a SMILES, and upstream's own enumerator KeyErrors on their output.
    """
    seen = set()
    for group in labels.split(";"):
        for alternative in group.split(","):
            match = _LABEL_ALT.match(alternative.strip())
            if match is None:
                raise ConversionError(f"unparsable Labels alternative {alternative!r}")
            seen.add(
                (
                    match["el"].upper(),
                    match["el"].islower(),
                    int(match["code"]),
                    bool(match["v"]),
                    2 if match["bond"] else 1,
                )
            )
    return [
        {"element": e, "aromatic": a, "code": c, "via_v": v, "bond": b}
        for e, a, c, v, b in sorted(seen)
    ]


def _fits(option: dict, target: dict) -> bool:
    """Aromaticity is deliberately NOT a filter here: the token does not encode it, the atom does,
    and a product template routinely spells an aromatic nitrogen `[N:2]`. It is a tie-breaker."""
    return (
        option["element"] == target["element"].upper()
        and option["via_v"] == target["via_v"]
        and option["bond"] == target["bond"]
    )


def _codes_for(options: list[dict], target: dict) -> list[int]:
    fits = [o for o in options if _fits(o, target)]
    codes = sorted({o["code"] for o in fits})
    if len(codes) > 1 and target["aromatic"] is not None:
        narrowed = sorted(
            {o["code"] for o in fits if o["aromatic"] == target["aromatic"]}
        )
        if narrowed:
            return narrowed
    return codes


def _component_targets(component: str) -> list[dict]:
    """Where each attachment stub of a product template sits: the mapped atom it labels."""
    query = smarts(to_chython(component))
    # an OR of mixed primitives translates to a mapped AnyElement, so "AnyElement" alone does not
    # identify a stub — an unmapped one does.
    mapped = {int(n) for n in re.findall(r":([1-9][0-9]*)\]", component)}
    targets = []
    for n, atom in query.atoms():
        if not isinstance(atom, AnyElement) or n in mapped:
            continue
        neighbours = list(query._bonds[n])
        if len(neighbours) != 1:
            raise ConversionError(
                f"stub with {len(neighbours)} neighbours in {component!r}"
            )
        (attached,) = neighbours
        bond = int(query._bonds[n][attached].order[0])
        via_v = query.atom(attached).atomic_symbol == "V"
        if via_v:
            rest = [m for m in query._bonds[attached] if m != n]
            if len(rest) != 1:
                raise ConversionError(
                    f"[V] with {len(rest)} real neighbours in {component!r}"
                )
            attached = rest[0]
        targets.append(
            {
                "map": attached,
                "element": query.atom(attached).atomic_symbol,
                "via_v": via_v,
                "bond": bond,
                "aromatic": _aromaticity(query)[attached],
            }
        )
    return targets


def _build_rule(
    reactant: str, product: str, labels: str | None, where: str
) -> list[str]:
    """One upstream (SMARTS, Labels) pair -> the rule SMARTS variants, tokens written inline."""
    left = to_chython(reactant)
    components = product.split(".")
    if labels in (None, "No"):
        return [f"{left}>>{_strip_stubs(to_chython(product))}"]

    options = _parse_label_options(labels)
    reactant_aromatic = _aromaticity(smarts(left))
    targets = []
    for index, component in enumerate(components):
        for target in _component_targets(component):
            if target["aromatic"] is None:
                target["aromatic"] = reactant_aromatic.get(target["map"])
            targets.append((index, target))

    # a stub matched by several distinct codes is several PRODUCTIVE labellings, not an ambiguity:
    # aniline emits both [NH2_nuc] and [NH2_nuc2].
    codes = []
    for _, target in targets:
        fits = _codes_for(options, target)
        if not fits:
            raise ConversionError(
                f"{where}: no Labels alternative fits the stub on atom {target['map']}"
            )
        codes.append(fits)
    branching = [i for i, c in enumerate(codes) if len(c) > 1]
    if len(branching) > 1:
        raise ConversionError(
            f"{where}: {len(branching)} stubs are each multiply labelled"
        )

    variants = []
    for choice in codes[branching[0]] if branching else [None]:
        assignment = {}
        for i, (_, target) in enumerate(targets):
            assignment[target["map"]] = choice if i in branching else codes[i][0]
        right = []
        for index, component in enumerate(components):
            # strip BEFORE stamping: the radical tokens end in a star of their own
            text = _strip_stubs(to_chython(component))
            for j, target in targets:
                if j == index:
                    text = _insert_token(
                        text,
                        target["map"],
                        PAPER_CODE_TO_LABEL[assignment[target["map"]]],
                    )
            right.append(text)
        variants.append(f"{left}>>{'.'.join(right)}")
    return variants


def _aromaticity(query) -> dict[int, bool | None]:
    out = {}
    for n, atom in query.atoms():
        hybridization = getattr(atom, "hybridization", ())
        out[n] = (
            True
            if hybridization == (4,)
            else (False if hybridization and 4 not in hybridization else None)
        )
    return out


def _split_label_sets(labels: str) -> list[str]:
    """`|` separates whole alternative labellings. Upstream keeps only the last (F8); we ship both."""
    return labels.split("|")


def convert_classes(path: Path) -> list[dict]:
    """SMARTSLibNew.json -> the ordered 147-class list. Order is load-bearing; never sort."""
    library = json.loads(path.read_text())
    out = []
    for big, subclasses in library.items():
        for sub, record in subclasses.items():
            unknown = set(record) - set(CLASS_KEYS)
            if unknown:
                raise ConversionError(f"{big}_{sub}: unknown keys {unknown}")
            out.append(
                {
                    "name": f"{big}_{sub}",
                    "at_least_one": [
                        to_chython(p) for p in record.get(CLASS_KEYS[0], ())
                    ],
                    "also": [to_chython(p) for p in record.get(CLASS_KEYS[1], ())],
                    "not": [to_chython(p) for p in record.get(CLASS_KEYS[2], ())],
                }
            )
    return out


def _strategy_and_func(name: str) -> tuple[str, int]:
    if name in TWO_PG_TRIFUNCTIONAL or "Trifunctional" in name:
        func = 3
    elif "Bifunctional" in name or name in ADDITIONAL_BIFUNCTIONAL:
        func = 2
    else:
        func = 1
    # the reference's own precedence chain
    if name in POLYMER_REAGENTS:
        strategy = "polymer"
    elif name in PG_BIFUNCTIONAL or name in TWO_PG_TRIFUNCTIONAL:
        strategy = "protecting_group"
    elif name in FIRST_AS_PREP:
        strategy = "first_as_prep"
    else:
        strategy = "normal"
    return strategy, func


def convert_marks(path: Path) -> list[dict]:
    """BB_Marks.xml -> 147 rule programs, one record per classifier subclass."""
    root = ElementTree.parse(path).getroot()
    out = []
    for big in root:
        for sub in big:
            if not sub.get("SMARTS"):
                continue
            name = f"{big.tag}_{sub.tag}"
            steps_smarts = sub.get("SMARTS").split("|")
            steps_labels = sub.get("Labels").split("|")
            if len(steps_smarts) != len(steps_labels):
                raise ConversionError(
                    f"{name}: {len(steps_smarts)} steps vs {len(steps_labels)} labels"
                )
            strategy, func = _strategy_and_func(name)
            steps = []
            for i, (step, label) in enumerate(zip(steps_smarts, steps_labels)):
                reactant, product = step.split(">>")
                variants = _build_rule(reactant, product, label, f"{name}[{i}]")
                steps.append(
                    {
                        "variants": variants,
                        # `|No|` is the section boundary and nothing else - upstream splits on it
                        # unconditionally (SyntOn_BBs.py `LabelsLIST.split("|No|")`).
                        "is_pg_removal": label == "No",
                        # nitro -> amine is a reduction, not a deprotection, so the "protected"
                        # form survives it at both keepPG values.
                        "keeps_protected": step == NITRO_REDUCTION,
                    }
                )
            out.append(
                {
                    "name": name,
                    "strategy": strategy,
                    "func": func,
                    "first_as_prep": name in FIRST_AS_PREP,
                    "steps": steps,
                }
            )
    return out


def _rule_records(path: Path, macro: bool) -> list[dict]:
    root = ElementTree.parse(path).getroot()
    out = []
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") == "None":  # the R14.1 / MR14.1 range sentinel
                continue
            reactant, product = rule.get("SMARTS").split(">>")
            label_sets = _split_label_sets(rule.get("Labels"))
            suffixes = (
                ("",) if len(label_sets) == 1 else tuple("abcdefg"[: len(label_sets)])
            )
            for suffix, labels in zip(suffixes, label_sets):
                variants = _build_rule(reactant, product, labels, f"{rule.tag}{suffix}")
                if len(variants) != 1:
                    raise ConversionError(
                        f"{rule.tag}: {len(variants)} labelling variants"
                    )
                out.append(
                    {
                        "id": f"{rule.tag}{suffix}",
                        "name": rule.get("name") or group.get("name"),
                        "macro": macro,
                        "smarts": variants[0],
                        "single_product": macro,
                    }
                )
    return out


def _macro_reactants(path: Path) -> dict[str, str]:
    root = ElementTree.parse(path).getroot()
    out = {}
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") != "None":  # MR14.1 is the range sentinel
                out[rule.tag] = rule.get("SMARTS")
    return out


def _macro_twins(rules: list[dict], path: Path) -> list[dict]:
    """The MR set: the upstream macrocyclic REACTANT with the R rule's labelled product.

    The reactant is authored — it says which bond is the ring bond and carries the `!r3..!r11`
    guard, which the fork now expresses as a real excluded-ring-sizes field. The product comes from
    the R twin, which is why the macro file's own defects (aliphatic C mapped onto aromatic c in
    MR10.1/10.2/12.2, the 60/70 swap in MR13.1) are not inherited.
    """
    upstream = _macro_reactants(path)
    out = []
    for rule in rules:
        base = rule["id"][:-1] if rule["id"][-1].isalpha() else rule["id"]
        raw = upstream.get(f"M{base}")
        if raw is None:
            raise ConversionError(f"no macrocyclic twin for {rule['id']}")
        left = to_chython(raw.split(">>")[0])
        right = rule["smarts"].split(">>", 1)[1]
        reactant_maps = set(smarts(left))
        product_maps = {n for n, _ in smarts(right).atoms()}
        if not product_maps <= reactant_maps:
            raise ConversionError(
                f"M{rule['id']}: product maps {product_maps - reactant_maps} "
                "are absent from the macrocyclic reactant"
            )
        out.append(
            {
                "id": f"M{rule['id']}",
                "name": f"{rule['name']} (macrocyclic)",
                "macro": True,
                "smarts": f"{left}>>{right}",
                "single_product": True,
            }
        )
    return out


def _partner_pairs() -> list[list]:
    combos = {k: list(v) for k, v in MARKS_COMBINATIONS.items()}
    for additions in (F7_ADDITIONS, F18_ADDITIONS):
        for key, partners in additions.items():
            combos.setdefault(key, []).extend(partners)
    for key, partners in list(combos.items()):
        for partner in partners:
            if key not in combos.setdefault(partner, []):
                combos[partner].append(key)
    pairs = set()
    for key, partners in combos.items():
        for partner in partners:
            pairs.add(tuple(sorted((_migrate_key(key), _migrate_key(partner)))))
    return sorted([list(a) + list(b) for a, b in pairs])


def _migrate_key(key: str) -> tuple[str, bool, str]:
    element, code = key.split(":")
    return element.upper(), element.islower(), PAPER_CODE_TO_LABEL[int(code)]


def _upstream_produced(path: Path) -> set[str]:
    """The `element:code` keys the disconnection rules actually emit, in upstream spelling.

    Deadness has to be decided BEFORE the migration: collapsing code 11 into `elec` would
    otherwise turn upstream's never-emitted `c:11` into a live `c:elec` entry and ban a
    perfectly ordinary aryl-electrophile / alkyl-nucleophile fragment.
    """
    root = ElementTree.parse(path).getroot()
    keys = set()
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") == "None":
                continue
            for label_set in _split_label_sets(rule.get("Labels")):
                for option in _parse_label_options(label_set):
                    symbol = (
                        option["element"].lower()
                        if option["aromatic"]
                        else option["element"]
                    )
                    keys.add(f"{symbol}:{option['code']}")
    return keys


def _forbidden_marks(emitted: set[str]) -> list[list]:
    """The 12 live entries, keyed on tokens. Four are dead upstream and one collapses (F5)."""
    out = []
    for entry in FORBIDDEN_MARKS:
        if (
            len(entry) == 1
        ):  # F5: {N:11, N:11} collapses and bans every mono-N:elec synthon
            continue
        if any(k not in emitted for k in entry):
            continue
        out.append(sorted([list(_migrate_key(k)) for k in entry]))
    return sorted(out)


def build(config_dir: Path) -> dict[str, object]:
    classes = convert_classes(config_dir / "SMARTSLibNew.json")
    marks = convert_marks(config_dir / "BB_Marks.xml")
    disconnections = _rule_records(config_dir / "Setup.xml", macro=False)
    rules = {
        "disconnections": disconnections
        + _macro_twins(disconnections, config_dir / "SetupForMacrocycles.xml"),
        "pairs": _partner_pairs(),
        "leaving_groups": LEAVING_GROUPS,
        "forbidden_marks": _forbidden_marks(
            _upstream_produced(config_dir / "Setup.xml")
        ),
    }
    return {"bb_classes": classes, "bb_marks": marks, "rules": rules}


def check(built: dict, config_dir: Path) -> list[str]:
    """Assert every property the plan pins. Returns the failures, empty when the build is good."""
    from synplan.chem.synthon.reactor import SynthonTransformer, query_labels

    problems = []
    classes, marks, rules = built["bb_classes"], built["bb_marks"], built["rules"]

    if len(classes) != 147:
        problems.append(f"{len(classes)} classes, expected 147")
    patterns = sum(len(c[k]) for c in classes for k in ("at_least_one", "also", "not"))
    if patterns != 2401:
        problems.append(f"{patterns} classifier patterns, expected 2401")
    for record in classes:
        for key in ("at_least_one", "also", "not"):
            for pattern in record[key]:
                try:
                    smarts(pattern)
                except Exception as exc:
                    problems.append(f"{record['name']}/{key}: {pattern!r} {exc}")

    if len(marks) != 147:
        problems.append(f"{len(marks)} rule programs, expected 147")
    if {c["name"] for c in classes} != {m["name"] for m in marks}:
        problems.append("bb_classes and bb_marks names are not in 1:1 correspondence")
    steps = sum(len(m["steps"]) for m in marks)
    if steps != 389:
        problems.append(f"{steps} rule steps, expected 389")
    strategies = {"polymer": 0, "protecting_group": 0, "first_as_prep": 0, "normal": 0}
    for record in marks:
        strategies[record["strategy"]] += 1
    if strategies != {
        "polymer": 4,
        "protecting_group": 38,
        "first_as_prep": 8,
        "normal": 97,
    }:
        problems.append(f"strategy dispatch counts {strategies}")
    # a protecting_group program with no boundary loses its whole deprotection stage silently
    sectionless = [
        m["name"]
        for m in marks
        if m["strategy"] == "protecting_group"
        and not any(s["is_pg_removal"] for s in m["steps"])
    ]
    if sectionless:
        problems.append(
            f"protecting-group programs with no `No` boundary: {sectionless}"
        )

    disconnections = [r for r in rules["disconnections"] if not r["macro"]]
    macro = [r for r in rules["disconnections"] if r["macro"]]
    if len(disconnections) != 39:
        problems.append(
            f"{len(disconnections)} disconnection rules, expected 39 (R12.3 splits)"
        )
    if len(macro) != 39:
        problems.append(f"{len(macro)} macro twins, expected 39")

    for record in rules["disconnections"] + [s for m in marks for s in m["steps"]]:
        for text in [record["smarts"]] if "smarts" in record else record["variants"]:
            left, _right = text.split(">>")
            try:
                if query_labels(smarts(left)):
                    problems.append(f"reactant-side token in {text!r}")
                transformer = SynthonTransformer.from_smarts(text)
            except Exception as exc:
                problems.append(f"{text!r} {type(exc).__name__}: {exc}")
                continue
            for token in transformer._synthon_labels.values():
                if token not in PAPER_CODE_TO_LABEL.values():
                    problems.append(f"{text!r} emits unknown token {token!r}")
            if '"slots"' in json.dumps(record):
                problems.append(f"{text!r} carries a slots key")

    if len(rules["pairs"]) != 29:
        problems.append(
            f"{len(rules['pairs'])} partner pairs, expected 29 (24 upstream + 2 for F7 + 3 for F18)"
        )
    if len(rules["forbidden_marks"]) != 12:
        problems.append(
            f"{len(rules['forbidden_marks'])} forbidden-mark entries, expected 12"
        )

    problems.extend(_macro_derivation_diff(config_dir, macro))
    return problems


def _macro_derivation_diff(config_dir: Path, generated: list[dict]) -> list[str]:
    """Build each MR rule from the macro file alone and diff it against the R twin's labelling."""
    notes = []
    root = ElementTree.parse(config_dir / "SetupForMacrocycles.xml").getroot()
    upstream = {
        r.tag: r
        for g in root.find("AvailableReactions")
        for r in g
        if r.get("SMARTS") != "None"
    }
    for record in generated:
        base = record["id"][:-1] if record["id"][-1].isalpha() else record["id"]
        rule = upstream.get(base)
        if rule is None:
            notes.append(f"note: {base} absent from SetupForMacrocycles.xml")
            continue
        reactant, product = rule.get("SMARTS").split(">>")
        if product.startswith("(") and product.endswith(")"):
            # the macro products are grouped to mean "one fragment"; chython parses the group
            # identically to the ungrouped form, so `single_product` carries that intent instead
            product = product[1:-1]
        try:
            theirs = _build_rule(
                reactant, product, _split_label_sets(rule.get("Labels"))[-1], base
            )
        except (ConversionError, DialectError, Exception) as exc:
            notes.append(
                f"note: {base} does not build from the macro file ({type(exc).__name__}: {exc})"
            )
            continue
        ours = query_labels_of(record["smarts"].split(">>", 1)[1])
        if len(theirs) == 1 and query_labels_of(theirs[0].split(">>", 1)[1]) != ours:
            notes.append(
                f"note: {base} upstream assigns {query_labels_of(theirs[0].split('>>', 1)[1])} "
                f"where the R twin assigns {ours} - taking the R twin"
            )
    return notes


def query_labels_of(template: str) -> dict[int, str]:
    return {
        n: a._label
        for n, a in smarts(template).atoms()
        if getattr(a, "_label", None) is not None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", help="Synt-On's config/ directory")
    parser.add_argument("--out", required=True, help="where the three JSON files go")
    parser.add_argument(
        "--check", action="store_true", help="verify the build before writing"
    )
    args = parser.parse_args(argv)

    config_dir, out = Path(args.config_dir), Path(args.out)
    built = build(config_dir)
    if args.check:
        problems = [p for p in check(built, config_dir) if not p.startswith("note:")]
        notes = [p for p in check(built, config_dir) if p.startswith("note:")]
        for note in notes:
            print(note)
        if problems:
            for problem in problems:
                print(f"FAIL {problem}", file=sys.stderr)
            return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("bb_classes", built["bb_classes"]),
        ("bb_marks", built["bb_marks"]),
        ("rules", built["rules"]),
    ):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"wrote {len(built['bb_classes'])} classes, {len(built['bb_marks'])} rule programs, "
        f"{len(built['rules']['disconnections'])} disconnection rules to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
