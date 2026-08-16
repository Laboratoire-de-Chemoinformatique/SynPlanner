"""The disconnection DAG: 39 rules cut a target into synthons, level by level, with the gates."""

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from chython import smiles, synthon_smiles
from chython.containers import MoleculeContainer, SynthonContainer

from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.synthon.reactor import SynthonTransformer
from synplan.chem.utils import safe_canonicalization

# the study list from the paper's own worked examples, H-capped: upstream spells each of these
# twice because `[V]` counts as a heavy atom and `*` does not. Off by default.
STUDY_FRAGMENTS_TO_IGNORE = ("CCC", "CC=O", "C=O", "CC(C)=O")

# a target with any SSSR ring larger than this is cut at level 1 by the macrocyclic rules ONLY
MACROCYCLE_RING = 11

_RANGE = re.compile(r"^(M?R)(\d+)(?:\.(\d+)([ab])?)?$")
# neutralise carboxylates before the first cut, exactly as the reference does
_CARBOXYLATE = "[O;-1;D1;$([O-][C]=[O]):1]>>[OH;+0:1]"


@dataclass(frozen=True, slots=True)
class Pathway:
    """A reagent SET, not a route: no step order, no intermediates, no yields."""

    key: tuple[str, ...]
    rules: tuple[str, ...]
    depth: int
    availability: float = 0.0


@dataclass
class DisconnectionDAG:
    target: str
    pathways: dict[tuple[str, ...], Pathway] = field(default_factory=dict)
    children: dict[tuple[str, ...], set[tuple[str, ...]]] = field(default_factory=dict)
    parents: dict[tuple[str, ...], set[tuple[str, ...]]] = field(default_factory=dict)

    def roots(self) -> list[Pathway]:
        return [p for p in self.pathways.values() if p.depth == 1]

    def leaves(self) -> list[Pathway]:
        return [p for p in self.pathways.values() if not self.children.get(p.key)]

    def best_available(self) -> list[Pathway]:
        """Most available first; at equal availability the FEWEST reagents wins.

        Upstream sorts `(availabilityRate, reagentsNumber)` with reverse=True, which returns the
        pathway with the most reagents — the commented-out line above it has the sign right.
        """
        return sorted(
            self.pathways.values(), key=lambda p: (-p.availability, len(p.key))
        )

    def is_acyclic(self) -> bool:
        seen, stack = set(), []

        def walk(node):
            if node in stack:
                return False
            if node in seen:
                return True
            seen.add(node)
            stack.append(node)
            ok = all(walk(child) for child in self.children.get(node, ()))
            stack.pop()
            return ok

        return all(walk(key) for key in self.pathways)


def _positions(rules: list[dict], token: str) -> list[int]:
    """Where one selector lands in the ORDERED rule list. `R12.3` covers `R12.3a` and `R12.3b`."""
    match = _RANGE.match(token.strip())
    if not match:
        raise ValueError(f"unparsable rule selector {token!r}")
    prefix, family, sub, suffix = match.groups()
    found = [
        index
        for index, rule in enumerate(rules)
        if (parts := _RANGE.match(rule["id"]))
        and parts.group(1) == prefix
        and parts.group(2) == family
        and (sub is None or parts.group(3) == sub)
        and (suffix is None or parts.group(4) == suffix)
    ]
    if not found:
        raise ValueError(f"no rule matches {token!r}")
    return found


def _select(rules: list[dict], mode: str, selection: str) -> list[dict]:
    """`Rn`, `Rn.m`, `Rn.ma`, `Rn-Rm`, comma-separated. A bare `Rn` covers the whole R`n` family.

    A range is a slice of the ordered list, so `R1.2-R1.4` excludes `R1.1`. A reversed range and a
    selector that matches no rule are errors, never a silently empty run.
    """
    if mode == "use_all":
        return rules
    wanted: set[int] = set()
    for token in selection.split(","):
        if "-" in token:
            low, high = (_positions(rules, part) for part in token.split("-", 1))
            if min(low) > max(high):
                raise ValueError(f"reversed rule range {token!r}")
            wanted.update(range(min(low), max(high) + 1))
            continue
        wanted.update(_positions(rules, token))
    if mode == "exclude_some":
        return [r for index, r in enumerate(rules) if index not in wanted]
    return [r for index, r in enumerate(rules) if index in wanted]


def _reparse(smi: str) -> SynthonContainer:
    """chython does not aromatise on parse, so every re-read synthon must be canonicalised."""
    return safe_canonicalization(synthon_smiles(smi))


def _labels(molecule: SynthonContainer) -> Counter:
    return Counter(molecule.synthon_labels.values())


class Fragmenter:
    """BFS by level with one memo dict. Upstream recurses with a raised stack limit and re-derives
    each depth-`d` node once per permutation of its cuts."""

    def __init__(
        self,
        config: SynthonConfig | None = None,
        stock: dict[str, set[str]] | None = None,
    ) -> None:
        self.config = config or SynthonConfig()
        self.stock = stock or {}
        data = load_data(self.config.rules_path)
        normal = [r for r in data["disconnections"] if not r["macro"]]
        macro = [r for r in data["disconnections"] if r["macro"]]
        self.rules = [
            (r, SynthonTransformer.from_smarts(r["smarts"]))
            for r in _select(normal, self.config.rule_mode, self.config.rules_selection)
        ]
        # the same selection, translated onto the MR ids exactly as `__getMacroCycleSetup` does
        self.macro_rules = [
            (r, SynthonTransformer.from_smarts(r["smarts"]))
            for r in _select(
                macro,
                self.config.rule_mode,
                re.sub(r"M?R", "MR", self.config.rules_selection),
            )
        ]
        self.forbidden = {
            frozenset(tuple(k) for k in entry) for entry in data["forbidden_marks"]
        }
        # the gate compares canonical SMILES, so a user's own spelling must be canonicalised too
        self.ignore = frozenset(
            str(safe_canonicalization(smiles(s)))
            for s in self.config.fragments_to_ignore
        )
        self._carboxylate = SynthonTransformer.from_smarts(_CARBOXYLATE)
        self._memo: dict[tuple[str, str], tuple[tuple[str, tuple[str, ...]], ...]] = {}
        self._rule_index: dict[str, int] = {}

    # --- gates ---------------------------------------------------------------------------

    def _accept(
        self,
        products: list[SynthonContainer],
        deep: bool,
        parent_labels: Counter | None,
    ) -> bool:
        """Every gate drops the WHOLE product set, never just the offending fragment."""
        produced = Counter()
        for product in products:
            heavy = len(product)
            # the floor counts attachment points as atoms. Upstream materialises each one as a
            # real marker element, so RDKit counts it and a hydroxymethyl synthon clears the
            # floor at 3; deleting the fake marker would otherwise silently drop every smallest
            # fragment the six marker-carrying rules produce. The density caps below stay on
            # real atoms, which is where upstream measures them too.
            attachments = sum(
                a.attachment_points
                for _, a in product.atoms()
                if getattr(a, "_label", None) is not None
            )
            if heavy + attachments < 3 or str(product.unlabelled()) in self.ignore:
                return False
            marks = product.synthon_labels
            count = len(marks)
            if count > self.config.max_rc_per_fragment:
                return False
            rings = len(product.sssr)
            if rings == 0 and count > (heavy + 1) / 3:
                return False
            if rings != 0 and count > (heavy + 1) / 2:
                return False
            if deep:
                keys = frozenset(
                    (a.atomic_symbol, a.hybridization == 4, a.label)
                    for _, a in product.atoms()
                    if getattr(a, "_label", None) is not None
                )
                if keys in self.forbidden:
                    return False
            produced.update(marks.values())
        if deep and parent_labels is not None:
            # a hierarchical cut may ADD labels; it may not destroy or mutate one
            if any(produced[label] < n for label, n in parent_labels.items()):
                return False
        return True

    # --- cutting -------------------------------------------------------------------------

    def _cut(
        self, molecule: SynthonContainer, rules
    ) -> Iterator[tuple[int, str, tuple[str, ...], tuple]]:
        """Yields the rule's POSITION too: `one_by_one` searches the list by index, not by id."""
        for index, (record, rule) in enumerate(rules):
            key = (str(molecule), record["id"])
            cached = self._memo.get(key)
            if cached is None:
                sets = []
                for index, transformed in enumerate(rule(molecule)):
                    parts = (
                        [transformed]
                        if record["single_product"]
                        else transformed.split()
                    )
                    # canonicalise before stringifying: the stored string IS the identity, and a
                    # non-canonical one makes the same synthon two dict keys and cuts differently
                    parts = [safe_canonicalization(p) for p in parts]
                    sets.append(
                        (
                            f"{record['id']}_{index}",
                            tuple(str(p) for p in parts),
                            tuple(parts),
                        )
                    )
                self._memo[key] = tuple((name, smis) for name, smis, _ in sets)
                for name, smis, parts in sets:
                    yield index, name, smis, parts
                continue
            for name, smis in cached:
                yield index, name, smis, tuple(_reparse(s) for s in smis)

    def fragment(self, target: MoleculeContainer) -> DisconnectionDAG:
        self._memo.clear()
        self._rule_index.clear()
        prepared = self._prepare(target)
        dag = DisconnectionDAG(target=str(prepared))
        root: tuple[str, ...] = ()
        dag.children[root] = set()

        stepwise = self.config.rule_mode == "one_by_one"
        macro = any(len(ring) > MACROCYCLE_RING for ring in prepared.sssr)
        level_rules = self.macro_rules if macro else self.rules
        frontier: list[Pathway] = []
        opened: int | None = None
        for index, name, smis, parts in self._cut(prepared, level_rules):
            if len(dag.pathways) >= self.config.max_pathways:
                break
            # upstream leaves level 1 after the first rule that MATCHED, accepted or not
            if stepwise:
                if opened is None:
                    opened = index
                elif index != opened:
                    break
            if not self._accept(list(parts), deep=False, parent_labels=None):
                continue
            if stepwise:
                self._rule_index.update(dict.fromkeys(smis, index))
            key = tuple(sorted(smis))
            if key in dag.pathways:
                dag.parents.setdefault(key, set()).add(root)
                dag.children[root].add(key)
                continue
            pathway = Pathway(key=key, rules=(name,), depth=1)
            dag.pathways[key] = pathway
            dag.children[root].add(key)
            dag.parents.setdefault(key, set()).add(root)
            frontier.append(pathway)

        depth = 1
        while frontier and depth < self.config.max_stages:
            if len(dag.pathways) >= self.config.max_pathways:
                break
            nxt: list[Pathway] = []
            for pathway in frontier:
                # the innermost break alone leaves two enclosing loops running, which overshot
                # the cap by about 2x; the bound has to be checked at every level that adds
                if len(dag.pathways) >= self.config.max_pathways:
                    break
                for position, smi in enumerate(pathway.key):
                    if len(dag.pathways) >= self.config.max_pathways:
                        break
                    parent = _reparse(smi)
                    parent_labels = _labels(parent)
                    floor = self._rule_index.get(smi, 0)
                    cut_at: int | None = None
                    for index, name, smis, parts in self._cut(parent, self.rules):
                        # monotone and non-backtracking: never look below the rule that made this
                        # synthon, and stop at the first rule that cuts it
                        if stepwise:
                            if index < floor:
                                continue
                            if cut_at is not None and index != cut_at:
                                break
                        if not self._accept(
                            list(parts), deep=True, parent_labels=parent_labels
                        ):
                            continue
                        if stepwise:
                            cut_at = index
                            self._rule_index.update(dict.fromkeys(smis, index))
                        rest = pathway.key[:position] + pathway.key[position + 1 :]
                        key = tuple(sorted(rest + smis))
                        if key in dag.pathways:
                            dag.parents.setdefault(key, set()).add(pathway.key)
                            dag.children.setdefault(pathway.key, set()).add(key)
                            continue
                        child = Pathway(
                            key=key, rules=(*pathway.rules, name), depth=depth + 1
                        )
                        dag.pathways[key] = child
                        dag.children.setdefault(pathway.key, set()).add(key)
                        dag.parents.setdefault(key, set()).add(pathway.key)
                        nxt.append(child)
                        if len(dag.pathways) >= self.config.max_pathways:
                            break
            frontier = nxt
            depth += 1

        self._score(dag, prepared)
        return dag

    @staticmethod
    def _parent(target: MoleculeContainer) -> MoleculeContainer:
        """A salt or hydrate is cut as its parent molecule: a counter-ion is not a reagent."""
        parts = target.split()
        if len(parts) == 1:
            return target
        parts.sort(key=len, reverse=True)
        # ponytail: size alone decides, so a 1:1 salt whose counter-ion is over half the parent
        # raises instead of guessing. Upgrade path: share synthonise.py's counter-ion list.
        if len(parts[1]) * 2 >= len(parts[0]):
            raise ValueError(
                f"ambiguous multi-component target {target}: fragment one component per line"
            )
        return parts[0]

    def _prepare(self, target: MoleculeContainer) -> SynthonContainer:
        """Drop the counter-ion and stereo, neutralise carboxylates, canonicalise. chython does not
        aromatise on parse, so 15 of the rule LHS would silently never match without this."""
        prepared = synthon_smiles(str(self._parent(target)))
        prepared.clean_stereo()
        for neutral in self._carboxylate(prepared):
            prepared = neutral
            break
        return safe_canonicalization(prepared)

    def _score(self, dag: DisconnectionDAG, target: SynthonContainer) -> None:
        denominator = len(target)
        for key, pathway in list(dag.pathways.items()):
            parts = [_reparse(s) for s in key]
            covered = sum(len(p) for p, s in zip(parts, key) if s in self.stock)
            total = (
                denominator
                if self.config.availability_denominator == "target"
                else sum(len(p) for p in parts)
            )
            dag.pathways[key] = Pathway(
                key=pathway.key,
                rules=pathway.rules,
                depth=pathway.depth,
                availability=covered / total if total else 0.0,
            )


def fragment_smiles(
    smi: str,
    config: SynthonConfig | None = None,
    stock: dict[str, set[str]] | None = None,
) -> DisconnectionDAG:
    return Fragmenter(config, stock).fragment(safe_canonicalization(smiles(smi)))


def iter_pathways(dag: DisconnectionDAG) -> Iterable[Pathway]:
    return dag.best_available()


__all__ = [
    "MACROCYCLE_RING",
    "STUDY_FRAGMENTS_TO_IGNORE",
    "DisconnectionDAG",
    "Fragmenter",
    "Pathway",
    "fragment_smiles",
    "iter_pathways",
]
